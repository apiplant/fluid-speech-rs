//! fp16 GEMMs through cuBLASLt with per-shape algorithm autotuning.
//!
//! Every distinct problem (shape, leading dimensions, batch strides) is timed once against up to `CANDIDATES`
//! heuristic algorithms plus plain `cublasGemmEx`; the winner is cached and reused (also inside CUDA graphs).

use anyhow::Result;
use candle_core::cuda_backend::cudarc::cublas::{result as blas, sys as bsys};
use candle_core::cuda_backend::cudarc::cublaslt::{result as lt, sys as lsys};
use candle_core::cuda_backend::cudarc::driver::{sys, CudaStream};
use half::f16;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

const CANDIDATES: usize = 16;
const TIMING_RUNS: usize = 5;

/// Column-major problem: C[m, n] = op(A)[m, k] . B[k, n], A transposed when `ta`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Problem {
    pub ta: bool,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    pub batch: usize,
}

#[derive(Clone, Copy)]
enum Choice {
    GemmEx,
    Lt(lsys::cublasLtMatmulAlgo_t),
}

pub struct Gemm {
    blas: bsys::cublasHandle_t,
    lt: lsys::cublasLtHandle_t,
    stream: Arc<CudaStream>,
    ws: u64,
    ws_bytes: usize,
    fp16_acc: bool,
    pub tune: bool,
    cache: Mutex<HashMap<Problem, Choice>>,
}

unsafe impl Send for Gemm {}
unsafe impl Sync for Gemm {}

struct Desc {
    op: lsys::cublasLtMatmulDesc_t,
    a: lsys::cublasLtMatrixLayout_t,
    b: lsys::cublasLtMatrixLayout_t,
    c: lsys::cublasLtMatrixLayout_t,
}

impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.a);
            let _ = lt::destroy_matrix_layout(self.b);
            let _ = lt::destroy_matrix_layout(self.c);
            let _ = lt::destroy_matmul_desc(self.op);
        }
    }
}

impl Gemm {
    /// `ws`/`ws_bytes`: device workspace shared by cuBLAS and cuBLASLt (must outlive this object).
    pub fn new(stream: Arc<CudaStream>, ws: u64, ws_bytes: usize, fp16_acc: bool, tune: bool) -> Result<Self> {
        let blas_h = blas::create_handle()?;
        unsafe {
            blas::set_stream(blas_h, stream.cu_stream() as _)?;
            bsys::cublasSetWorkspace_v2(blas_h, ws as *mut c_void, ws_bytes).result()?;
        }
        let lt_h = lt::create_handle()?;
        Ok(Self { blas: blas_h, lt: lt_h, stream, ws, ws_bytes, fp16_acc, tune, cache: Mutex::new(HashMap::new()) })
    }

    fn compute(&self) -> (lsys::cublasComputeType_t, lsys::cudaDataType_t) {
        if self.fp16_acc {
            (lsys::cublasComputeType_t::CUBLAS_COMPUTE_16F, lsys::cudaDataType_t::CUDA_R_16F)
        } else {
            (lsys::cublasComputeType_t::CUBLAS_COMPUTE_32F, lsys::cudaDataType_t::CUDA_R_32F)
        }
    }

    fn desc(&self, p: &Problem) -> Result<Desc> {
        let (ct, st) = self.compute();
        let h = lsys::cudaDataType_t::CUDA_R_16F;
        let op = lt::create_matmul_desc(ct, st)?;
        let ta = if p.ta { bsys::cublasOperation_t::CUBLAS_OP_T } else { bsys::cublasOperation_t::CUBLAS_OP_N };
        unsafe {
            lt::set_matmul_desc_attribute(
                op,
                lsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &ta as *const _ as *const c_void,
                std::mem::size_of_val(&ta),
            )?;
        }
        let (ar, ac) = if p.ta { (p.k, p.m) } else { (p.m, p.k) };
        let a = lt::create_matrix_layout(h, ar as u64, ac as u64, p.lda as i64)?;
        let b = lt::create_matrix_layout(h, p.k as u64, p.n as u64, p.ldb as i64)?;
        let c = lt::create_matrix_layout(h, p.m as u64, p.n as u64, p.ldc as i64)?;
        if p.batch > 1 {
            for (l, s) in [(a, p.sa), (b, p.sb), (c, p.sc)] {
                let n = p.batch as i32;
                let s = s as i64;
                unsafe {
                    lt::set_matrix_layout_attribute(
                        l,
                        lsys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                        &n as *const _ as *const c_void,
                        4,
                    )?;
                    lt::set_matrix_layout_attribute(
                        l,
                        lsys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                        &s as *const _ as *const c_void,
                        8,
                    )?;
                }
            }
        }
        Ok(Desc { op, a, b, c })
    }

    fn scalars(&self) -> ([u8; 4], [u8; 4]) {
        if self.fp16_acc {
            let (o, z) = (f16::ONE.to_bits().to_le_bytes(), f16::ZERO.to_bits().to_le_bytes());
            ([o[0], o[1], 0, 0], [z[0], z[1], 0, 0])
        } else {
            (1f32.to_le_bytes(), 0f32.to_le_bytes())
        }
    }

    fn run(&self, p: &Problem, choice: &Choice, a: u64, b: u64, c: u64) -> Result<()> {
        let (alpha, beta) = self.scalars();
        let (al, be) = (alpha.as_ptr() as *const c_void, beta.as_ptr() as *const c_void);
        match choice {
            Choice::GemmEx => {
                let f16t = bsys::cudaDataType_t::CUDA_R_16F;
                let ct = if self.fp16_acc {
                    bsys::cublasComputeType_t::CUBLAS_COMPUTE_16F
                } else {
                    bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F
                };
                let opa = if p.ta { bsys::cublasOperation_t::CUBLAS_OP_T } else { bsys::cublasOperation_t::CUBLAS_OP_N };
                let opb = bsys::cublasOperation_t::CUBLAS_OP_N;
                let algo = bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                unsafe {
                    if p.batch == 1 {
                        blas::gemm_ex(
                            self.blas, opa, opb, p.m as i32, p.n as i32, p.k as i32, al, a as *const c_void, f16t,
                            p.lda as i32, b as *const c_void, f16t, p.ldb as i32, be, c as *mut c_void, f16t,
                            p.ldc as i32, ct, algo,
                        )?;
                    } else {
                        blas::gemm_strided_batched_ex(
                            self.blas, opa, opb, p.m as i32, p.n as i32, p.k as i32, al, a as *const c_void, f16t,
                            p.lda as i32, p.sa as i64, b as *const c_void, f16t, p.ldb as i32, p.sb as i64, be,
                            c as *mut c_void, f16t, p.ldc as i32, p.sc as i64, p.batch as i32, ct, algo,
                        )?;
                    }
                }
            }
            Choice::Lt(algo) => {
                let d = self.desc(p)?;
                unsafe {
                    lt::matmul(
                        self.lt, d.op, al, be, a as *const c_void, d.a, b as *const c_void, d.b, c as *const c_void,
                        d.c, c as *mut c_void, d.c, algo, self.ws as *mut c_void, self.ws_bytes,
                        self.stream.cu_stream() as _,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn candidates(&self, p: &Problem) -> Result<Vec<lsys::cublasLtMatmulAlgo_t>> {
        let d = self.desc(p)?;
        let pref = lt::create_matmul_pref()?;
        let ws = self.ws_bytes;
        let mut res: Vec<lsys::cublasLtMatmulHeuristicResult_t> = Vec::with_capacity(CANDIDATES);
        let mut count = 0i32;
        unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                lsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const _ as *const c_void,
                8,
            )?;
            let st = lsys::cublasLtMatmulAlgoGetHeuristic(
                self.lt, d.op, d.a, d.b, d.c, d.c, pref, CANDIDATES as i32, res.as_mut_ptr(), &mut count,
            );
            let _ = lt::destroy_matmul_pref(pref);
            if st != lsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Ok(vec![]);
            }
            res.set_len(count.max(0) as usize);
        }
        Ok(res.into_iter().filter(|r| r.state == lsys::cublasStatus_t::CUBLAS_STATUS_SUCCESS).map(|r| r.algo).collect())
    }

    fn time(&self, p: &Problem, choice: &Choice, a: u64, b: u64, c: u64) -> Result<f32> {
        let ctx = self.stream.context();
        let flags = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let (e0, e1) = (ctx.new_event(flags)?, ctx.new_event(flags)?);
        self.run(p, choice, a, b, c)?; // warm-up
        e0.record(&self.stream)?;
        for _ in 0..TIMING_RUNS {
            self.run(p, choice, a, b, c)?;
        }
        e1.record(&self.stream)?;
        e1.synchronize()?;
        Ok(e0.elapsed_ms(&e1)?)
    }

    /// Enqueue C = op(A) B. When tuning is on and the problem is new, all candidates are timed first (this
    /// synchronises, so it must not run while a graph is being captured).
    pub fn gemm(&self, p: Problem, a: u64, b: u64, c: u64, capturing: bool) -> Result<()> {
        let cached = self.cache.lock().unwrap().get(&p).copied();
        let choice = match cached {
            Some(ch) => ch,
            None if !self.tune || capturing => Choice::GemmEx,
            None => {
                let mut best = (self.time(&p, &Choice::GemmEx, a, b, c)?, Choice::GemmEx);
                for algo in self.candidates(&p)? {
                    let ch = Choice::Lt(algo);
                    if let Ok(t) = self.time(&p, &ch, a, b, c) {
                        if t < best.0 {
                            best = (t, ch);
                        }
                    }
                }
                self.cache.lock().unwrap().insert(p, best.1);
                best.1
            }
        };
        self.run(&p, &choice, a, b, c)
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        unsafe {
            let _ = blas::destroy_handle(self.blas);
            let _ = lt::destroy_handle(self.lt);
        }
    }
}
