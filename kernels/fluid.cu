// Fused kernels for the Fluid Speech (Parakeet TDT / FastConformer) inference engine.
// Activations are fp16 in memory, all arithmetic is fp32. Layouts are channel-last.
#include <cuda_fp16.h>
#include <stdint.h>

#define F2H __float2half
#define H2F __half2float

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// block-wide reductions; blockDim.x must be a multiple of 32 and <= 1024
__device__ float block_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) r = warp_sum(r);
    if (threadIdx.x == 0) sh[0] = r;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float r = (threadIdx.x < nw) ? sh[threadIdx.x] : -INFINITY;
    if (wid == 0) r = warp_max(r);
    if (threadIdx.x == 0) sh[0] = r;
    __syncthreads();
    return sh[0];
}

// ------------------------------------------------------------------------------------------------ subsampling
// conv0: 3x3, stride 2, pad 1, 1 -> C channels, + bias, time mask, relu.
// x: [B, T, F] (mel, time-major), w: [C, 9], bias: [C], out: [B, To, Fo, C]
extern "C" __global__ void sub_conv0(const __half* __restrict__ x, const __half* __restrict__ w,
                                     const __half* __restrict__ bias, const uint32_t* __restrict__ lens,
                                     __half* __restrict__ out, int B, int T, int F, int To, int Fo, int C) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * To * Fo * C;
    if (idx >= total) return;
    int c = idx % C;
    long r = idx / C;
    int fo = r % Fo; r /= Fo;
    int to = r % To;
    int b = r / To;
    if (to >= (int)lens[b]) { out[idx] = F2H(0.f); return; }
    float acc = H2F(bias[c]);
    const __half* xb = x + (long)b * T * F;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        int t = 2 * to - 1 + i;
        if (t < 0 || t >= T) continue;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int f = 2 * fo - 1 + j;
            if (f < 0 || f >= F) continue;
            acc += H2F(w[c * 9 + i * 3 + j]) * H2F(xb[(long)t * F + f]);
        }
    }
    out[idx] = F2H(fmaxf(acc, 0.f));
}

// depthwise 3x3, stride 2, pad 1, + bias, time mask (no activation).
// x: [B, Ti, Fi, C], w: [C, 9], out: [B, To, Fo, C]
extern "C" __global__ void sub_dwconv(const __half* __restrict__ x, const __half* __restrict__ w,
                                      const __half* __restrict__ bias, const uint32_t* __restrict__ lens,
                                      __half* __restrict__ out, int B, int Ti, int Fi, int To, int Fo, int C) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * To * Fo * C;
    if (idx >= total) return;
    int c = idx % C;
    long r = idx / C;
    int fo = r % Fo; r /= Fo;
    int to = r % To;
    int b = r / To;
    if (to >= (int)lens[b]) { out[idx] = F2H(0.f); return; }
    float acc = H2F(bias[c]);
    const __half* xb = x + (long)b * Ti * Fi * C;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        int t = 2 * to - 1 + i;
        if (t < 0 || t >= Ti) continue;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int f = 2 * fo - 1 + j;
            if (f < 0 || f >= Fi) continue;
            acc += H2F(w[c * 9 + i * 3 + j]) * H2F(xb[((long)t * Fi + f) * C + c]);
        }
    }
    out[idx] = F2H(acc);
}

// y = relu((x + bias[c]) * mask[b, t]) for x: [B, T, F, C]
extern "C" __global__ void bias_mask_relu(const __half* __restrict__ x, const __half* __restrict__ bias,
                                          const uint32_t* __restrict__ lens, __half* __restrict__ y,
                                          int B, int T, int F, int C) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * F * C;
    if (idx >= total) return;
    int c = idx % C;
    long r = idx / C / F;
    int t = r % T;
    int b = r / T;
    float v = (t < (int)lens[b]) ? fmaxf(H2F(x[idx]) + H2F(bias[c]), 0.f) : 0.f;
    y[idx] = F2H(v);
}

// ------------------------------------------------------------------------------------------------ norms
// out = LN(x) * g + b over the last dim D (one block per row).
extern "C" __global__ void layer_norm(const __half* __restrict__ x, const __half* __restrict__ g,
                                      const __half* __restrict__ beta, __half* __restrict__ out, int D, float eps) {
    __shared__ float sh[32];
    long row = blockIdx.x;
    const __half* xr = x + row * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += H2F(xr[i]);
    float mean = block_sum(s, sh) / D;
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float d = H2F(xr[i]) - mean; v += d * d; }
    float rstd = rsqrtf(block_sum(v, sh) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out[row * D + i] = F2H((H2F(xr[i]) - mean) * rstd * H2F(g[i]) + H2F(beta[i]));
}

// xo = x + alpha * y, out = LN(xo) * g + b. Dynamic smem: D floats.
extern "C" __global__ void add_layer_norm(const __half* __restrict__ x, const __half* __restrict__ y, float alpha,
                                          const __half* __restrict__ g, const __half* __restrict__ beta,
                                          __half* __restrict__ xo, __half* __restrict__ out, int D, float eps) {
    __shared__ float sh[32];
    extern __shared__ float buf[];
    long row = blockIdx.x;
    const __half* xr = x + row * D;
    __half* xor_ = xo + row * D;
    const __half* yr = y + row * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = H2F(xr[i]) + alpha * H2F(yr[i]);
        __half hv = F2H(v);
        xor_[i] = hv;
        v = H2F(hv);
        buf[i] = v;
        s += v;
    }
    float mean = block_sum(s, sh) / D;
    float var = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float d = buf[i] - mean; var += d * d; }
    float rstd = rsqrtf(block_sum(var, sh) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out[row * D + i] = F2H((buf[i] - mean) * rstd * H2F(g[i]) + H2F(beta[i]));
}

// out = x + alpha * y
extern "C" __global__ void axpy(const __half* __restrict__ x, const __half* __restrict__ y, float alpha,
                                __half* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = F2H(H2F(x[i]) + alpha * H2F(y[i]));
}

// out = silu(x)
extern "C" __global__ void silu(const __half* __restrict__ x, __half* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = H2F(x[i]); out[i] = F2H(v * sigmoidf_(v)); }
}

// ------------------------------------------------------------------------------------------------ attention
// qkv: [B, T, 3, H, dk] -> qu: [B, H, T, dk] (q + u), qv: [H, B, T, dk] (q + v), k, v: [B, H, T, dk]
extern "C" __global__ void qkv_split(const __half* __restrict__ qkv, const __half* __restrict__ pu,
                                     const __half* __restrict__ pv, __half* __restrict__ qu, __half* __restrict__ qvo,
                                     __half* __restrict__ k, __half* __restrict__ v, int B, int T, int H, int dk) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * H * dk;
    if (idx >= total) return;
    int d = idx % dk;
    long r = idx / dk;
    int h = r % H; r /= H;
    int t = r % T;
    int b = r / T;
    long D = (long)H * dk;
    const __half* row = qkv + ((long)b * T + t) * 3 * D;
    int hd = h * dk + d;
    float q = H2F(row[hd]);
    long o_bhtd = (((long)b * H + h) * T + t) * dk + d;
    long o_hbtd = (((long)h * B + b) * T + t) * dk + d;
    qu[o_bhtd] = F2H(q + H2F(pu[hd]));
    qvo[o_hbtd] = F2H(q + H2F(pv[hd]));
    k[o_bhtd] = row[D + hd];
    v[o_bhtd] = row[2 * D + hd];
}

// probs[b,h,i,:] = softmax((ac[b,h,i,j] + bd[h,b,i,j+T-1-i]) * scale  + (j >= len[b] ? -10000 : 0))
// ac: [B, H, T, T], bd: [H, B, T, 2T-1], probs: [B, H, T, T]. One block per (b, h, i); dynamic smem T floats.
extern "C" __global__ void relpos_softmax(const __half* __restrict__ ac, const __half* __restrict__ bd,
                                          const uint32_t* __restrict__ lens, __half* __restrict__ probs,
                                          int B, int H, int T, float scale) {
    __shared__ float sh[32];
    extern __shared__ float row[];
    long r = blockIdx.x;
    int i = r % T;
    long bh = r / T;
    int h = bh % H;
    int b = bh / H;
    int len = lens[b];
    const __half* acr = ac + r * T;
    const __half* bdr = bd + (((long)h * B + b) * T + i) * (2 * T - 1) + (T - 1 - i);
    float m = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float s = (H2F(acr[j]) + H2F(bdr[j])) * scale + (j >= len ? -10000.f : 0.f);
        row[j] = s;
        m = fmaxf(m, s);
    }
    m = block_max(m, sh);
    float sum = 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float e = __expf(row[j] - m);
        row[j] = e;
        sum += e;
    }
    float inv = 1.f / block_sum(sum, sh);
    __half* pr = probs + r * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x) pr[j] = F2H(row[j] * inv);
}

// [B, H, T, dk] -> [B, T, H*dk]
extern "C" __global__ void merge_heads(const __half* __restrict__ x, __half* __restrict__ out, int B, int H, int T, int dk) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * H * dk;
    if (idx >= total) return;
    int d = idx % dk;
    long r = idx / dk;
    int h = r % H; r /= H;
    int t = r % T;
    int b = r / T;
    out[idx] = x[(((long)b * H + h) * T + t) * dk + d];
}

// ------------------------------------------------------------------------------------------------ conv module
// GLU over the last dim + time mask: x [B, T, 2D] -> out [B, T, D]
extern "C" __global__ void glu_mask(const __half* __restrict__ x, const uint32_t* __restrict__ lens,
                                    __half* __restrict__ out, int B, int T, int D) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * D;
    if (idx >= total) return;
    int c = idx % D;
    long bt = idx / D;
    int t = bt % T;
    int b = bt / T;
    float v = 0.f;
    if (t < (int)lens[b]) {
        float a = H2F(x[bt * 2 * D + c]);
        float g = H2F(x[bt * 2 * D + D + c]);
        v = a * sigmoidf_(g);
    }
    out[idx] = F2H(v);
}

// depthwise conv1d over time (same padding) + bias + silu: x [B, T, D], w [D, K] -> out [B, T, D]
extern "C" __global__ void dwconv1d_silu(const __half* __restrict__ x, const __half* __restrict__ w,
                                         const __half* __restrict__ bias, __half* __restrict__ out,
                                         int B, int T, int D, int K) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * T * D;
    if (idx >= total) return;
    int c = idx % D;
    long bt = idx / D;
    int t = bt % T;
    int b = bt / T;
    int p = K / 2;
    float acc = H2F(bias[c]);
    const __half* xb = x + (long)b * T * D;
    for (int k = 0; k < K; k++) {
        int tt = t + k - p;
        if (tt >= 0 && tt < T) acc += H2F(w[c * K + k]) * H2F(xb[(long)tt * D + c]);
    }
    out[idx] = F2H(acc * sigmoidf_(acc));
}

// ------------------------------------------------------------------------------------------------ TDT decoding (fp32)
// z[b, :] = relu(ep[b, frame[b], :] + dp[b, :]);  ep: [B, T, J], dp: [B, J]
extern "C" __global__ void joint_hidden(const float* __restrict__ ep, const float* __restrict__ dp,
                                        const uint32_t* __restrict__ frame, float* __restrict__ z, int B, int T, int J) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * J) return;
    int j = idx % J, b = idx / J;
    z[idx] = fmaxf(ep[((long)b * T + frame[b]) * J + j] + dp[idx], 0.f);
}

// per row: out[b] = (argmax logits[b, :V], argmax logits[b, V:V+ND]). One block per row.
extern "C" __global__ void tdt_argmax(const float* __restrict__ logits, uint32_t* __restrict__ out, int V, int ND) {
    __shared__ float sv[32];
    __shared__ int si[32];
    int b = blockIdx.x;
    const float* row = logits + (long)b * (V + ND);
    float best = -INFINITY;
    int bi = 0;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float v = row[j];
        if (v > best) { best = v; bi = j; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) { sv[wid] = best; si[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < nw; w++)
            if (sv[w] > sv[0] || (sv[w] == sv[0] && si[w] < si[0])) { sv[0] = sv[w]; si[0] = si[w]; }
        float db = row[V];
        int di = 0;
        for (int d = 1; d < ND; d++) if (row[V + d] > db) { db = row[V + d]; di = d; }
        out[2 * b] = si[0];
        out[2 * b + 1] = di;
    }
}

// LSTM cell with PyTorch gate order (i, f, g, o); rows with mask == 0 keep their state.
// gates: [B, 4H] (all terms summed), h, c: [B, H] -> ho, co: [B, H]
extern "C" __global__ void lstm_cell(const float* __restrict__ gates, const float* __restrict__ h,
                                     const float* __restrict__ c, const uint32_t* __restrict__ mask,
                                     float* __restrict__ ho, float* __restrict__ co, int B, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;
    int j = idx % H, b = idx / H;
    if (!mask[b]) { ho[idx] = h[idx]; co[idx] = c[idx]; return; }
    const float* g = gates + (long)b * 4 * H;
    float i_ = sigmoidf_(g[j]), f_ = sigmoidf_(g[H + j]), g_ = tanhf(g[2 * H + j]), o_ = sigmoidf_(g[3 * H + j]);
    float cn = f_ * c[idx] + i_ * g_;
    co[idx] = cn;
    ho[idx] = o_ * tanhf(cn);
}

// ================================================================================================ device-side TDT loop
// State lives on the device; the host only polls `flags[1]` (rows still active) every few steps.
// Weights fp16 (row-major [rows, K]), math fp32. Staging of per-row inputs in shared memory by groups of 8 rows.
#define DEC_G 8

struct DecState {
    const __half* embed;     // [V, P]
    const __half* wih0; const __half* whh0; const float* b0;   // [4P, P], [4P, P], [4P]
    const __half* wih1; const __half* whh1; const float* b1;
    const __half* pred_w; const float* pred_b;                  // [J, P], [J]
    const __half* head_w; const float* head_b;                  // [V + ND, J], [V + ND]
    const __half* ep;        // [B, T, J]
    const uint32_t* lens;    // [B]
    float* h0; float* c0; float* h1; float* c1;                 // [B, P] committed state
    float* h0n; float* c0n; float* h1n; float* c1n;             // [B, P] candidates
    float* dp;               // [B, J]
    uint32_t* pos;           // [B] current frame
    uint32_t* emitted;       // [B] symbols emitted at the current frame
    uint32_t* last_tok;      // [B]
    uint32_t* update;        // [B] row emitted a token this step -> advance prediction net
    unsigned long long* best;// [B] packed argmax of token logits
    float* dur;              // [B, ND] duration logits
    uint32_t* out_tok;       // [B, cap]
    uint32_t* out_frame;     // [B, cap]
    uint32_t* count;         // [B]
    uint32_t* flags;         // [0] any row updated, [1] rows still active
    const uint32_t* durations; // [ND]
    float* pmax;             // [B, jblocks] per-block max of the token logits
    float* psum;             // [B, jblocks] per-block sum of exp(logit - pmax)
    float* out_prob;         // [B, cap] softmax probability of each emitted token
    int B, T, P, J, V, ND, cap, blank, max_symbols, jblocks;
};

__device__ __forceinline__ float unpack_max(unsigned long long key) {
    uint32_t u = (uint32_t)(key >> 32);
    u = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    return __uint_as_float(u);
}

__device__ __forceinline__ unsigned long long pack_max(float v, uint32_t idx) {
    uint32_t u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | (uint32_t)(~idx);
}

// Register-cached fp16 weight rows of length DEC_K (= P = J = 640): lane l holds half2 elements l, l+32, ...
#define DEC_K 640
#define DEC_R (DEC_K / 64)
struct WRow { __half2 w[DEC_R]; };

__device__ __forceinline__ void load_row(WRow& r, const __half* __restrict__ w, int lane) {
    const __half2* w2 = reinterpret_cast<const __half2*>(w);
#pragma unroll
    for (int i = 0; i < DEC_R; i++) r.w[i] = w2[lane + 32 * i];
}
// acc[q][g] (+)= W[q] . x_g for N cached rows and G vectors in shared memory (row stride `ld` floats)
template <int N, int G>
__device__ __forceinline__ void rows_dot_g(const WRow* r, const float* __restrict__ x, int ld, int lane, float (*acc)[G]) {
#pragma unroll
    for (int i = 0; i < DEC_R; i++) {
        float2 wf[N];
#pragma unroll
        for (int q = 0; q < N; q++) wf[q] = __half22float2(r[q].w[i]);
#pragma unroll
        for (int g = 0; g < G; g++) {
            float2 v = reinterpret_cast<const float2*>(x + g * ld)[lane + 32 * i];
#pragma unroll
            for (int q = 0; q < N; q++) acc[q][g] += wf[q].x * v.x + wf[q].y * v.y;
        }
    }
}
template <int N, int G>
__device__ __forceinline__ void warp_sum_g(float (*v)[G]) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int q = 0; q < N; q++)
#pragma unroll
            for (int g = 0; g < G; g++) v[q][g] += __shfl_xor_sync(0xffffffff, v[q][g], o);
}

// One LSTM layer for all rows with update[b] == 1. Warp per hidden unit j computes its 4 gates (i, f, g, o)
// for a whole group of G rows at once. layer 0 input: embed[last_tok[b]]; layer 1 input: h0n[b].
template <int G>
__device__ void dec_lstm_t(const DecState& s, int layer) {
    if (!s.flags[0]) return;
    extern __shared__ float sm[];  // x: [G][P] then h: [G][P]
    const int P = DEC_K, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int j = blockIdx.x * nw + warp;
    const __half* wih = layer ? s.wih1 : s.wih0;
    const __half* whh = layer ? s.whh1 : s.whh0;
    const float* bias = layer ? s.b1 : s.b0;
    const float* h = layer ? s.h1 : s.h0;
    const float* c = layer ? s.c1 : s.c0;
    float* hn = layer ? s.h1n : s.h0n;
    float* cn = layer ? s.c1n : s.c0n;
    float* xs = sm;
    float* hs = sm + G * P;
    WRow wi[4], wh[4];
    float bq[4];
    if (j < P) {
#pragma unroll
        for (int q = 0; q < 4; q++) {
            long r = (long)q * P + j;
            load_row(wi[q], wih + r * P, lane);
            load_row(wh[q], whh + r * P, lane);
            bq[q] = bias[r];
        }
    }
    for (int g0 = 0; g0 < s.B; g0 += G) {
        int ng = min(G, s.B - g0);
        __syncthreads();
        __shared__ long xbase[G];
        __shared__ int ok[G];
        if (threadIdx.x < G) {
            int b = g0 + threadIdx.x;
            int on = threadIdx.x < ng && s.update[b];
            ok[threadIdx.x] = on;
            xbase[threadIdx.x] = on ? (layer ? (long)b * P : (long)s.last_tok[b] * P) : 0;
        }
        __syncthreads();
#pragma unroll 4
        for (int i = 2 * threadIdx.x; i < G * P; i += 2 * blockDim.x) {
            int bb = i / P, k = i % P, b = g0 + bb;
            float2 xv = make_float2(0.f, 0.f), hv = make_float2(0.f, 0.f);
            if (ok[bb]) {
                xv = layer ? *reinterpret_cast<const float2*>(s.h0n + xbase[bb] + k)
                           : __half22float2(*reinterpret_cast<const __half2*>(s.embed + xbase[bb] + k));
                hv = *reinterpret_cast<const float2*>(h + (long)b * P + k);
            }
            *reinterpret_cast<float2*>(xs + i) = xv;
            *reinterpret_cast<float2*>(hs + i) = hv;
        }
        __syncthreads();
        if (j >= P) continue;
        float acc[4][G];
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int g = 0; g < G; g++) acc[q][g] = 0.f;
        rows_dot_g<4, G>(wi, xs, P, lane, acc);
        rows_dot_g<4, G>(wh, hs, P, lane, acc);
        warp_sum_g<4, G>(acc);
        // lane g finalises row g0 + g
#pragma unroll
        for (int g = 0; g < G; g++) {
            int b = g0 + g;
            if (lane == g && g < ng && s.update[b]) {
                float ig = sigmoidf_(acc[0][g] + bq[0]), fg = sigmoidf_(acc[1][g] + bq[1]);
                float gg = tanhf(acc[2][g] + bq[2]), og = sigmoidf_(acc[3][g] + bq[3]);
                float cv = fg * c[b * P + j] + ig * gg;
                cn[b * P + j] = cv;
                hn[b * P + j] = og * tanhf(cv);
            }
        }
    }
}

// dp[b] = pred_w . h1n[b] + pred_b for updated rows, then commit candidates to the state. Warp per output row.
template <int G>
__device__ void dec_pred_t(const DecState& s) {
    if (!s.flags[0]) return;
    extern __shared__ float sm[];  // [G][P]
    const int P = DEC_K, J = s.J, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r = blockIdx.x * nw + warp;
    WRow w[1];
    float bias = 0.f;
    if (r < J) { load_row(w[0], s.pred_w + (long)r * P, lane); bias = s.pred_b[r]; }
    for (int g0 = 0; g0 < s.B; g0 += G) {
        int ng = min(G, s.B - g0);
        __syncthreads();
        __shared__ int ok[G];
        if (threadIdx.x < G) ok[threadIdx.x] = threadIdx.x < ng && s.update[g0 + threadIdx.x];
        __syncthreads();
#pragma unroll 4
        for (int i = 2 * threadIdx.x; i < G * P; i += 2 * blockDim.x) {
            int bb = i / P;
            float2 v = ok[bb] ? *reinterpret_cast<const float2*>(s.h1n + (long)(g0 + bb) * P + i % P) : make_float2(0.f, 0.f);
            *reinterpret_cast<float2*>(sm + i) = v;
        }
        __syncthreads();
        if (r >= J) continue;
        float acc[1][G];
#pragma unroll
        for (int g = 0; g < G; g++) acc[0][g] = 0.f;
        rows_dot_g<1, G>(w, sm, P, lane, acc);
        warp_sum_g<1, G>(acc);
#pragma unroll
        for (int g = 0; g < G; g++) {
            int b = g0 + g;
            if (lane == g && g < ng && s.update[b]) s.dp[b * J + r] = acc[0][g] + bias;
        }
    }
    // commit candidates (other blocks only read h1n, which is not modified here)
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < (long)s.B * P; i += nt) {
        if (!s.update[i / P]) continue;
        s.h0[i] = s.h0n[i]; s.c0[i] = s.c0n[i]; s.h1[i] = s.h1n[i]; s.c1[i] = s.c1n[i];
    }
}

// Joint network for every active row + argmax over tokens (packed atomicMax) + duration logits.
// Each warp owns DEC_JR consecutive output rows (cached in registers).
#define DEC_JR 4
template <int G>
__device__ void dec_joint_t(const DecState& s) {
    if (!s.flags[1]) return;
    extern __shared__ float sm[];  // [G][J] hidden, then best[nw][G] (u64), then max/sum[nw][G] (f32)
    const int J = DEC_K, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int rows = s.V + s.ND;
    unsigned long long* wbest = reinterpret_cast<unsigned long long*>(sm + G * J);
    float* wmax = reinterpret_cast<float*>(wbest + nw * G);
    float* wsum = wmax + nw * G;
    const int r0 = (blockIdx.x * nw + warp) * DEC_JR;
    WRow w[DEC_JR];
    float bias[DEC_JR];
#pragma unroll
    for (int q = 0; q < DEC_JR; q++) {
        int r = min(r0 + q, rows - 1);
        load_row(w[q], s.head_w + (long)r * J, lane);
        bias[q] = s.head_b[r];
    }
    for (int g0 = 0; g0 < s.B; g0 += G) {
        int ng = min(G, s.B - g0);
        __syncthreads();
        __shared__ long ebase[G];
        __shared__ int ok[G];
        if (threadIdx.x < G) {
            int b = g0 + threadIdx.x;
            int on = threadIdx.x < ng && s.pos[b] < s.lens[b];
            ok[threadIdx.x] = on;
            ebase[threadIdx.x] = on ? ((long)b * s.T + min((int)s.pos[b], s.T - 1)) * J : 0;
        }
        __syncthreads();
#pragma unroll 4
        for (int i = 2 * threadIdx.x; i < G * J; i += 2 * blockDim.x) {
            int bb = i / J, k = i % J;
            float2 z = make_float2(0.f, 0.f);
            if (ok[bb]) {
                float2 e = __half22float2(*reinterpret_cast<const __half2*>(s.ep + ebase[bb] + k));
                float2 d = *reinterpret_cast<const float2*>(s.dp + (long)(g0 + bb) * J + k);
                z = make_float2(fmaxf(e.x + d.x, 0.f), fmaxf(e.y + d.y, 0.f));
            }
            *reinterpret_cast<float2*>(sm + i) = z;
        }
        for (int i = threadIdx.x; i < nw * G; i += blockDim.x) wbest[i] = 0ull;
        __syncthreads();
        if (r0 < rows) {
            float acc[DEC_JR][G];
#pragma unroll
            for (int q = 0; q < DEC_JR; q++)
#pragma unroll
                for (int g = 0; g < G; g++) acc[q][g] = 0.f;
            rows_dot_g<DEC_JR, G>(w, sm, J, lane, acc);
            warp_sum_g<DEC_JR, G>(acc);
#pragma unroll
            for (int g = 0; g < G; g++) {
                if (lane != g) continue;
                int b = g0 + g;
                unsigned long long m = 0ull;
                float lm = -INFINITY, ls = 0.f;
                if (g < ng && s.pos[b] < s.lens[b]) {
#pragma unroll
                    for (int q = 0; q < DEC_JR; q++) {
                        int r = r0 + q;
                        if (r >= rows) break;
                        float v = acc[q][g] + bias[q];
                        if (r < s.V) {
                            m = max(m, pack_max(v, r));
                            // running log-sum-exp of the token logits
                            if (v > lm) { ls = ls * __expf(lm - v) + 1.f; lm = v; } else { ls += __expf(v - lm); }
                        } else {
                            s.dur[b * s.ND + (r - s.V)] = v;
                        }
                    }
                }
                wbest[warp * G + g] = m;
                wmax[warp * G + g] = lm;
                wsum[warp * G + g] = ls;
            }
        } else if (lane < G) {
            wbest[warp * G + lane] = 0ull;
            wmax[warp * G + lane] = -INFINITY;
            wsum[warp * G + lane] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < ng) {
            unsigned long long m = 0ull;
            float bm = -INFINITY, bs = 0.f;
            for (int w2 = 0; w2 < nw; w2++) {
                m = max(m, wbest[w2 * G + threadIdx.x]);
                float wm = wmax[w2 * G + threadIdx.x], ws = wsum[w2 * G + threadIdx.x];
                if (ws > 0.f) {
                    if (wm > bm) { bs = bs * __expf(bm - wm) + ws; bm = wm; } else { bs += ws * __expf(wm - bm); }
                }
            }
            if (m) atomicMax(&s.best[g0 + threadIdx.x], m);
            const int b = g0 + threadIdx.x;
            s.pmax[(long)b * s.jblocks + blockIdx.x] = bm;
            s.psum[(long)b * s.jblocks + blockIdx.x] = bs;
        }
    }
}

#define DEC_INST(G)                                                                                  \
    extern "C" __global__ void __launch_bounds__(256, 1) dec_lstm_g##G(DecState s, int layer) { dec_lstm_t<G>(s, layer); } \
    extern "C" __global__ void __launch_bounds__(256) dec_pred_g##G(DecState s) { dec_pred_t<G>(s); }                 \
    extern "C" __global__ void __launch_bounds__(256) dec_joint_g##G(DecState s) { dec_joint_t<G>(s); }
DEC_INST(1)
DEC_INST(2)
DEC_INST(4)
DEC_INST(8)

// Greedy TDT bookkeeping for every row (single block).
__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    if (s2 <= 0.f) return;
    if (m2 > m) { s = s * __expf(m - m2) + s2; m = m2; } else { s += s2 * __expf(m2 - m); }
}

// Greedy TDT bookkeeping for every row (single block). Dynamic smem: 2 * B floats.
extern "C" __global__ void dec_update(DecState s) {
    __shared__ uint32_t any_upd, active;
    extern __shared__ float lse[];  // [B] max, [B] sum of the token logits
    if (threadIdx.x == 0) { any_upd = 0; active = 0; }
    // combine the joint blocks' partial log-sum-exps: one warp per row
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int b = warp; b < s.B; b += nw) {
        float gm = -INFINITY, gs = 0.f;
        if (s.pos[b] < s.lens[b])
            for (int k = lane; k < s.jblocks; k += 32)
                lse_merge(gm, gs, s.pmax[(long)b * s.jblocks + k], s.psum[(long)b * s.jblocks + k]);
        for (int o = 16; o > 0; o >>= 1) {
            float m2 = __shfl_xor_sync(0xffffffff, gm, o), s2 = __shfl_xor_sync(0xffffffff, gs, o);
            lse_merge(gm, gs, m2, s2);
        }
        if (lane == 0) { lse[b] = gm; lse[s.B + b] = gs; }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < s.B; b += blockDim.x) {
        s.update[b] = 0;
        uint32_t len = s.lens[b];
        if (s.pos[b] >= len) continue;
        unsigned long long key = s.best[b];
        s.best[b] = 0ull;
        uint32_t tok = ~(uint32_t)(key & 0xffffffffull);
        const float* d = s.dur + b * s.ND;
        int di = 0;
        for (int k = 1; k < s.ND; k++) if (d[k] > d[di]) di = k;
        uint32_t step = s.durations[di];
        if ((int)tok != s.blank) {
            uint32_t n = s.count[b];
            if ((int)n < s.cap) {
                // softmax probability of the token
                const float gm = lse[b], gs = lse[s.B + b];
                s.out_prob[(long)b * s.cap + n] = gs > 0.f ? __expf(unpack_max(key) - gm) / gs : 0.f;
                s.out_tok[(long)b * s.cap + n] = tok;
                s.out_frame[(long)b * s.cap + n] = s.pos[b];
                s.count[b] = n + 1;
            }
            s.last_tok[b] = tok;
            s.update[b] = 1;
            s.emitted[b] += 1;
            atomicOr(&any_upd, 1u);
        } else if (step == 0) {
            step = 1;
        }
        if (step == 0 && (int)s.emitted[b] >= s.max_symbols) step = 1;
        if (step > 0) s.emitted[b] = 0;
        s.pos[b] += step;
        if (s.pos[b] < len) atomicAdd(&active, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) { s.flags[0] = any_upd; s.flags[1] = active; }
}

// x[r, c] += b[c] for x: [n / d, d]
extern "C" __global__ void add_bias(__half* __restrict__ x, const __half* __restrict__ b, long n, int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = F2H(H2F(x[i]) + H2F(b[i % d]));
}

// conv0 (3x3/s2, 1 -> C, bias, time mask, relu) fused with the following depthwise 3x3/s2 conv (bias, time mask).
// mel: [B, T, F]; out: [B, T2, F2, C]. One block per output position (b, t2, f2), one thread per channel.
extern "C" __global__ void sub_conv01(const __half* __restrict__ mel, const __half* __restrict__ w0,
                                      const __half* __restrict__ b0, const __half* __restrict__ w2,
                                      const __half* __restrict__ b2, const uint32_t* __restrict__ lens1,
                                      const uint32_t* __restrict__ lens2, __half* __restrict__ out,
                                      int B, int T, int F, int T1, int F1, int T2, int F2, int C) {
    __shared__ float patch[7][7];
    const int pos = blockIdx.x;
    const int f2 = pos % F2, t2 = (pos / F2) % T2, b = pos / F2 / T2;
    const int c = threadIdx.x;
    const int tm0 = 4 * t2 - 3, fm0 = 4 * f2 - 3;  // mel origin of the 7x7 receptive field
    if (threadIdx.x < 49) {
        int r = threadIdx.x / 7, q = threadIdx.x % 7, tt = tm0 + r, ff = fm0 + q;
        patch[r][q] = (tt >= 0 && tt < T && ff >= 0 && ff < F) ? __half2float(mel[((long)b * T + tt) * F + ff]) : 0.f;
    }
    __syncthreads();
    if (c >= C) return;
    long o = (((long)b * T2 + t2) * F2 + f2) * C + c;
    if (t2 >= (int)lens2[b]) { out[o] = __float2half(0.f); return; }
    float w0r[9], w2r[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { w0r[i] = __half2float(w0[c * 9 + i]); w2r[i] = __half2float(w2[c * 9 + i]); }
    const float bias0 = __half2float(b0[c]);
    const int l1 = lens1[b];
    float acc = __half2float(b2[c]);
#pragma unroll
    for (int i = 0; i < 3; i++) {
        const int t1 = 2 * t2 - 1 + i;
        if (t1 < 0 || t1 >= T1 || t1 >= l1) continue;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int f1 = 2 * f2 - 1 + j;
            if (f1 < 0 || f1 >= F1) continue;
            // conv0 at (t1, f1): mel rows 2*t1-1+p = tm0 + 2i + p, cols fm0 + 2j + q
            float v = bias0;
#pragma unroll
            for (int p = 0; p < 3; p++)
#pragma unroll
                for (int q = 0; q < 3; q++) v += w0r[p * 3 + q] * patch[2 * i + p][2 * j + q];
            acc += w2r[i * 3 + j] * fmaxf(v, 0.f);
        }
    }
    out[o] = __float2half(acc);
}

// out = silu(x), 8 halves per thread (n % 8 == 0)
extern "C" __global__ void silu8(const __half* __restrict__ x, __half* __restrict__ out, long n8) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 v = reinterpret_cast<const uint4*>(x)[i];
    __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float2 f = __half22float2(h[k]);
        f.x = f.x * sigmoidf_(f.x);
        f.y = f.y * sigmoidf_(f.y);
        h[k] = __floats2half2_rn(f.x, f.y);
    }
    reinterpret_cast<uint4*>(out)[i] = v;
}
