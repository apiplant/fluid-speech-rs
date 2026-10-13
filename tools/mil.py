"""Minimal reader + numpy interpreter for compiled Core ML ML Programs (model.mil + weights/weight.bin).

Only the ops used by the Fluid Speech / Parakeet packages are implemented. Everything is computed in fp32
(fp16 tensors are widened), so the interpreter is a numerics reference for the graph, not a bit-exact ANE replay.
"""
import os
import re

import numpy as np

# ----------------------------------------------------------------------------------------------- tokenizer

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<num>[-+]?(?:0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[-+]?\d+|\d+\.\d*(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?|inf|nan)(?![A-Za-z_]))
  | (?P<id>[A-Za-z_@][A-Za-z0-9_@.]*)
  | (?P<p>->|[{}()\[\]<>,=;])
""",
    re.X,
)


def tokenize(text):
    out = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise SyntaxError(f"bad token at {text[pos:pos + 40]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append((kind, m.group()))
    return out


class Blob:
    def __init__(self, path, offset):
        self.path, self.offset = path, offset


class TensorType:
    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, shape

    def __repr__(self):
        return f"{self.dtype}{self.shape}"


class Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else (None, None)

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, v):
        k, val = self.next()
        if val != v:
            raise SyntaxError(f"expected {v!r} got {val!r} near {self.t[self.i - 5:self.i + 5]}")

    def accept(self, v):
        if self.peek()[1] == v:
            self.i += 1
            return True
        return False

    # type := tensor<dtype, [dims]> | scalar_dtype | dict<...> | list<...>
    def parse_type(self):
        _, name = self.next()
        if name == "tensor":
            self.expect("<")
            _, dtype = self.next()
            self.expect(",")
            self.expect("[")
            shape = []
            while not self.accept("]"):
                _, d = self.next()
                shape.append(int(d) if d.lstrip("-").isdigit() else d)
                self.accept(",")
            self.expect(">")
            return TensorType(dtype, tuple(shape))
        if name in ("dict", "list"):
            depth = 0
            while True:
                _, v = self.next()
                if v == "<":
                    depth += 1
                elif v == ">":
                    depth -= 1
                    if depth == 0:
                        break
            return TensorType(name, ())
        return TensorType(name, ())

    def parse_nested(self):
        if self.accept("["):
            items = []
            while not self.accept("]"):
                items.append(self.parse_nested())
                self.accept(",")
            return items
        k, v = self.next()
        if k == "num":
            return _num(v)
        if k == "id" and v in ("true", "false"):
            return v == "true"
        if k == "str":
            return _unquote(v)
        raise SyntaxError(f"bad literal {v}")

    def parse_value(self):
        k, v = self.peek()
        if v == "(":  # tuple of values
            self.next()
            items = []
            while not self.accept(")"):
                items.append(self.parse_value())
                self.accept(",")
            return ("tuple", items)
        if k == "id" and v == "BLOBFILE":
            self.next()
            self.expect("(")
            args = self.parse_args(")")
            return ("blob", Blob(_lit(args["path"]), int(_lit(args["offset"]))))
        if k == "id" and (v == "tensor" or v in _SCALARS or v in ("dict", "list")) and self.peek(1)[1] in ("<", "("):
            ty = self.parse_type()
            self.expect("(")
            if self.peek()[1] == "BLOBFILE":
                inner = self.parse_value()
                self.expect(")")
                return ("blob_typed", ty, inner[1])
            if ty.dtype in ("dict", "list"):
                depth = 1
                while depth:
                    _, x = self.next()
                    depth += {"(": 1, ")": -1}.get(x, 0)
                return ("lit", ty, None)
            val = self.parse_nested()
            self.expect(")")
            return ("lit", ty, val)
        if k == "str":
            self.next()
            return ("lit", TensorType("string", ()), _unquote(v))
        if k == "num":
            self.next()
            return ("lit", TensorType("fp32", ()), _num(v))
        self.next()
        return ("var", v)

    def parse_args(self, close):
        args = {}
        while not self.accept(close):
            _, key = self.next()
            self.expect("=")
            args[key] = self.parse_value()
            self.accept(",")
        return args


_SCALARS = {"fp16", "fp32", "fp64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "string"}


def _num(v):
    s = v.lstrip("+")
    neg = s.startswith("-")
    body = s[1:] if neg else s
    if body.startswith("0x"):
        x = float.fromhex(body)
    elif body in ("inf", "nan"):
        x = float(body)
    elif re.fullmatch(r"\d+", body):
        x = int(body)
    else:
        x = float(body)
    return -x if neg else x


def _unquote(s):
    return bytes(s[1:-1], "utf-8").decode("unicode_escape")


def _lit(v):
    assert v[0] == "lit", v
    return v[2]


class Op:
    def __init__(self, outputs, op, args, attrs):
        self.outputs, self.op, self.args, self.attrs = outputs, op, args, attrs

    @property
    def name(self):
        n = self.attrs.get("name")
        return _lit(n) if n else self.outputs[0][0]


class Function:
    def __init__(self, name, inputs, ops, returns):
        self.name, self.inputs, self.ops, self.returns = name, inputs, ops, returns


def parse_mil(path):
    text = open(path).read()
    p = Parser(tokenize(text))
    # skip "program(x)" and optional "[buildInfo = ...]"
    p.expect("program")
    p.expect("(")
    p.next()
    p.expect(")")
    if p.accept("["):
        depth = 1
        while depth:
            _, v = p.next()
            depth += {"[": 1, "]": -1}.get(v, 0)
    p.expect("{")
    funcs = {}
    while not p.accept("}"):
        p.expect("func")
        _, fname = p.next()
        if p.accept("<"):
            p.next()
            p.expect(">")
        p.expect("(")
        inputs = []
        while not p.accept(")"):
            ty = p.parse_type()
            _, n = p.next()
            inputs.append((n, ty))
            p.accept(",")
        p.expect("{")
        ops = []
        while not p.accept("}"):
            outs = []
            while True:
                ty = p.parse_type()
                _, n = p.next()
                outs.append((n, ty))
                if not p.accept(","):
                    break
            p.expect("=")
            _, opname = p.next()
            p.expect("(")
            args = p.parse_args(")")
            attrs = {}
            if p.accept("["):
                attrs = p.parse_args("]")
            p.expect(";")
            ops.append(Op(outs, opname, args, attrs))
        p.expect("->")
        p.expect("(")
        rets = []
        while not p.accept(")"):
            rets.append(p.next()[1])
            p.accept(",")
        p.accept(";")
        funcs[fname] = Function(fname, inputs, ops, rets)
    return funcs


# ----------------------------------------------------------------------------------------------- blobs

_NP = {
    "fp16": np.float16, "fp32": np.float32, "fp64": np.float64, "int8": np.int8, "uint8": np.uint8,
    "int16": np.int16, "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64,
    "uint64": np.uint64, "bool": np.bool_,
}
_SUBBYTE = {"uint1": 1, "uint2": 2, "uint3": 3, "uint4": 4, "uint6": 6, "int4": 4}


class BlobFile:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def read(self, offset, ty):
        hdr = self.data[offset:offset + 64]
        sentinel, _dtype = np.frombuffer(hdr[:8].tobytes(), dtype="<u4")
        assert sentinel == 0xDEADBEEF, hex(sentinel)
        size, data_off = np.frombuffer(hdr[8:24].tobytes(), dtype="<u8")
        raw = np.asarray(self.data[data_off:data_off + size])
        n = int(np.prod(ty.shape)) if ty.shape else 1
        if ty.dtype in _SUBBYTE:
            nbits = _SUBBYTE[ty.dtype]
            bits = np.unpackbits(raw, bitorder="little")[: n * nbits].reshape(n, nbits)
            vals = (bits.astype(np.uint16) << np.arange(nbits, dtype=np.uint16)).sum(axis=1)
            if ty.dtype.startswith("int"):
                vals = vals.astype(np.int16)
                vals[vals >= 1 << (nbits - 1)] -= 1 << nbits
            return vals.astype(np.int32).reshape(ty.shape)
        arr = np.frombuffer(raw.tobytes(), dtype=_NP[ty.dtype])[:n]
        return arr.reshape(ty.shape).copy()


class Model:
    """A compiled .mlmodelc directory."""

    def __init__(self, path):
        self.path = path
        self.funcs = parse_mil(os.path.join(path, "model.mil"))
        self.main = self.funcs["main"]
        wpath = os.path.join(path, "weights", "weight.bin")
        self.blob = BlobFile(wpath) if os.path.exists(wpath) else None

    def literal(self, v):
        kind = v[0]
        if kind == "lit":
            ty, val = v[1], v[2]
            if ty.dtype == "string" or val is None:
                return val
            arr = np.array(val, dtype=_NP.get(ty.dtype, np.float32))
            return arr.reshape(ty.shape) if ty.shape else arr.reshape(())
        if kind == "blob_typed":
            return self.blob.read(v[2].offset, v[1])
        raise ValueError(v)

    # constants (const + constexpr_*) resolved without running the graph
    def constants(self):
        env = {}
        for op in self.main.ops:
            if op.op == "const" or op.op.startswith("constexpr_"):
                env[op.outputs[0][0]] = self.eval_op(op, env)
        return env

    def eval_op(self, op, env):
        def get(k, default=None):
            v = op.args.get(k, op.attrs.get(k))
            if v is None:
                return default
            if v[0] == "var":
                return env[v[1]]
            if v[0] == "tuple":
                return [env[x[1]] if x[0] == "var" else self.literal(x) for x in v[1]]
            return self.literal(v)

        return OPS[op.op](get, op)

    def run(self, inputs, keep=False, fp16=False):
        env = dict(inputs)
        for op in self.main.ops:
            out = self.eval_op(op, env)
            if len(op.outputs) == 1:
                out = [out]
            for (n, ty), val in zip(op.outputs, out):
                if not isinstance(val, str) and (isinstance(val, np.ndarray) or np.isscalar(val)):
                    val = np.asarray(val)
                    if ty.dtype in _NP and ty.dtype not in ("fp16",):
                        val = val.astype(_NP[ty.dtype])
                    elif ty.dtype == "fp16":
                        val = val.astype(np.float16 if fp16 else np.float32)
                    if ty.shape and val.shape != ty.shape:
                        raise ValueError(f"{op.op} {n}: got {val.shape}, declared {ty.shape}")
                env[n] = val
        return env if keep else {r: env[r] for r in self.main.returns}


# ----------------------------------------------------------------------------------------------- ops


def _f(x):
    return np.asarray(x, dtype=np.float32) if np.asarray(x).dtype in (np.float16, np.float64) else np.asarray(x)


def op_const(get, op):
    return get("val")


def op_blockwise(get, op):
    data = get("data").astype(np.float32)
    scale = _f(get("scale")).astype(np.float32)
    offset = get("offset")
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float32)
        offset = _expand_blocks(offset, data.shape)
        data = data - offset
    return data * _expand_blocks(scale, data.shape)


def _expand_blocks(s, shape):
    s = np.asarray(s)
    if s.ndim == 0:
        return s
    for ax, (a, b) in enumerate(zip(s.shape, shape)):
        if a != b:
            s = np.repeat(s, b // a, axis=ax)
    return s


def op_lut(get, op):
    idx = get("indices").astype(np.int64)
    lut = _f(get("lut")).astype(np.float32)  # [*blocks, n_palette, vector_size]
    assert lut.shape[-1] == 1, "vector palettization not supported"
    r = idx.ndim
    blocks = lut.shape[:r]
    out = np.empty(idx.shape, dtype=np.float32)
    bs = [idx.shape[i] // blocks[i] for i in range(r)]
    for bidx in np.ndindex(*blocks):
        sl = tuple(slice(b * s, (b + 1) * s) for b, s in zip(bidx, bs))
        out[sl] = lut[bidx][:, 0][idx[sl]]
    return out


def op_affine(get, op):
    q = get("quantized_data").astype(np.float32)
    zp = np.asarray(get("zero_point"), dtype=np.float32)
    scale = _f(get("scale")).astype(np.float32)
    axis = int(get("axis"))
    shape = [1] * q.ndim
    if scale.ndim:
        shape[axis] = -1
        scale = scale.reshape(shape)
    if zp.ndim:
        zp = zp.reshape(shape)
    return (q - zp) * scale


def op_cast(get, op):
    x = get("x")
    dt = get("dtype")
    if dt in ("int32", "int16", "int8", "int64"):
        xf = np.asarray(x)
        if xf.dtype.kind == "f":
            xf = np.trunc(xf)
        return xf.astype(_NP[dt])
    if dt == "bool":
        return np.asarray(x).astype(bool)
    return np.asarray(x).astype(np.float32)


def _bin(fn):
    def f(get, op):
        return fn(_f(get("x")), _f(get("y")))
    return f


def op_floor_div(get, op):
    x, y = get("x"), get("y")
    return np.floor_divide(x, y)


def op_transpose(get, op):
    return np.transpose(get("x"), get("perm"))


def op_reshape(get, op):
    return np.reshape(get("x"), tuple(int(v) for v in get("shape")))


def op_expand_dims(get, op):
    x = get("x")
    for a in sorted(int(a) for a in get("axes")):
        x = np.expand_dims(x, a if a >= 0 else a + x.ndim + 1)
    return x


def op_squeeze(get, op):
    axes = get("axes")
    x = get("x")
    return np.squeeze(x, axis=tuple(int(a) for a in axes)) if axes is not None else np.squeeze(x)


def op_conv(get, op):
    x = _f(get("x")).astype(np.float32)
    w = _f(get("weight")).astype(np.float32)
    b = get("bias")
    groups = int(get("groups", 1))
    nsp = x.ndim - 2
    strides = [int(s) for s in get("strides", [1] * nsp)]
    dil = [int(d) for d in get("dilations", [1] * nsp)]
    pad_type = get("pad_type", "valid")
    pad = [int(p) for p in get("pad", [0] * 2 * nsp)]
    if pad_type == "valid":
        pad = [0] * 2 * nsp
    elif pad_type == "same":
        pad = []
        for i in range(nsp):
            k = (w.shape[2 + i] - 1) * dil[i] + 1
            tot = max(0, k - 1)
            pad += [tot // 2, tot - tot // 2]
    x = np.pad(x, [(0, 0), (0, 0)] + [(pad[2 * i], pad[2 * i + 1]) for i in range(nsp)])
    # unify to 2D
    if nsp == 1:
        x = x[:, :, None, :]
        w = w[:, :, None, :]
        strides = [1] + strides
        dil = [1] + dil
    N, C, H, W = x.shape
    Co, Cg, kh, kw = w.shape
    oh = (H - (kh - 1) * dil[0] - 1) // strides[0] + 1
    ow = (W - (kw - 1) * dil[1] - 1) // strides[1] + 1
    out = np.zeros((N, Co, oh, ow), dtype=np.float32)
    cpg_out = Co // groups
    if groups == C and Cg == 1:  # depthwise
        wd = w.reshape(groups, cpg_out, kh, kw)
        for i in range(kh):
            for j in range(kw):
                patch = x[:, :, i * dil[0]: i * dil[0] + strides[0] * (oh - 1) + 1: strides[0],
                          j * dil[1]: j * dil[1] + strides[1] * (ow - 1) + 1: strides[1]]
                out += np.repeat(patch, cpg_out, axis=1) * wd.reshape(Co, kh, kw)[None, :, i, j, None, None]
    else:
        for g in range(groups):
            xg = x[:, g * Cg:(g + 1) * Cg]
            wg = w[g * cpg_out:(g + 1) * cpg_out]
            acc = np.zeros((N, cpg_out, oh, ow), dtype=np.float32)
            for i in range(kh):
                for j in range(kw):
                    patch = xg[:, :, i * dil[0]: i * dil[0] + strides[0] * (oh - 1) + 1: strides[0],
                               j * dil[1]: j * dil[1] + strides[1] * (ow - 1) + 1: strides[1]]
                    acc += np.einsum("nchw,oc->nohw", patch, wg[:, :, i, j], optimize=True)
            out[:, g * cpg_out:(g + 1) * cpg_out] = acc
    if b is not None:
        out += _f(b).astype(np.float32).reshape(1, -1, 1, 1)
    if nsp == 1:
        out = out[:, :, 0, :]
    return out


def op_linear(get, op):
    x = _f(get("x"))
    w = _f(get("weight"))
    b = get("bias")
    y = x @ w.T
    return y + _f(b) if b is not None else y


def op_matmul(get, op):
    x, y = _f(get("x")), _f(get("y"))
    if get("transpose_x", False):
        x = np.swapaxes(x, -1, -2)
    if get("transpose_y", False):
        y = np.swapaxes(y, -1, -2)
    return np.matmul(x, y)


def op_layer_norm(get, op):
    x = _f(get("x"))
    axes = tuple(int(a) for a in get("axes"))
    eps = float(get("epsilon", 1e-5))
    mu = x.mean(axis=axes, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    g, b = get("gamma"), get("beta")
    if g is not None:
        y = y * _f(g)
    if b is not None:
        y = y + _f(b)
    return y


def op_softmax(get, op):
    x = _f(get("x"))
    ax = int(get("axis", -1))
    m = x.max(axis=ax, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=ax, keepdims=True)


def op_split(get, op):
    x = get("x")
    ax = int(get("axis"))
    sizes = get("split_sizes")
    if sizes is not None:
        idx = np.cumsum(sizes)[:-1]
        return np.split(x, idx, axis=ax)
    return np.split(x, int(get("num_splits")), axis=ax)


def op_pad(get, op):
    x = get("x")
    pad = [int(p) for p in get("pad")]
    mode = get("mode", "constant")
    n = len(pad) // 2
    widths = [(0, 0)] * (x.ndim - n) + [(pad[2 * i], pad[2 * i + 1]) for i in range(n)]
    if mode == "constant":
        return np.pad(x, widths, constant_values=float(get("constant_val", 0.0)))
    return np.pad(x, widths, mode={"reflect": "reflect", "replicate": "edge"}[mode])


def op_slice(get, op):
    x = get("x")
    r = x.ndim
    begin = list(get("begin"))
    end = list(get("end"))
    stride = list(get("stride", [1] * r))
    bm = list(get("begin_mask", [False] * r))
    em = list(get("end_mask", [False] * r))
    sm = list(get("squeeze_mask", [False] * r))
    sl = []
    for i in range(r):
        b = None if bm[i] else int(begin[i])
        e = None if em[i] else int(end[i])
        sl.append(slice(b, e, int(stride[i])))
    y = x[tuple(sl)]
    sq = tuple(i for i in range(r) if sm[i])
    return np.squeeze(y, axis=sq) if sq else y


def op_gather(get, op):
    x, idx = get("x"), np.asarray(get("indices")).astype(np.int64)
    ax = int(get("axis", 0))
    return np.take(x, idx, axis=ax)


def op_select(get, op):
    return np.where(get("cond"), get("a"), get("b"))


def op_stack(get, op):
    return np.stack(get("values"), axis=int(get("axis")))


def op_concat(get, op):
    return np.concatenate(get("values"), axis=int(get("axis")))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def op_lstm(get, op):
    """Core ML lstm, single direction. Gate order in weights/bias: input, forget, output, cell(z)."""
    x = _f(get("x"))  # [T, B, I]
    wih, whh = _f(get("weight_ih")), _f(get("weight_hh"))
    b = get("bias")
    b = _f(b) if b is not None else np.zeros(wih.shape[0], np.float32)
    h, c = _f(get("initial_h")), _f(get("initial_c"))
    H = whh.shape[1]
    outs = []
    for t in range(x.shape[0]):
        g = x[t] @ wih.T + h @ whh.T + b
        i, f, o, z = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
        c = _sig(f) * c + _sig(i) * np.tanh(z)
        h = _sig(o) * np.tanh(c)
        outs.append(h)
    seq = np.stack(outs) if get("output_sequence", False) else h[None]
    return [seq, h, c]


def op_reduce(fn):
    def f(get, op):
        x = _f(get("x"))
        axes = get("axes")
        keep = bool(get("keep_dims", False))
        ax = tuple(int(a) for a in axes) if axes is not None else None
        return fn(x, axis=ax, keepdims=keep)
    return f


def op_argmax(get, op):
    x = _f(get("x"))
    ax = int(get("axis", -1))
    keep = bool(get("keep_dims", False))
    r = np.argmax(x, axis=ax)
    return np.expand_dims(r, ax) if keep else r


def op_log(get, op):
    return np.log(_f(get("x")) + float(get("epsilon", 0.0)))


def op_tile(get, op):
    return np.tile(get("x"), tuple(int(r) for r in get("reps")))


OPS = {
    "const": op_const,
    "constexpr_blockwise_shift_scale": op_blockwise,
    "constexpr_lut_to_dense": op_lut,
    "constexpr_affine_dequantize": op_affine,
    "cast": op_cast,
    "add": _bin(np.add), "sub": _bin(np.subtract), "mul": _bin(np.multiply), "real_div": _bin(np.divide),
    "pow": _bin(np.power),
    "less": _bin(np.less), "greater_equal": _bin(np.greater_equal),
    "floor_div": op_floor_div,
    "transpose": op_transpose, "reshape": op_reshape, "expand_dims": op_expand_dims, "squeeze": op_squeeze,
    "conv": op_conv, "linear": op_linear, "matmul": op_matmul, "layer_norm": op_layer_norm,
    "softmax": op_softmax, "split": op_split, "pad": op_pad, "slice_by_index": op_slice,
    "gather": op_gather, "select": op_select, "stack": op_stack, "concat": op_concat, "lstm": op_lstm,
    "relu": lambda g, o: np.maximum(_f(g("x")), 0),
    "silu": lambda g, o: _f(g("x")) * _sig(_f(g("x"))),
    "sigmoid": lambda g, o: _sig(_f(g("x"))),
    "sqrt": lambda g, o: np.sqrt(_f(g("x"))),
    "identity": lambda g, o: g("x"),
    "reduce_sum": op_reduce(np.sum), "reduce_max": op_reduce(np.max),
    "reduce_argmax": op_argmax, "log": op_log, "tile": op_tile,
}
