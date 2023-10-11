import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CSB: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    # tl.device_print("aaa", start_m)
    # get the (b, h) location
    qk_offset = off_hz * stride_qh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qk_offset,
        shape=(N_CTX, BLOCK_DMODEL_QK),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL_QK),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qk_offset,
        shape=(BLOCK_DMODEL_QK, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL_QK, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL_V),
        strides=(stride_vn, stride_ve),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL_V),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.dot(q, k)

        if IS_CAUSAL:
            index = offs_m[:, None] - (start_n + offs_n[None, :])
            if USE_CSB:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                qk = tl.exp(s_index) * qk
            else:
                qk = tl.where(index >= 0, qk, 0)

        acc += tl.dot(qk.to(v.dtype), v)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL_V),
        strides=(stride_om, stride_oe),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL_V),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(q.dtype))


@triton.jit
def _bwd_kernel_kv(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_CSB: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q1 = Q + off_z * stride_qz + off_h * stride_qh
    K1 = K + off_z * stride_kz + off_h * stride_kh
    V1 = V + off_z * stride_vz + off_h * stride_vh
    DO1 = DO + off_z * stride_oz + off_h * stride_oh
    DQ1 = DQ + off_z * stride_qz + off_h * stride_qh
    DK1 = DK + off_z * stride_kz + off_h * stride_kh
    DV1 = DV + off_z * stride_vz + off_h * stride_vh

    # start of q
    if CAUSAL:
        lo = start_n * BLOCK_N
    else:
        lo = 0
    # initialize row/col offsets
    # seqlence offset
    offs_qm = lo + tl.arange(0, BLOCK_M)
    offs_kvn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # initialize pointers to value-like data
    q_ptrs = Q1 + (offs_qm[:, None] * stride_qm +
                   offs_qkk[None, :] * stride_qk)
    k_ptrs = K1 + (offs_kvn[:, None] * stride_kn +
                   offs_qkk[None, :] * stride_kk)
    v_ptrs = V1 + (offs_kvn[:, None] * stride_vn +
                   offs_ve[None, :] * stride_ve)
    do_ptrs = DO1 + (offs_qm[:, None] * stride_om +
                     offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ1 + (offs_qm[:, None] * stride_qm +
                     offs_qkk[None, :] * stride_qk)
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # loop over rows
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(q_ptrs)
        qk = tl.dot(q, tl.trans(k))
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_CSB:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                qk = qk * s
            else:
                qk = tl.where(index >= 0, qk, 0)

        p = qk
        # compute dv
        do = tl.load(do_ptrs)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # dp = tl.dot(do.to(v.dtype), tl.trans(v))
        dp = tl.dot(do, tl.trans(v.to(do.dtype)))
        if CAUSAL:
            if USE_CSB:
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)

        dk += tl.dot(tl.trans(dp.to(q.dtype)), q).to(tl.float32)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_om
    # write-back
    dv_ptrs = DV1 + (offs_kvn[:, None] * stride_vn +
                     offs_ve[None, :] * stride_ve)
    dk_ptrs = DK1 + (offs_kvn[:, None] * stride_kn +
                     offs_qkk[None, :] * stride_kk)
    tl.store(dv_ptrs, dv)
    tl.store(dk_ptrs, dk)

    start_n = num_block - 1 - tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q1 = Q + off_z * stride_qz + off_h * stride_qh
    K1 = K + off_z * stride_kz + off_h * stride_kh
    V1 = V + off_z * stride_vz + off_h * stride_vh
    DO1 = DO + off_z * stride_oz + off_h * stride_oh
    DQ1 = DQ + off_z * stride_qz + off_h * stride_qh
    DK1 = DK + off_z * stride_kz + off_h * stride_kh
    DV1 = DV + off_z * stride_vz + off_h * stride_vh

    # start of q
    if CAUSAL:
        lo = start_n * BLOCK_N
    else:
        lo = 0
    # initialize row/col offsets
    # seqlence offset
    offs_qm = lo + tl.arange(0, BLOCK_M)
    offs_kvn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # initialize pointers to value-like data
    q_ptrs = Q1 + (offs_qm[:, None] * stride_qm +
                   offs_qkk[None, :] * stride_qk)
    k_ptrs = K1 + (offs_kvn[:, None] * stride_kn +
                   offs_qkk[None, :] * stride_kk)
    v_ptrs = V1 + (offs_kvn[:, None] * stride_vn +
                   offs_ve[None, :] * stride_ve)
    do_ptrs = DO1 + (offs_qm[:, None] * stride_om +
                     offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ1 + (offs_qm[:, None] * stride_qm +
                     offs_qkk[None, :] * stride_qk)
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)

    # loop over rows
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(q_ptrs)
        qk = tl.dot(q, tl.trans(k))
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_CSB:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                qk = qk * s
            else:
                qk = tl.where(index >= 0, qk, 0)

        p = qk
        # compute dv
        do = tl.load(do_ptrs)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # dp = tl.dot(do.to(v.dtype), tl.trans(v))
        dp = tl.dot(do, tl.trans(v.to(do.dtype)))
        if CAUSAL:
            if USE_CSB:
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)

        dk += tl.dot(tl.trans(dp.to(q.dtype)), q).to(tl.float32)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_om
    # write-back
    dv_ptrs = DV1 + (offs_kvn[:, None] * stride_vn +
                     offs_ve[None, :] * stride_ve)
    dk_ptrs = DK1 + (offs_kvn[:, None] * stride_kn +
                     offs_qkk[None, :] * stride_kk)
    tl.store(dv_ptrs, dv)
    tl.store(dk_ptrs, dk)


@triton.jit
def _bwd_kernel_q(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ve,
    stride_oz,
    stride_oh,
    stride_om,
    stride_oe,
    stride_sh,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_CSB: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    K1 = K + off_z * stride_kz + off_h * stride_kh
    V1 = V + off_z * stride_vz + off_h * stride_vh
    DO1 = DO + off_z * stride_oz + off_h * stride_oh
    DQ1 = DQ + off_z * stride_qz + off_h * stride_qh
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # row block index
    offs_qm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # do
    do_ptrs = DO1 + (offs_qm[:, None] * stride_om +
                     offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ1 + (offs_qm[:, None] * stride_qm +
                     offs_qkk[None, :] * stride_qk)

    do = tl.load(do_ptrs)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    offs_m_curr = start_m * BLOCK_M + offs_m

    for start_n in range(lo, hi, BLOCK_N):
        offs_kvn = start_n + tl.arange(0, BLOCK_N)
        # initialize pointers to value-like data
        k_ptrs = K1 + (offs_kvn[:, None] * stride_kn +
                       offs_qkk[None, :] * stride_kk)
        v_ptrs = V1 + (offs_kvn[:, None] * stride_vn +
                       offs_ve[None, :] * stride_ve)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # dp = do vT
        dp = tl.dot(do, tl.trans(v.to(do.dtype)))
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_CSB:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)

        dq += tl.dot(dp.to(k.dtype), k)

    tl.store(dq_ptrs, dq)

    start_m = num_block - 1 - tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    K1 = K + off_z * stride_kz + off_h * stride_kh
    V1 = V + off_z * stride_vz + off_h * stride_vh
    DO1 = DO + off_z * stride_oz + off_h * stride_oh
    DQ1 = DQ + off_z * stride_qz + off_h * stride_qh
    # feature offset
    offs_qkk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_ve = tl.arange(0, BLOCK_DMODEL_V)
    # row block index
    offs_m = tl.arange(0, BLOCK_M)
    # row block index
    offs_qm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # do
    do_ptrs = DO1 + (offs_qm[:, None] * stride_om +
                     offs_ve[None, :] * stride_oe)
    dq_ptrs = DQ1 + (offs_qm[:, None] * stride_qm +
                     offs_qkk[None, :] * stride_qk)

    do = tl.load(do_ptrs)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    offs_m_curr = start_m * BLOCK_M + offs_m

    for start_n in range(lo, hi, BLOCK_N):
        offs_kvn = start_n + tl.arange(0, BLOCK_N)
        # initialize pointers to value-like data
        k_ptrs = K1 + (offs_kvn[:, None] * stride_kn +
                       offs_qkk[None, :] * stride_kk)
        v_ptrs = V1 + (offs_kvn[:, None] * stride_vn +
                       offs_ve[None, :] * stride_ve)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # dp = do vT
        dp = tl.dot(do, tl.trans(v.to(do.dtype)))
        if CAUSAL:
            index = offs_m_curr[:, None] - offs_kvn[None, :]
            if USE_CSB:
                S_block_ptr = S + off_h * stride_sh
                s = tl.load(S_block_ptr)
                s_index = s * index
                s_index = tl.where(s_index >= 0, -s_index, float("-inf"))
                s = tl.exp(s_index)
                dp = dp * s
            else:
                dp = tl.where(index >= 0, dp, 0)
        dq += tl.dot(dp.to(k.dtype), k)

    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, s):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError(
                "Flash attention currently only supported for compute capability >= 80"
            )
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # right
        o = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], v.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )

        BLOCK_M = 32
        BLOCK_N = 32

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        use_csb = s.shape[0] > 0

        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=Lk,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=Lv,
            IS_CAUSAL=causal,
            USE_CSB=use_csb,
        )

        ctx.save_for_backward(q, k, v, s)
        ctx.grid = grid
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_DMODEL_QK = Lk
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_DMODEL_V = Lv
        ctx.causal = causal
        ctx.use_csb = use_csb
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s = ctx.saved_tensors

        BLOCK_M = 32
        BLOCK_N = 32

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid_kv = (triton.cdiv(k.shape[2] // 2,
                               BLOCK_N), k.shape[0] * k.shape[1], 1)
        num_block = k.shape[2] // BLOCK_N
        _bwd_kernel_kv[grid_kv](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            num_block,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=ctx.BLOCK_DMODEL_QK,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=ctx.BLOCK_DMODEL_V,
            CAUSAL=ctx.causal,
            USE_CSB=ctx.use_csb,
        )

        BLOCK_M = 32
        BLOCK_N = 32

        grid_q = (triton.cdiv(q.shape[2] // 2,
                              BLOCK_M), q.shape[0] * q.shape[1], 1)
        num_block = q.shape[2] // BLOCK_M
        _bwd_kernel_q[grid_q](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            s.stride(0),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            num_block,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL_QK=ctx.BLOCK_DMODEL_QK,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL_V=ctx.BLOCK_DMODEL_V,
            CAUSAL=ctx.causal,
            USE_CSB=ctx.use_csb,
        )

        return dq.to(q.dtype), dk, dv, None, None


attention = _attention.apply

# treat for feature dim that is not power of 2
def lightning_attention(q, k, v, causal, ed):
    d = q.shape[-1]
    e = v.shape[-1]
    if d >= 128:
        m = 128
    else:
        m = 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o = attention(q1, k1, v, causal, ed)
        output = output + o

    return output


