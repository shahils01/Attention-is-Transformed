"""Optional CUDA kernel for fixed-base LU query expansion during training."""

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _expand_lu_queries(
        Q, Metrics, Out,
        qb, qc, qt, qd, mc, mh, mi, mj,
        TIME: tl.constexpr, BASES: tl.constexpr, NONANCHORS: tl.constexpr,
        DIM: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_head = tl.program_id(1)
        heads_per_base: tl.constexpr = NONANCHORS + 1
        heads: tl.constexpr = BASES * heads_per_base
        batch = batch_head // heads
        head = batch_head % heads
        base = head // heads_per_base
        local_head = head % heads_per_base
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        mask = (t[:, None] < TIME) & (d[None, :] < DIM)

        q_offset = batch.to(tl.int64) * qb + base * qc
        q = tl.load(Q + q_offset + t[:, None] * qt + d[None, :] * qd, mask, 0)
        out_offset = (batch_head.to(tl.int64) * TIME + t[:, None]) * DIM + d[None, :]
        if local_head == 0:
            tl.store(Out + out_offset, q, mask)
        else:
            metric_offset = base * mc + (local_head - 1) * mh
            matrix_mask = (d[:, None] < DIM) & (d[None, :] < DIM)
            metric = tl.load(
                Metrics + metric_offset + d[:, None] * mi + d[None, :] * mj,
                matrix_mask, 0,
            )
            tl.store(Out + out_offset, tl.dot(q, metric), mask)


    @triton.jit
    def _grad_lu_base_queries(
        GradOut, Metrics, GradQ,
        gb, gh, gt, gd, mc, mh, mi, mj,
        TIME: tl.constexpr, BASES: tl.constexpr, NONANCHORS: tl.constexpr,
        DIM: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_base = tl.program_id(1)
        batch = batch_base // BASES
        base = batch_base % BASES
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        mask = (t[:, None] < TIME) & (d[None, :] < DIM)
        matrix_mask = (d[:, None] < DIM) & (d[None, :] < DIM)

        grad_head = GradOut + batch.to(tl.int64) * gb + base * (NONANCHORS + 1) * gh
        grad_tile = grad_head + t[:, None] * gt + d[None, :] * gd
        metric_transpose = Metrics + base * mc + d[:, None] * mj + d[None, :] * mi

        # Accumulate across heads in one tile, without storing per-head dQ.
        grad_q = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for local_head in range(NONANCHORS):
            dy = tl.load(grad_tile + (local_head + 1) * gh, mask, 0)
            metric_t = tl.load(metric_transpose + local_head * mh, matrix_mask, 0)
            grad_q = tl.dot(dy, metric_t, grad_q)

        anchor = tl.load(grad_tile, mask, 0).to(tl.float32)
        out_offset = (batch_base.to(tl.int64) * TIME + t[:, None]) * DIM + d[None, :]
        tl.store(GradQ + out_offset, grad_q + anchor, mask)


    @triton.jit
    def _grad_lu_metrics(
        Q, GradOut, GradMetrics,
        qb, qc, qt, qd, gb, gh, gt, gd,
        TIME: tl.constexpr, TOKENS: tl.constexpr, NONANCHORS: tl.constexpr,
        DIM: tl.constexpr, METRIC_ELEMENTS: tl.constexpr, TILES_D: tl.constexpr,
        SPLIT_T: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        split = tl.program_id(0)
        head = tl.program_id(1)
        base = head // NONANCHORS
        nonanchor_head = head + base + 1
        i = tl.program_id(2) // TILES_D * BLOCK_D + tl.arange(0, BLOCK_D)
        j = tl.program_id(2) % TILES_D * BLOCK_D + tl.arange(0, BLOCK_D)
        token = split * SPLIT_T + tl.arange(0, BLOCK_T)

        q_base = Q + base * qc + i[:, None] * qd
        grad_head = GradOut + nonanchor_head * gh + j[None, :] * gd
        grad_metric = tl.zeros((BLOCK_D, BLOCK_D), tl.float32)
        for start in range(0, SPLIT_T, BLOCK_T):
            tokens = token + start
            b = (tokens // TIME).to(tl.int64)
            t = tokens % TIME
            q_transpose = tl.load(
                q_base + b[None, :] * qb + t[None, :] * qt,
                (i[:, None] < DIM) & (tokens[None, :] < TOKENS), 0,
            )
            dy = tl.load(
                grad_head + b[:, None] * gb + t[:, None] * gt,
                (j[None, :] < DIM) & (tokens[:, None] < TOKENS), 0,
            )
            grad_metric = tl.dot(q_transpose, dy, grad_metric)

        out_offset = split * METRIC_ELEMENTS + head * DIM * DIM + i[:, None] * DIM + j[None, :]
        tl.store(GradMetrics + out_offset, grad_metric, (i[:, None] < DIM) & (j[None, :] < DIM))


    @triton.jit
    def _reduce_lu_metric_partials(
        Partials, GradMetrics,
        ELEMENTS: tl.constexpr, SPLITS: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr, BLOCK_ELEMENTS: tl.constexpr,
    ):
        e = tl.program_id(0) * BLOCK_ELEMENTS + tl.arange(0, BLOCK_ELEMENTS)
        split = tl.arange(0, BLOCK_SPLITS)
        partial = tl.load(
            Partials + split[:, None] * ELEMENTS + e[None, :],
            (split[:, None] < SPLITS) & (e[None, :] < ELEMENTS), 0,
        )
        tl.store(GradMetrics + e, tl.sum(partial, axis=0), e < ELEMENTS)


class _FusedLUQueries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_base, nonanchor_metrics):
        batch, bases, time, dim = q_base.shape
        nonanchors = nonanchor_metrics.size(1)
        heads = bases * (nonanchors + 1)
        out = q_base.new_empty((batch, heads, time, dim))
        _expand_lu_queries[(triton.cdiv(time, 32), batch * heads)](
            q_base, nonanchor_metrics, out,
            *q_base.stride(), *nonanchor_metrics.stride(),
            time, bases, nonanchors, dim, 32, max(16, triton.next_power_of_2(dim)),
            num_warps=4,
        )
        ctx.save_for_backward(q_base, nonanchor_metrics)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q_base, nonanchor_metrics = ctx.saved_tensors
        grad_output = grad_output.resolve_neg()
        batch, bases, time, dim = q_base.shape
        nonanchors = nonanchor_metrics.size(1)
        grad_q = None
        grad_metrics = None
        if ctx.needs_input_grad[0]:
            grad_q = q_base.new_empty(q_base.shape)
            _grad_lu_base_queries[(triton.cdiv(time, 32), batch * bases)](
                grad_output, nonanchor_metrics, grad_q,
                *grad_output.stride(), *nonanchor_metrics.stride(),
                time, bases, nonanchors, dim, 32, max(16, triton.next_power_of_2(dim)),
                num_warps=4,
            )

        if ctx.needs_input_grad[1]:
            grad_metrics = nonanchor_metrics.new_empty(nonanchor_metrics.shape)
            elements = grad_metrics.numel()
            tokens = batch * time
            # Parallelize Q^T @ dY over tokens while bounding partial storage.
            block_t = 64
            split_t = max(2, triton.cdiv(tokens, 32 * block_t)) * block_t
            splits = triton.cdiv(tokens, split_t)
            block_d = min(64, max(16, triton.next_power_of_2(dim)))
            tiles_d = triton.cdiv(dim, block_d)
            partials = grad_metrics
            if splits > 1:
                partials = grad_metrics.new_empty((splits, elements), dtype=torch.float32)
            _grad_lu_metrics[(splits, bases * nonanchors, tiles_d * tiles_d)](
                q_base, grad_output, partials,
                *q_base.stride(), *grad_output.stride(),
                time, tokens, nonanchors, dim, elements, tiles_d, split_t, block_t, block_d,
                num_warps=4,
            )
            if splits > 1:
                _reduce_lu_metric_partials[(triton.cdiv(elements, 128),)](
                    partials, grad_metrics, elements, splits, triton.next_power_of_2(splits), 128,
                    num_warps=4,
                )
        return grad_q, grad_metrics


def fused_lu_queries(q_base: torch.Tensor, nonanchor_metrics: torch.Tensor) -> torch.Tensor | None:
    """Return [B, H, T, D] fused queries or None for inputs requiring the torch path.

    Queries are [B, C, T, D] and metrics are [C, H/C - 1, D, D]. Both inputs
    have the same half-precision dtype. Each group's first head is an exact
    copy of its base query, and each subsequent head is a right matrix product.
    """
    if (
        triton is None
        or torch.version.hip is not None
        or not q_base.is_cuda
        or nonanchor_metrics.device != q_base.device
        or q_base.layout != torch.strided
        or nonanchor_metrics.layout != torch.strided
        or q_base.dtype not in (torch.float16, torch.bfloat16)
        or nonanchor_metrics.dtype != q_base.dtype
        or q_base.is_neg()
        or nonanchor_metrics.is_neg()
        or q_base.ndim != 4
        or nonanchor_metrics.ndim != 4
    ):
        return None

    batch, bases, time, dim = q_base.shape
    metric_bases, nonanchors, rows, columns = nonanchor_metrics.shape
    if (
        min(batch, bases, time, dim, nonanchors) == 0
        or dim > 128
        or (metric_bases, rows, columns) != (bases, dim, dim)
        or batch * bases * (nonanchors + 1) > 65535
        or torch.cuda.get_device_capability(q_base.device)[0] < 8
    ):
        return None

    with torch.cuda.device(q_base.device):
        return _FusedLUQueries.apply(q_base, nonanchor_metrics)
