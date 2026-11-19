import pytest
import torch

import lgma.lu_kernel as lu_kernel


def test_fused_queries_defer_cpu_tensors():
    query = torch.randn(2, 2, 7, 16, dtype=torch.bfloat16)
    metrics = torch.randn(2, 3, 16, 16, dtype=torch.bfloat16)
    assert lu_kernel.fused_lu_queries(query, metrics) is None


@pytest.fixture
def triton_cuda():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA")
    if torch.cuda.get_device_capability()[0] < 8 or lu_kernel.triton is None:
        pytest.skip("requires Triton on Ampere or newer")


@pytest.mark.parametrize(
    "shape, nonanchors, dtype, strided",
    [
        ((3, 2, 143, 64), 3, torch.bfloat16, True),
        ((1, 1, 37, 33), 2, torch.float16, False),
    ],
)
def test_fused_queries_cuda_forward_and_backward(
    monkeypatch, triton_cuda, shape, nonanchors, dtype, strided,
):
    torch.manual_seed(29)
    batch, bases, time, dim = shape
    heads_per_base = nonanchors + 1
    heads = bases * heads_per_base
    launches = []

    class RecordLaunch:
        def __init__(self, kernel, name):
            self.kernel = kernel
            self.name = name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def recorded_launch(*args, **kwargs):
                launches.append(self.name)
                return launch(*args, **kwargs)

            return recorded_launch

    for name in (
        "_expand_lu_queries", "_grad_lu_base_queries", "_grad_lu_metrics",
        "_reduce_lu_metric_partials",
    ):
        monkeypatch.setattr(lu_kernel, name, RecordLaunch(getattr(lu_kernel, name), name))

    # Model projection layout and a sliced, transposed metric. The upstream
    # gradient also has a non-unit last stride and time-major ordering.
    if strided:
        query = torch.randn(batch, time, bases, dim * 2, device="cuda", dtype=dtype)
        query = query.transpose(1, 2)[..., ::2]
        metric_master = torch.randn(bases, nonanchors, dim, dim * 2, device="cuda")
        metric_master = metric_master[..., ::2].transpose(-1, -2)
        dy = torch.randn(batch, time, heads, dim * 2, device="cuda", dtype=dtype)
        dy = dy.transpose(1, 2)[..., 1::2]
        assert not query.is_contiguous() and not metric_master.is_contiguous()
        assert not dy.is_contiguous() and dy.stride(-1) == 2
    else:
        query = torch.randn(shape, device="cuda", dtype=dtype)
        metric_master = torch.randn(bases, nonanchors, dim, dim, device="cuda")
        dy = torch.randn(batch, heads, time, dim, device="cuda", dtype=dtype)

    query.requires_grad_()
    metric_master.div_(dim ** 0.5).requires_grad_()
    with torch.autocast("cuda", dtype=dtype):
        # Preserve noncontiguous metric strides at the kernel boundary and
        # test the cast backward to an FP32 upstream parameter.
        if strided:
            metric = metric_master.transpose(-1, -2).to(dtype).transpose(-1, -2)
            assert not metric.is_contiguous()
        else:
            metric = metric_master.to(dtype)
        actual = lu_kernel.fused_lu_queries(query, metric)
        assert actual is not None
        assert actual.grad_fn is not None
        assert type(actual.grad_fn).__name__ == "_FusedLUQueriesBackward"
        assert actual.shape == (batch, heads, time, dim)
        assert actual.dtype == dtype

    actual_grad_q, actual_grad_m = torch.autograd.grad(actual, (query, metric_master), dy)
    # Check the FP32 accumulator contract using the values quantized at the
    # kernel boundary. Round once at each output, including both gradients.
    ref_query = query.detach().float().requires_grad_()
    ref_metric = metric.detach().float().requires_grad_()
    transformed = torch.einsum("bcti,cnij->bcntj", ref_query, ref_metric)
    expected = torch.cat((ref_query.unsqueeze(2), transformed), dim=2).flatten(1, 2)
    expected_grad_q, expected_grad_m = torch.autograd.grad(
        expected, (ref_query, ref_metric), dy.float(),
    )
    assert launches[:3] == ["_expand_lu_queries", "_grad_lu_base_queries", "_grad_lu_metrics"]
    assert launches[3:] == (["_reduce_lu_metric_partials"] if strided else [])
    assert torch.equal(actual[:, ::heads_per_base], query)
    assert actual_grad_q.dtype == query.dtype
    assert actual_grad_m.dtype == metric_master.dtype

    half_atol = 0.0078125 if dtype == torch.bfloat16 else 0.0009765625
    torch.testing.assert_close(actual, expected.to(dtype), atol=half_atol, rtol=half_atol)
    torch.testing.assert_close(actual_grad_q, expected_grad_q.to(dtype), atol=half_atol, rtol=half_atol)
    torch.testing.assert_close(
        actual_grad_m, expected_grad_m.to(dtype).float(), atol=half_atol, rtol=half_atol,
    )


def test_fused_queries_defer_unsupported_cuda_inputs(triton_cuda):
    query = torch.randn(1, 2, 7, 16, device="cuda", dtype=torch.bfloat16)
    metrics = torch.randn(2, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    assert lu_kernel.fused_lu_queries(query.float(), metrics.float()) is None
    assert lu_kernel.fused_lu_queries(query, metrics.float()) is None
    assert lu_kernel.fused_lu_queries(query, metrics.cpu()) is None
    assert lu_kernel.fused_lu_queries(query[..., 0], metrics) is None
    assert lu_kernel.fused_lu_queries(query, metrics[:, :0]) is None
    assert lu_kernel.fused_lu_queries(query, metrics[:, :, :-1]) is None
