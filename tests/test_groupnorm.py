# tests/test_groupnorm.py

import pytest
import torch

from quack.groupnorm import groupnorm_fwd, groupnorm, groupnorm_ref


TOLERANCES = {
    torch.bfloat16: (1e-1, 1e-1),
    torch.float16: (1e-2, 1e-2),
    torch.float32: (1e-4, 1e-4),
}


def _valid_combo(C, num_groups):
    return C % num_groups == 0


@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [1, 4, 16])
@pytest.mark.parametrize(
    "C,spatial,num_groups",
    [
        (32, (8, 8), 1),
        (32, (8, 8), 4),
        (32, (8, 8), 32),
        (64, (8, 8), 4),
        (64, (8, 8), 8),
        (128, (8, 8), 4),
        (128, (16, 16), 8),
        (32, (1,), 4),
        (64, (1,), 1),
    ],
)
def test_groupnorm_fwd(N, C, spatial, num_groups, input_dtype, eps):
    """Test GroupNorm forward pass against torch.nn.functional.group_norm."""
    if not _valid_combo(C, num_groups):
        pytest.skip(f"C={C} not divisible by num_groups={num_groups}")
    device = "cuda"
    atol, rtol = TOLERANCES[input_dtype]

    torch.manual_seed(0)
    x = torch.randn(N, C, *spatial, device=device, dtype=input_dtype)
    weight = torch.randn(C, device=device, dtype=input_dtype)
    bias = torch.randn(C, device=device, dtype=input_dtype)

    out, mean, rstd = groupnorm_fwd(x, num_groups, weight, bias, eps=eps)
    out_ref = groupnorm_ref(x, num_groups, weight, bias, eps=eps)

    assert out.shape == x.shape
    assert out.dtype == input_dtype
    assert mean.shape == (N, num_groups)
    assert mean.dtype == torch.float32
    assert rstd.shape == (N, num_groups)
    assert rstd.dtype == torch.float32

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize(
    "C,spatial,num_groups",
    [
        (32, (8, 8), 1),
        (32, (8, 8), 4),
        (64, (8, 8), 4),
        (32, (1,), 4),
        (64, (1,), 1),
    ],
)
def test_groupnorm_bwd(N, C, spatial, num_groups, input_dtype):
    """Test GroupNorm backward pass against PyTorch autograd reference."""
    if not _valid_combo(C, num_groups):
        pytest.skip(f"C={C} not divisible by num_groups={num_groups}")
    device = "cuda"
    eps = 1e-5
    atol, rtol = TOLERANCES[input_dtype]

    torch.manual_seed(0)
    x = torch.randn(N, C, *spatial, device=device, dtype=input_dtype, requires_grad=True)
    weight = torch.randn(C, device=device, dtype=input_dtype, requires_grad=True)
    bias = torch.randn(C, device=device, dtype=input_dtype, requires_grad=True)
    dout = torch.randn(N, C, *spatial, device=device, dtype=input_dtype)

    # Reference via torch.nn.functional.group_norm
    x_ref = x.detach().clone().float().requires_grad_()
    w_ref = weight.detach().clone().float().requires_grad_()
    b_ref = bias.detach().clone().float().requires_grad_()
    out_ref = torch.nn.functional.group_norm(x_ref, num_groups, w_ref, b_ref, eps)
    out_ref.backward(dout.float())

    # Our implementation
    out = groupnorm(x, num_groups, weight, bias, eps)
    out.backward(dout)

    torch.testing.assert_close(x.grad, x_ref.grad.to(input_dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(weight.grad, w_ref.grad.to(input_dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(bias.grad, b_ref.grad.to(input_dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_groupnorm_optional_params(has_weight, has_bias):
    """Test GroupNorm with optional weight/bias."""
    device = "cuda"
    N, C, H, W = 2, 32, 8, 8
    num_groups = 4
    eps = 1e-5

    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device, dtype=torch.float32) if has_weight else None
    bias = torch.randn(C, device=device, dtype=torch.float32) if has_bias else None

    out, mean, rstd = groupnorm_fwd(x, num_groups, weight, bias, eps=eps)
    out_ref = groupnorm_ref(x, num_groups, weight, bias, eps=eps)

    assert out.shape == x.shape
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


def test_groupnorm_3d_input():
    """Test GroupNorm with 3D input (N, C, L)."""
    device = "cuda"
    N, C, L = 4, 64, 128
    num_groups = 8

    x = torch.randn(N, C, L, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)

    out, mean, rstd = groupnorm_fwd(x, num_groups, weight, bias)
    out_ref = groupnorm_ref(x, num_groups, weight, bias)

    assert out.shape == (N, C, L)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


def test_groupnorm_5d_input():
    """Test GroupNorm with 5D input (N, C, D, H, W)."""
    device = "cuda"
    N, C, D, H, W = 2, 32, 4, 8, 8
    num_groups = 4

    x = torch.randn(N, C, D, H, W, device=device, dtype=torch.float32)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)

    out, mean, rstd = groupnorm_fwd(x, num_groups, weight, bias)
    out_ref = groupnorm_ref(x, num_groups, weight, bias)

    assert out.shape == (N, C, D, H, W)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


def test_groupnorm_input_validation():
    """Test input validation and error handling."""
    device = "cuda"

    # 2D input should fail (needs at least 3D)
    x_2d = torch.randn(32, 64, device=device, dtype=torch.float16)
    with pytest.raises(AssertionError, match="at least 3D"):
        groupnorm_fwd(x_2d, num_groups=4)

    # C not divisible by num_groups
    x = torch.randn(2, 33, 8, 8, device=device, dtype=torch.float16)
    with pytest.raises(AssertionError, match="divisible"):
        groupnorm_fwd(x, num_groups=4)

    # CPU tensor
    x_cpu = torch.randn(2, 32, 8, 8, dtype=torch.float16)
    with pytest.raises(AssertionError, match="CUDA"):
        groupnorm_fwd(x_cpu, num_groups=4)

    # Unsupported dtype
    x_f64 = torch.randn(2, 32, 8, 8, device=device, dtype=torch.float64)
    with pytest.raises(AssertionError, match="Unsupported dtype"):
        groupnorm_fwd(x_f64, num_groups=4)

    # Wrong weight shape
    x = torch.randn(2, 32, 8, 8, device=device, dtype=torch.float16)
    w_bad = torch.randn(16, device=device, dtype=torch.float16)
    with pytest.raises(AssertionError, match="Weight must have shape"):
        groupnorm_fwd(x, num_groups=4, weight=w_bad)
