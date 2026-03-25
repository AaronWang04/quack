# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from quack.rmsnorm import _rmsnorm_fwd


def groupnorm_fwd(
    x: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """GroupNorm forward pass.

    Args:
        x: Input tensor of shape (N, C, *spatial). Must be contiguous.
        num_groups: Number of groups G. C must be divisible by G.
        weight: Optional affine weight of shape (C,).
        bias: Optional affine bias of shape (C,).
        eps: Small value for numerical stability.

    Returns:
        (out, mean, rstd) where out has same shape as x,
        mean and rstd have shape (N, G).
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() >= 3, "Input must be at least 3D (N, C, *spatial)"
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_dtypes, f"Unsupported dtype {x.dtype}"

    N, C = x.shape[0], x.shape[1]
    G = num_groups
    assert C % G == 0, f"C={C} must be divisible by num_groups={G}"
    D = C // G
    HxW = math.prod(x.shape[2:])

    if weight is not None:
        assert weight.shape == (C,), f"Weight must have shape ({C},), got {weight.shape}"
        assert weight.is_cuda, "Weight must be on CUDA"
    if bias is not None:
        assert bias.shape == (C,), f"Bias must have shape ({C},), got {bias.shape}"
        assert bias.is_cuda, "Bias must be on CUDA"

    orig_shape = x.shape
    reduction_dim = D * HxW
    x_reshaped = x.reshape(N * G, reduction_dim).contiguous()

    mean = torch.empty(N * G, device=x.device, dtype=torch.float32)
    rstd = torch.empty(N * G, device=x.device, dtype=torch.float32)

    if reduction_dim >= 32:
        out_reshaped = torch.empty_like(x_reshaped)
        _rmsnorm_fwd(x_reshaped, None, out_reshaped, None, rstd, mean, None, None, eps, True)
    else:
        # Fallback for very small reduction dims where the CuTe kernel is inaccurate
        x_f32 = x_reshaped.float()
        mean[:] = x_f32.mean(dim=1)
        var = ((x_f32 - mean.unsqueeze(1)) ** 2).mean(dim=1)
        rstd[:] = 1.0 / torch.sqrt(var + eps)
        out_reshaped = ((x_f32 - mean.unsqueeze(1)) * rstd.unsqueeze(1)).to(x.dtype)

    out = out_reshaped.reshape(orig_shape)
    if weight is not None:
        view_shape = (1, C) + (1,) * (x.dim() - 2)
        out = out * weight.view(view_shape)
    if bias is not None:
        view_shape = (1, C) + (1,) * (x.dim() - 2)
        out = out + bias.view(view_shape)

    return out, mean.reshape(N, G), rstd.reshape(N, G)


def groupnorm_bwd(
    dout: Tensor,
    x: Tensor,
    weight: Optional[Tensor],
    mean: Tensor,
    rstd: Tensor,
    num_groups: int,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """GroupNorm backward pass using PyTorch ops.

    Args:
        dout: Upstream gradient, same shape as x (N, C, *spatial).
        x: Saved input from forward, shape (N, C, *spatial).
        weight: Affine weight of shape (C,) or None.
        mean: Saved mean of shape (N, G).
        rstd: Saved rstd of shape (N, G).
        num_groups: Number of groups G.

    Returns:
        (dx, dgamma, dbeta)
    """
    N, C = x.shape[0], x.shape[1]
    G = num_groups
    D = C // G
    HxW = math.prod(x.shape[2:])
    K = D * HxW  # elements per group

    # Compute x_hat in float32
    x_reshaped = x.reshape(N * G, K).float()
    mean_flat = mean.reshape(N * G, 1)
    rstd_flat = rstd.reshape(N * G, 1)
    x_hat = (x_reshaped - mean_flat) * rstd_flat  # (N*G, K)

    # dgamma, dbeta: reduce over batch and spatial dims
    x_hat_orig = x_hat.reshape(x.shape)
    reduce_dims = [0] + list(range(2, x.dim()))
    dgamma = (dout.float() * x_hat_orig).sum(dim=reduce_dims) if weight is not None else None
    dbeta = dout.float().sum(dim=reduce_dims) if weight is not None else None

    # dx through normalization
    if weight is not None:
        view_shape = (1, C) + (1,) * (x.dim() - 2)
        dout_w = (dout.float() * weight.float().view(view_shape)).reshape(N * G, K)
    else:
        dout_w = dout.float().reshape(N * G, K)

    # LayerNorm backward: dx = (dy - mean(dy) - x_hat * mean(x_hat * dy)) * rstd
    mean_dy = dout_w.mean(dim=1, keepdim=True)
    mean_xhat_dy = (x_hat * dout_w).mean(dim=1, keepdim=True)
    dx = ((dout_w - mean_dy - x_hat * mean_xhat_dy) * rstd_flat).to(x.dtype)
    dx = dx.reshape(x.shape)

    if dgamma is not None:
        dgamma = dgamma.to(weight.dtype)
    if dbeta is not None:
        dbeta = dbeta.to(weight.dtype)

    return dx, dgamma, dbeta


class GroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_groups, weight, bias, eps):
        out, mean, rstd = groupnorm_fwd(x, num_groups, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.num_groups = num_groups
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dgamma, dbeta = groupnorm_bwd(dout, x, weight, mean, rstd, ctx.num_groups)
        return dx, None, dgamma, dbeta, None


def groupnorm(
    x: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """GroupNorm with automatic differentiation support.

    Args:
        x: Input tensor of shape (N, C, *spatial).
        num_groups: Number of groups G. C must be divisible by G.
        weight: Optional affine weight of shape (C,).
        bias: Optional affine bias of shape (C,).
        eps: Small value for numerical stability.

    Returns:
        Normalized output tensor of same shape as x.
    """
    return GroupNormFunction.apply(x, num_groups, weight, bias, eps)


def groupnorm_ref(
    x: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Reference GroupNorm using torch.nn.functional.group_norm."""
    return torch.nn.functional.group_norm(
        x.float(),
        num_groups,
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).to(x.dtype)
