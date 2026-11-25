"""Fused Triton GPU selective scan kernel with exact O(T) forward and backward passes."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


def is_triton_mamba_available() -> bool:
    """Return True if CUDA and Triton are available for fused GPU selective scan."""
    return bool(HAS_TRITON and torch.cuda.is_available())


if HAS_TRITON:

    @triton.jit
    def _ssm_fwd_kernel(
        a_ptr,
        bx_ptr,
        c_ptr,
        y_ptr,
        h_ptr,
        T,
        D,
        N,
        stride_ab,
        stride_at,
        stride_ad,
        stride_an,
        stride_cb,
        stride_ct,
        stride_cn,
        stride_yb,
        stride_yt,
        stride_yd,
        SAVE_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        h = tl.zeros([BLOCK_N], dtype=tl.float32)
        base_a = pid_b * stride_ab + pid_d * stride_ad
        base_c = pid_b * stride_cb
        base_y = pid_b * stride_yb + pid_d * stride_yd

        for t in range(0, T):
            idx_a = base_a + t * stride_at + offs_n * stride_an
            a_val = tl.load(a_ptr + idx_a, mask=n_mask, other=0.0).to(tl.float32)
            bx_val = tl.load(bx_ptr + idx_a, mask=n_mask, other=0.0).to(tl.float32)
            c_val = tl.load(c_ptr + base_c + t * stride_ct + offs_n * stride_cn, mask=n_mask, other=0.0).to(
                tl.float32
            )

            h = a_val * h + bx_val
            if SAVE_H:
                tl.store(h_ptr + idx_a, h, mask=n_mask)
            y_val = tl.sum(h * c_val, axis=0)
            tl.store(y_ptr + base_y + t * stride_yt, y_val)

    @triton.jit
    def _ssm_bwd_kernel(
        grad_y_ptr,
        a_ptr,
        h_ptr,
        c_ptr,
        grad_a_ptr,
        grad_bx_ptr,
        T,
        D,
        N,
        stride_ab,
        stride_at,
        stride_ad,
        stride_an,
        stride_cb,
        stride_ct,
        stride_cn,
        stride_yb,
        stride_yt,
        stride_yd,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        dh = tl.zeros([BLOCK_N], dtype=tl.float32)
        a_next = tl.zeros([BLOCK_N], dtype=tl.float32)
        base_a = pid_b * stride_ab + pid_d * stride_ad
        base_c = pid_b * stride_cb
        base_y = pid_b * stride_yb + pid_d * stride_yd

        for rev_t in range(0, T):
            t = T - 1 - rev_t
            gy = tl.load(grad_y_ptr + base_y + t * stride_yt).to(tl.float32)
            c_val = tl.load(c_ptr + base_c + t * stride_ct + offs_n * stride_cn, mask=n_mask, other=0.0).to(
                tl.float32
            )
            dh = gy * c_val + a_next * dh

            idx_cur = base_a + t * stride_at + offs_n * stride_an
            tl.store(grad_bx_ptr + idx_cur, dh, mask=n_mask)

            if t > 0:
                h_prev = tl.load(
                    h_ptr + base_a + (t - 1) * stride_at + offs_n * stride_an, mask=n_mask, other=0.0
                ).to(tl.float32)
            else:
                h_prev = tl.zeros([BLOCK_N], dtype=tl.float32)

            tl.store(grad_a_ptr + idx_cur, dh * h_prev, mask=n_mask)
            a_next = tl.load(a_ptr + idx_cur, mask=n_mask, other=0.0).to(tl.float32)


class _TritonScanFunction(torch.autograd.Function):
    """Differentiable Triton wrapper for ZOH recurrence h_t = a_t h_{t-1} + bx_t."""

    @staticmethod
    def forward(ctx, a_bar: torch.Tensor, bx: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
        a_bar_c = a_bar.contiguous()
        bx_c = bx.contiguous()
        c_c = c_proj.contiguous()
        bsz, seq_len, d_model, n_state = a_bar_c.shape
        block_n = triton.next_power_of_2(n_state)

        y = torch.empty((bsz, seq_len, d_model), device=a_bar_c.device, dtype=torch.float32)
        need_bwd = any(ctx.needs_input_grad)
        h_all = (
            torch.empty_like(a_bar_c, dtype=torch.float32)
            if need_bwd
            else torch.empty((1,), device=a_bar_c.device, dtype=torch.float32)
        )

        grid = (bsz, d_model)
        _ssm_fwd_kernel[grid](
            a_bar_c,
            bx_c,
            c_c,
            y,
            h_all,
            seq_len,
            d_model,
            n_state,
            *a_bar_c.stride(),
            *c_c.stride(),
            *y.stride(),
            SAVE_H=need_bwd,
            BLOCK_N=block_n,
        )
        if need_bwd:
            ctx.save_for_backward(a_bar_c, h_all, c_c)
        return y.to(bx.dtype)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        a_bar_c, h_all, c_c = ctx.saved_tensors
        grad_y_c = grad_y.contiguous()
        bsz, seq_len, d_model, n_state = a_bar_c.shape
        block_n = triton.next_power_of_2(n_state)

        grad_a = torch.empty_like(a_bar_c, dtype=torch.float32)
        grad_bx = torch.empty_like(a_bar_c, dtype=torch.float32)
        grid = (bsz, d_model)
        _ssm_bwd_kernel[grid](
            grad_y_c,
            a_bar_c,
            h_all,
            c_c,
            grad_a,
            grad_bx,
            seq_len,
            d_model,
            n_state,
            *a_bar_c.stride(),
            *c_c.stride(),
            *grad_y_c.stride(),
            BLOCK_N=block_n,
        )
        grad_c = torch.einsum("btd,btdn->btn", grad_y_c.float(), h_all)
        return grad_a.to(a_bar_c.dtype), grad_bx.to(a_bar_c.dtype), grad_c.to(c_c.dtype)


def triton_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_neg: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused Triton GPU ZOH selective scan without requiring mamba_ssm C++ builds."""
    from vascumamba_spot.models.ssm_scan import zoh_discretize

    a_bar, b_bar = zoh_discretize(delta, a_neg, b_proj)
    bx = b_bar * u.unsqueeze(-1)
    y = _TritonScanFunction.apply(a_bar, bx, c_proj)
    if d_skip is not None:
        y = y + u * d_skip.view(1, 1, -1)
    return y
