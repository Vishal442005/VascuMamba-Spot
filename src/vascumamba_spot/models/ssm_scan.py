"""Dual-backend Zero-Order Hold (ZOH) selective state-space scan (pure_torch & cuda_mamba)."""

from __future__ import annotations

import torch
import torch.nn as nn


def is_cuda_mamba_available() -> bool:
    """Return True if CUDA and mamba_ssm selective_scan_fn are available."""
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm.ops.selective_scan_interface  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_ssm_backend(requested: str = "auto") -> str:
    """Resolve 'auto' into 'cuda_mamba' on CUDA+mamba_ssm or 'pure_torch' otherwise."""
    if requested == "pure_torch":
        return "pure_torch"
    if requested == "cuda_mamba":
        if not is_cuda_mamba_available():
            raise RuntimeError("cuda_mamba backend requested, but CUDA or mamba_ssm is unavailable.")
        return "cuda_mamba"
    return "cuda_mamba" if is_cuda_mamba_available() else "pure_torch"


def zoh_discretize(
    delta: torch.Tensor,
    a_neg: torch.Tensor,
    b_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute stable Zero-Order Hold discrete matrices A_bar and B_bar.

    Args:
        delta: Step size tensor of shape (B, T, D) > 0.
        a_neg: Strictly negative continuous state diagonal of shape (D, N).
        b_proj: Time-varying input projection of shape (B, T, N).

    Returns:
        a_bar: (B, T, D, N)
        b_bar: (B, T, D, N)
    """
    da = delta.unsqueeze(-1) * a_neg.unsqueeze(0).unsqueeze(0)  # (B, T, D, N)
    a_bar = torch.exp(da)
    # Numerically stable (exp(da) - 1) / da via expm1 + small-angle Taylor guard
    small = torch.abs(da) < 1e-4
    safe_da = torch.where(small, torch.ones_like(da), da)
    zoh_scale = torch.where(
        small,
        1.0 + 0.5 * da + (da * da) / 6.0,
        torch.expm1(safe_da) / safe_da,
    )
    b_bar = (delta.unsqueeze(-1) * zoh_scale) * b_proj.unsqueeze(2)  # (B, T, D, N)
    return a_bar, b_bar


def _parallel_scan_chunk(
    a_chunk: torch.Tensor,
    b_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Hillis-Steele parallel prefix scan on a chunk (B, K, D, N) in O(log K) steps."""
    k_len = a_chunk.shape[1]
    a_cur = a_chunk
    b_cur = b_chunk
    stride = 1
    while stride < k_len:
        a_prev = a_cur[:, :-stride]
        b_prev = b_cur[:, :-stride]
        a_right = a_cur[:, stride:] * a_prev
        b_right = a_cur[:, stride:] * b_prev + b_cur[:, stride:]
        a_cur = torch.cat([a_cur[:, :stride], a_right], dim=1)
        b_cur = torch.cat([b_cur[:, :stride], b_right], dim=1)
        stride *= 2
    return a_cur, b_cur


def pure_torch_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_neg: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor | None = None,
    return_final_state: bool = False,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exact differentiable ZOH selective scan in pure PyTorch using O(T)-memory chunked prefix scan.

    Discretizes per chunk so intermediate (B, K, D, N) tensors stay cache-resident on CPU/GPU.
    """
    bsz, seq_len, d_model = u.shape
    n_state = a_neg.shape[1]

    h_carry = torch.zeros(bsz, d_model, n_state, device=u.device, dtype=u.dtype)
    y_chunks: list[torch.Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        a_bar, b_bar = zoh_discretize(delta[:, start:end], a_neg, b_proj[:, start:end])
        bx = b_bar * u[:, start:end].unsqueeze(-1)
        a_pref, b_pref = _parallel_scan_chunk(a_bar, bx)
        h_chunk = b_pref + a_pref * h_carry.unsqueeze(1)  # (B, K, D, N)
        h_carry = h_chunk[:, -1]
        y_chunk = torch.sum(h_chunk * c_proj[:, start:end].unsqueeze(2), dim=-1)  # (B, K, D)
        y_chunks.append(y_chunk)

    y = torch.cat(y_chunks, dim=1)
    if d_skip is not None:
        y = y + u * d_skip.view(1, 1, d_model)
    return y, (h_carry if return_final_state else None)


def cuda_mamba_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_neg: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invoke mamba_ssm CUDA selective_scan_fn with identical tensor semantics."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    # selective_scan_fn expects (B, D, L) for u, delta; (D, N) for A; (B, N, L) for B, C
    u_bdl = u.transpose(1, 2).contiguous()
    delta_bdl = delta.transpose(1, 2).contiguous()
    b_bnl = b_proj.transpose(1, 2).contiguous()
    c_bnl = c_proj.transpose(1, 2).contiguous()
    y_bdl = selective_scan_fn(
        u_bdl,
        delta_bdl,
        a_neg,
        b_bnl,
        c_bnl,
        D=d_skip,
        z=None,
        delta_bias=None,
        delta_softplus=False,
    )
    return y_bdl.transpose(1, 2).contiguous()


class SelectiveSSMCore(nn.Module):
    """Stable selective state-space core enforcing Re(A) <= -mu < 0 via log parameterization."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        min_decay_mu: float = 0.05,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.min_decay_mu = min_decay_mu
        self.backend = resolve_ssm_backend(backend)

        # S4D/HiPPO-inspired initialization for negative eigenvalues: A = -mu - exp(A_log)
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.a_log = nn.Parameter(torch.log(a_init))
        self.d_skip = nn.Parameter(torch.ones(d_model))

    def negative_a(self) -> torch.Tensor:
        """Return strictly negative diagonal matrix A with Re(lambda_i(A)) <= -min_decay_mu."""
        return -self.min_decay_mu - torch.exp(self.a_log)

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run forward selective scan using CUDA kernel or pure-PyTorch reference."""
        a_neg = self.negative_a()
        if self.backend == "cuda_mamba" and u.is_cuda and not return_final_state:
            y = cuda_mamba_selective_scan(u, delta, a_neg, b_proj, c_proj, self.d_skip)
            return y, None
        return pure_torch_selective_scan(
            u, delta, a_neg, b_proj, c_proj, d_skip=self.d_skip, return_final_state=return_final_state
        )
