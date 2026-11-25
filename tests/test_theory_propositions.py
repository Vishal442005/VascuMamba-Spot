"""Mathematical property tests verifying Proposition 1, Proposition 2, and monochrome integrity."""

from __future__ import annotations

import numpy as np
import torch

from vascumamba_spot.models.ssm_scan import (
    SelectiveSSMCore,
    pure_torch_selective_scan,
    zoh_discretize,
)
from vascumamba_spot.preprocessing.landmarks import default_canonical_rois
from vascumamba_spot.preprocessing.optical_strain import extract_au_optical_strain
from vascumamba_spot.preprocessing.rppg_vascular import (
    bandpass_cardiac_signal,
    compute_hr_snr_db,
    extract_vascular_features,
    pos_chrominance_projection,
)


def test_proposition1_exponential_blink_suppression_bound() -> None:
    """Verify Proposition 1: under null arousal (g_t <= eps), blink state energy is O(eps)."""
    torch.manual_seed(0)
    bsz, t_blink, d_model, d_state = 1, 15, 8, 4
    mu = 0.5
    eps = 0.02
    delta_max = 1.0
    m_k = 5.0

    ssm = SelectiveSSMCore(d_model=d_model, d_state=d_state, min_decay_mu=mu, backend="pure_torch")
    a_neg = ssm.negative_a()
    assert torch.all(a_neg <= -mu + 1e-6)

    # Large kinematic blink burst u_blink bounded by M_k per timestep
    raw_u = torch.randn(bsz, t_blink, d_model)
    u_blink = raw_u * (m_k / torch.clamp(raw_u.norm(dim=-1, keepdim=True), min=1e-6))

    # Under multiplicative autonomic gating with g_t <= eps:
    delta_null = torch.full((bsz, t_blink, d_model), eps * delta_max)
    b_unit = torch.ones(bsz, t_blink, d_state) / np.sqrt(d_state)  # ||B||_2 = 1
    b_gated = eps * b_unit
    c_proj = torch.ones(bsz, t_blink, d_state)

    _, h_null = pure_torch_selective_scan(
        u_blink, delta_null, a_neg, b_gated, c_proj, return_final_state=True
    )
    _, h_aroused = pure_torch_selective_scan(
        u_blink,
        torch.full((bsz, t_blink, d_model), delta_max),
        a_neg,
        b_unit,
        c_proj,
        return_final_state=True,
    )

    assert h_null is not None and h_aroused is not None
    norm_null = float(h_null.norm().item())
    norm_aroused = float(h_aroused.norm().item())

    # Exact Proposition 1 ratio check: null arousal state norm is O(eps^2) with both Delta & B gated
    assert norm_null < 0.01 * norm_aroused, f"Expected strong blink suppression, got {norm_null} vs {norm_aroused}"


def test_proposition2_pos_illumination_and_melanin_invariance() -> None:
    """Verify Proposition 2: POS recovers cardiac BVP under severe illumination drift & melanin shift."""
    fps = 50.0
    duration_sec = 10.0
    t = np.arange(int(fps * duration_sec), dtype=np.float32) / fps
    true_hr_hz = 1.25  # 75 BPM
    pulse = np.sin(2.0 * np.pi * true_hr_hz * t)

    # Time-varying ambient illumination I(t) with strong 0.3 Hz drift and specular glare
    illum = 1.0 + 0.35 * np.sin(2.0 * np.pi * 0.3 * t)
    melanin_dc = np.array([95.0, 125.0, 105.0], dtype=np.float32)  # darker skin baseline
    pulsatile_vec = np.array([0.15, 1.0, 0.45], dtype=np.float32)  # hemoglobin absorption weights
    specular = 2.0 * np.cos(2.0 * np.pi * 0.4 * t)[:, None] * np.ones((1, 3), dtype=np.float32)

    rgb_ts = illum[:, None] * (melanin_dc[None, :] + pulse[:, None] * pulsatile_vec[None, :] + specular)
    bvp = pos_chrominance_projection(rgb_ts, fps=fps)
    bvp_filt = bandpass_cardiac_signal(bvp, fps=fps)
    est_bpm, snr_db = compute_hr_snr_db(bvp_filt, fps=fps)

    assert abs(est_bpm - true_hr_hz * 60.0) < 4.0, f"HR mismatch: {est_bpm} vs {true_hr_hz * 60.0}"
    assert snr_db > 8.0, f"Expected SNR > 8 dB, got {snr_db:.2f} dB"


def test_monochrome_and_optical_strain_pipeline() -> None:
    """Verify monochrome mode (SAMM-LV / SMIC-NIR) runs cleanly with has_rgb=0 and zero NaNs."""
    kin_masks, vasc_masks = default_canonical_rois(size=64)
    rng = np.random.default_rng(7)
    frames_mono = rng.integers(90, 150, size=(30, 64, 64), dtype=np.uint8)

    x_kin = extract_au_optical_strain(frames_mono, fps=200.0, kin_masks=kin_masks)
    x_phys, _ = extract_vascular_features(frames_mono, fps=200.0, vasc_masks=vasc_masks, color_mode="mono")

    assert x_kin.shape == (30, 144)
    assert x_phys.shape == (30, 36)
    assert np.all(x_phys[:, -1] == 0.0)
    assert not np.isnan(x_kin).any()
    assert not np.isnan(x_phys).any()


def test_parallel_scan_exactness_across_chunks() -> None:
    """Verify chunked Hillis-Steele parallel scan matches sequential recurrence across chunks."""
    torch.manual_seed(11)
    bsz, seq_len, d_model, d_state = 2, 300, 16, 4
    u = torch.randn(bsz, seq_len, d_model)
    delta = torch.rand(bsz, seq_len, d_model) * 0.5 + 0.01
    a_neg = -0.1 - torch.rand(d_model, d_state)
    b_proj = torch.randn(bsz, seq_len, d_state)
    c_proj = torch.randn(bsz, seq_len, d_state)
    d_skip = torch.randn(d_model)

    y_par, h_par = pure_torch_selective_scan(
        u, delta, a_neg, b_proj, c_proj, d_skip=d_skip, return_final_state=True, chunk_size=128
    )

    a_bar, b_bar = zoh_discretize(delta, a_neg, b_proj)
    bx = b_bar * u.unsqueeze(-1)
    h = torch.zeros(bsz, d_model, d_state)
    ys = []
    for t in range(seq_len):
        h = a_bar[:, t] * h + bx[:, t]
        ys.append(torch.sum(h * c_proj[:, t].unsqueeze(1), dim=-1))
    y_seq = torch.stack(ys, dim=1) + u * d_skip.view(1, 1, d_model)

    assert h_par is not None
    assert torch.allclose(y_par, y_seq, atol=1e-5, rtol=1e-5)
    assert torch.allclose(h_par, h, atol=1e-5, rtol=1e-5)


def test_triton_gpu_parity_when_cuda_available() -> None:
    """Verify fused Triton GPU scan forward and backward gradients match pure_torch on CUDA."""
    from vascumamba_spot.models.triton_scan import is_triton_mamba_available, triton_selective_scan

    if not is_triton_mamba_available():
        return

    torch.manual_seed(7)
    bsz, seq_len, d_model, d_state = 2, 200, 32, 8
    u1 = torch.randn(bsz, seq_len, d_model, device="cuda", requires_grad=True)
    d1 = (torch.rand(bsz, seq_len, d_model, device="cuda") * 0.4 + 0.01).detach().requires_grad_(True)
    a1 = (-0.1 - torch.rand(d_model, d_state, device="cuda")).detach().requires_grad_(True)
    b1 = torch.randn(bsz, seq_len, d_state, device="cuda", requires_grad=True)
    c1 = torch.randn(bsz, seq_len, d_state, device="cuda", requires_grad=True)
    ds1 = torch.randn(d_model, device="cuda", requires_grad=True)

    u2 = u1.detach().clone().requires_grad_(True)
    d2 = d1.detach().clone().requires_grad_(True)
    a2 = a1.detach().clone().requires_grad_(True)
    b2 = b1.detach().clone().requires_grad_(True)
    c2 = c1.detach().clone().requires_grad_(True)
    ds2 = ds1.detach().clone().requires_grad_(True)

    y_ref, _ = pure_torch_selective_scan(u1, d1, a1, b1, c1, d_skip=ds1)
    y_tri = triton_selective_scan(u2, d2, a2, b2, c2, d_skip=ds2)
    assert torch.allclose(y_tri, y_ref, atol=1e-4, rtol=1e-4)

    y_ref.sum().backward()
    y_tri.sum().backward()
    for g_tri, g_ref in zip([u2.grad, d2.grad, a2.grad, b2.grad, c2.grad, ds2.grad], [u1.grad, d1.grad, a1.grad, b1.grad, c1.grad, ds1.grad]):
        assert g_tri is not None and g_ref is not None
        assert torch.allclose(g_tri, g_ref, atol=1e-3, rtol=1e-3)
