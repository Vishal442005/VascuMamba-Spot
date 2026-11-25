# VascuMamba-Spot

**Continuous-Time Autonomic-Gated Neuro-Vascular and Kinematic Selective State-Space Modeling for Untrimmed Micro-Expression Spotting and Recognition**

VascuMamba-Spot unifies **12-ROI Cauchy optical strain rate tensors** and **7-ROI motion-debiased remote photoplethysmography (rPPG)** dynamics inside a **Continuous-Time Bidirectional Selective State-Space Model (Mamba)** running in linear $O(T)$ complexity. By using autonomic arousal to multiplicatively gate $(\Delta_t^{\text{phys}}, B_t^{\text{kin}})$ state transitions, VascuMamba-Spot suppresses non-emotional eyeblink and head-motion false positives without wearable contact sensors.

## Key Capabilities
- **Rigorous Multiplicative Autonomic Gating:** Continuous physical-time step scaling ($\Delta_t^{\text{phys}} = \Delta t_f \cdot g_t^{\text{aut}} \odot \text{softplus}(\cdot)$) invariant across 25 / 30 / 100 / 200 fps sensors, with formal exponential blink-suppression guarantees.
- **Dual-Regime Vascular Extractor (`rgb` vs. `mono`/`nir`):** Plane-Orthogonal-to-Skin (POS) chrominance rPPG for RGB benchmarks (`CAS(ME)³`, `CAS(ME)²`, `CASME II`, `SMIC-HS/VIS`, `MMEW`) and calibrated monochrome/BCG fallback for grayscale benchmarks (`SAMM`, `SAMM-LV`, `SMIC-NIR`).
- **Motion-Orthogonal Adaptive Noise Cancellation (`MNCFilter`):** Scrubs residual muscle-twitch light-angle contamination from vascular ROIs.
- **4M-Frame Self-Supervised Pretraining & Privileged BIOPAC Distillation:** Pretraining on unlabeled `CAS(ME)³` videos + distilling contact BIOPAC ECG/PPG/EDA teacher arousal into contactless rPPG student gates.
- **Dual SSM Backend:** Numerically verified parity between pure PyTorch (`pure_torch`) on CPU/GPU and fused CUDA `mamba_ssm` (`cuda_mamba`).

## Installation

### Local CPU / Development
```bash
pip install -e ".[dev]"
pytest
ruff check src tests scripts
```

### GPU Production (`CUDA 12.x + mamba-ssm`)
```bash
pip install -e ".[cuda,dev]"
```
Or build via Docker:
```bash
docker build -t vascumamba-spot:latest .
```

## Quick Demo (Runs Out-of-the-Box Without External Datasets)
```bash
python scripts/demo_infer.py --out-dir outputs/demo
python scripts/benchmark_scaling.py --lengths 250 500 1000 2000 4000
```

## Full Benchmark Reproduction Pipeline
1. **Obtain Official Academic Dataset Licenses** (`CAS(ME)³`, `CAS(ME)²`, `CASME II`, `SAMM-LV`, `SMIC`, `MMEW`). Raw datasets are never redistributed in this repository.
2. **Prepare Canonical Subject-Disjoint Manifests:**
   ```bash
   python scripts/prepare_manifests.py --dataset casme3 --data-root /path/to/CASME3 --out-manifest data/casme3.jsonl
   ```
3. **Extract Stabilized Dual-Stream Features & Run Phase I SNR/BIOPAC Audit:**
   ```bash
   python scripts/extract_features.py --manifest data/casme3.jsonl --out-cache cache/casme3.h5
   python scripts/audit_rppg_snr.py --manifest data/casme3.jsonl --cache cache/casme3.h5
   ```
4. **Optional 4M-Frame Neuro-Vascular SSL Pretraining:**
   ```bash
   python scripts/pretrain_ssl.py --manifest data/casme3_unlabeled.jsonl --cache cache/casme3_unlabeled.h5 --out-ckpt checkpoints/ssl_backbone.pt
   ```
5. **Run LOSO Benchmarks, Baselines, Ablations & Automated LaTeX Tables:**
   ```bash
   python scripts/train_loso.py --manifest data/casme3.jsonl --cache cache/casme3.h5 --ssl-ckpt checkpoints/ssl_backbone.pt --seeds 42 101 2026 --out-json outputs/casme3_loso.json
   python scripts/run_ablations.py --manifest data/casme3.jsonl --cache cache/casme3.h5 --out-json outputs/casme3_ablations.json
   python scripts/generate_paper_assets.py --ablation-json outputs/casme3_ablations.json --out-tex paper/tables/ablation_table.tex
   ```

## Citation
See [CITATION.bib](CITATION.bib).
