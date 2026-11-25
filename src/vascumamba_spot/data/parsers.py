"""Dataset-specific annotation parsers producing canonical VideoRecord manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vascumamba_spot.data.manifests import (
    ColorMode,
    ExpressionAnnotation,
    VideoRecord,
    map_emotion_3cls,
)

VIDEO_EXTS = (".avi", ".mp4", ".mkv", ".mov")


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if pd.isna(val):
            return default
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _count_frames(path: Path) -> int:
    """Count video frames if video file exists, or image files if frame directory."""
    if path.is_dir():
        return sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if path.is_file():
        import cv2

        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n)
    return 0


def _find_video_or_frame_dir(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        base = root / rel
        if base.is_dir():
            return base
        for ext in VIDEO_EXTS:
            cand = base.with_suffix(ext)
            if cand.is_file():
                return cand
    return None


def parse_casme2(root_dir: str | Path) -> list[VideoRecord]:
    """Parse CASME II (200 fps RGB clipped dataset) from standard coding spreadsheet."""
    root = Path(root_dir)
    coding_files = list(root.glob("CASME2-coding*.xlsx")) + list(root.glob("CASME2-coding*.csv"))
    if not coding_files:
        raise FileNotFoundError(f"Could not find CASME2-coding*.xlsx/csv under {root}")
    df = pd.read_excel(coding_files[0]) if coding_files[0].suffix == ".xlsx" else pd.read_csv(coding_files[0])

    records: list[VideoRecord] = []
    for _, row in df.iterrows():
        sub = f"sub{int(row['Subject']):02d}"
        ep = str(row["Filename"]).strip()
        vpath = _find_video_or_frame_dir(root, [f"CASME2_RAW_selected/{sub}/{ep}", f"Cropped/{sub}/{ep}", f"{sub}/{ep}"])
        if vpath is None:
            continue
        onset = _safe_int(row.get("OnsetFrame", 1))
        apex = _safe_int(row.get("ApexFrame", (onset + _safe_int(row.get("OffsetFrame", onset + 20))) // 2))
        offset = _safe_int(row.get("OffsetFrame", onset + 40))
        emo = str(row.get("Estimated Emotion", "others")).strip().lower()
        aus = tuple(str(row.get("Action Units", "")).split("+")) if pd.notna(row.get("Action Units")) else ()
        ann = ExpressionAnnotation(
            onset=max(0, onset - 1),
            apex=max(0, apex - 1),
            offset=max(0, offset - 1),
            expr_type="ME",
            emotion_raw=emo,
            emotion_3cls=map_emotion_3cls(emo),
            au_list=tuple(a.strip() for a in aus if a.strip()),
        )
        records.append(
            VideoRecord(
                video_id=f"casme2_{sub}_{ep}",
                dataset="casme2",
                subject_id=sub,
                color_mode="rgb",
                fps=200.0,
                num_frames=_count_frames(vpath),
                video_path=str(vpath),
                annotations=(ann,),
            )
        )
    return records


def parse_untrimmed_spreadsheet(
    root_dir: str | Path,
    dataset_name: str,
    coding_glob: str,
    fps: float,
    color_mode: ColorMode,
) -> list[VideoRecord]:
    """Generic parser for untrimmed datasets (CAS(ME)^2, CAS(ME)^3, SAMM-LV)."""
    root = Path(root_dir)
    coding_files = list(root.rglob(coding_glob))
    if not coding_files:
        raise FileNotFoundError(f"Could not find annotation table matching {coding_glob} under {root}")
    cf = coding_files[0]
    df = pd.read_excel(cf) if cf.suffix in {".xlsx", ".xls"} else pd.read_csv(cf)

    grouped: dict[tuple[str, str], list[ExpressionAnnotation]] = {}
    for _, row in df.iterrows():
        sub = str(row.get("Subject", row.get("subject", "unknown"))).strip()
        vid = str(row.get("Filename", row.get("Video", row.get("video_id", "")))).strip()
        if not vid:
            continue
        onset = _safe_int(row.get("Onset", row.get("OnsetFrame", 0)))
        apex = _safe_int(row.get("Apex", row.get("ApexFrame", onset)))
        offset = _safe_int(row.get("Offset", row.get("OffsetFrame", onset + 10)))
        etype_raw = str(row.get("Type", row.get("type", "me"))).lower()
        expr_type = "MaE" if "macro" in etype_raw or "mae" in etype_raw else "ME"
        emo = str(row.get("Emotion", row.get("Estimated Emotion", "others"))).strip().lower()
        ann = ExpressionAnnotation(
            onset=max(0, onset),
            apex=max(0, apex),
            offset=max(onset + 1, offset),
            expr_type=expr_type,
            emotion_raw=emo,
            emotion_3cls=map_emotion_3cls(emo),
        )
        grouped.setdefault((sub, vid), []).append(ann)

    records: list[VideoRecord] = []
    for (sub, vid), anns in sorted(grouped.items()):
        vpath = _find_video_or_frame_dir(root, [f"{sub}/{vid}", f"rawvideo/{sub}/{vid}", vid])
        if vpath is None:
            continue
        biopac_cand = root / "biopac" / sub / f"{vid}.mat"
        records.append(
            VideoRecord(
                video_id=f"{dataset_name}_{sub}_{vid}",
                dataset=dataset_name,
                subject_id=sub,
                color_mode=color_mode,
                fps=fps,
                num_frames=_count_frames(vpath),
                video_path=str(vpath),
                annotations=tuple(sorted(anns, key=lambda a: a.onset)),
                biopac_path=str(biopac_cand) if biopac_cand.exists() else None,
            )
        )
    return records


def parse_samm(root_dir: str | Path, fps: float = 200.0) -> list[VideoRecord]:
    """Parse clipped SAMM (200 fps monochrome) from SAMM_Micro_FACS_Codes*.xlsx/csv."""
    root = Path(root_dir)
    coding_files = list(root.rglob("SAMM_Micro_FACS_Codes*.xlsx")) + list(
        root.rglob("SAMM_Micro_FACS_Codes*.csv")
    )
    if not coding_files:
        raise FileNotFoundError(f"Could not find SAMM_Micro_FACS_Codes*.xlsx/csv under {root}")
    cf = coding_files[0]
    df = pd.read_excel(cf, header=13) if cf.suffix == ".xlsx" else pd.read_csv(cf)
    if "Subject" not in df.columns and cf.suffix == ".xlsx":
        df = pd.read_excel(cf)

    records: list[VideoRecord] = []
    for _, row in df.iterrows():
        sub = str(row.get("Subject", "")).strip().zfill(3)
        ep = str(row.get("Filename", "")).strip()
        if not ep or sub == "000":
            continue
        vpath = _find_video_or_frame_dir(root, [f"{sub}/{ep}", f"SAMM/{sub}/{ep}"])
        if vpath is None:
            continue
        onset = _safe_int(row.get("Onset Frame", row.get("Onset", 0)))
        apex = _safe_int(row.get("Apex Frame", row.get("Apex", onset)))
        offset = _safe_int(row.get("Offset Frame", row.get("Offset", onset + 20)))
        emo = str(row.get("Estimated Emotion", "others")).strip().lower()
        aus = tuple(str(row.get("Action Units", "")).split("+")) if pd.notna(row.get("Action Units")) else ()
        ann = ExpressionAnnotation(
            onset=max(0, onset),
            apex=max(0, apex),
            offset=max(onset + 1, offset),
            expr_type="ME",
            emotion_raw=emo,
            emotion_3cls=map_emotion_3cls(emo),
            au_list=tuple(a.strip() for a in aus if a.strip()),
        )
        records.append(
            VideoRecord(
                video_id=f"samm_{sub}_{ep}",
                dataset="samm",
                subject_id=sub,
                color_mode="mono",
                fps=fps,
                num_frames=_count_frames(vpath),
                video_path=str(vpath),
                annotations=(ann,),
            )
        )
    return records


def parse_smic(root_dir: str | Path, sensor: str = "hs") -> list[VideoRecord]:
    """Parse SMIC subset ('hs'=100fps RGB, 'vis'=25fps RGB, 'nir'=25fps NIR monochrome)."""
    sensor_l = sensor.lower()
    if sensor_l not in {"hs", "vis", "nir"}:
        raise ValueError(f"Unsupported SMIC sensor: {sensor}")
    fps = 100.0 if sensor_l == "hs" else 25.0
    color_mode: ColorMode = "nir" if sensor_l == "nir" else "rgb"

    root = Path(root_dir)
    sub_root = root / sensor_l.upper() if (root / sensor_l.upper()).exists() else root
    records: list[VideoRecord] = []

    for emo_dir_name in ("negative", "positive", "surprise"):
        for clip_dir in sorted(sub_root.rglob(f"*/micro/{emo_dir_name}/*")):
            if not clip_dir.is_dir():
                continue
            sub = clip_dir.parents[2].name
            n_frames = _count_frames(clip_dir)
            if n_frames < 2:
                continue
            ann = ExpressionAnnotation(
                onset=0,
                apex=n_frames // 2,
                offset=n_frames - 1,
                expr_type="ME",
                emotion_raw=emo_dir_name,
                emotion_3cls=emo_dir_name,
            )
            records.append(
                VideoRecord(
                    video_id=f"smic_{sensor_l}_{sub}_{clip_dir.name}",
                    dataset=f"smic_{sensor_l}",
                    subject_id=sub,
                    color_mode=color_mode,
                    fps=fps,
                    num_frames=n_frames,
                    video_path=str(clip_dir),
                    annotations=(ann,),
                )
            )
    return records


def parse_mmew(root_dir: str | Path, fps: float = 90.0) -> list[VideoRecord]:
    """Parse MMEW (90 fps RGB) clipped micro-expression benchmark."""
    root = Path(root_dir)
    coding_files = list(root.rglob("MMEW_Micro*.xlsx")) + list(root.rglob("MMEW_Micro*.csv"))
    if not coding_files:
        raise FileNotFoundError(f"Could not find MMEW_Micro*.xlsx/csv under {root}")
    cf = coding_files[0]
    df = pd.read_excel(cf) if cf.suffix == ".xlsx" else pd.read_csv(cf)

    records: list[VideoRecord] = []
    for _, row in df.iterrows():
        sub = str(row.get("Subject", "unknown")).strip()
        ep = str(row.get("Filename", "")).strip()
        emo = str(row.get("Estimated Emotion", "others")).strip().lower()
        vpath = _find_video_or_frame_dir(root, [f"Micro_Expression/{emo}/{ep}", f"{sub}/{ep}", ep])
        if vpath is None:
            continue
        onset = _safe_int(row.get("OnsetFrame", 1))
        apex = _safe_int(row.get("ApexFrame", onset + 10))
        offset = _safe_int(row.get("OffsetFrame", onset + 25))
        ann = ExpressionAnnotation(
            onset=max(0, onset - 1),
            apex=max(0, apex - 1),
            offset=max(onset, offset - 1),
            expr_type="ME",
            emotion_raw=emo,
            emotion_3cls=map_emotion_3cls(emo),
        )
        records.append(
            VideoRecord(
                video_id=f"mmew_{sub}_{ep}",
                dataset="mmew",
                subject_id=sub,
                color_mode="rgb",
                fps=fps,
                num_frames=_count_frames(vpath),
                video_path=str(vpath),
                annotations=(ann,),
            )
        )
    return records


def parse_casme3_unlabeled(root_dir: str | Path, fps: float = 30.0) -> list[VideoRecord]:
    """Parse unlabeled CAS(ME)^3 videos for strictly self-supervised neuro-vascular pretraining."""
    root = Path(root_dir)
    unlabeled_dir = root / "unlabeled"
    if not unlabeled_dir.exists():
        raise FileNotFoundError(f"Expected unlabeled directory under {root}")

    records: list[VideoRecord] = []
    for p in sorted(unlabeled_dir.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        sub = p.parent.name
        records.append(
            VideoRecord(
                video_id=f"casme3_unlabeled_{sub}_{p.stem}",
                dataset="casme3_unlabeled",
                subject_id=sub,
                color_mode="rgb",
                fps=fps,
                num_frames=_count_frames(p),
                video_path=str(p),
                annotations=(),
            )
        )
    return records


def build_composite_megc_3cls(manifests: list[list[VideoRecord]]) -> list[VideoRecord]:
    """Merge multiple dataset manifests into a composite MEGC 3-class benchmark (neg/pos/surprise)."""
    merged: list[VideoRecord] = []
    for ds_records in manifests:
        for rec in ds_records:
            kept_anns = tuple(
                ExpressionAnnotation(
                    onset=a.onset,
                    apex=a.apex,
                    offset=a.offset,
                    expr_type=a.expr_type,
                    emotion_raw=a.emotion_3cls,
                    emotion_3cls=a.emotion_3cls,
                    au_list=a.au_list,
                )
                for a in rec.annotations
                if a.emotion_3cls is not None
            )
            if not kept_anns:
                continue
            merged.append(
                VideoRecord(
                    video_id=rec.video_id,
                    dataset="megc_composite",
                    subject_id=f"{rec.dataset}_{rec.subject_id}",
                    color_mode=rec.color_mode,
                    fps=rec.fps,
                    num_frames=rec.num_frames,
                    video_path=rec.video_path,
                    annotations=kept_anns,
                    biopac_path=rec.biopac_path,
                )
            )
    return merged
