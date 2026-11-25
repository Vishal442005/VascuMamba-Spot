"""Canonical dataset manifests, emotion label mappings, and strict subject-disjoint LOSO folds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ColorMode = Literal["rgb", "mono", "nir"]
ExprType = Literal["ME", "MaE"]

CANONICAL_3CLS_MAP: dict[str, str] = {
    "happiness": "positive",
    "happy": "positive",
    "positive": "positive",
    "surprise": "surprise",
    "disgust": "negative",
    "anger": "negative",
    "fear": "negative",
    "sadness": "negative",
    "repression": "negative",
    "contempt": "negative",
    "negative": "negative",
    "tense": "negative",
    "others": "others",
    "other": "others",
}

EMOTION_TO_ID_3CLS: dict[str, int] = {"negative": 0, "positive": 1, "surprise": 2}
EMOTION_TO_ID_5CLS: dict[str, int] = {
    "happiness": 0,
    "surprise": 1,
    "disgust": 2,
    "repression": 3,
    "others": 4,
}


@dataclass(frozen=True)
class ExpressionAnnotation:
    """Single facial expression interval annotation within a video."""

    onset: int
    apex: int
    offset: int
    expr_type: ExprType
    emotion_raw: str
    emotion_3cls: str | None = None
    au_list: tuple[str, ...] = ()

    def duration_sec(self, fps: float) -> float:
        """Return interval duration in seconds."""
        return max(0.0, (self.offset - self.onset) / fps)


@dataclass(frozen=True)
class VideoRecord:
    """Canonical metadata record for a clipped or untrimmed facial video."""

    video_id: str
    dataset: str
    subject_id: str
    color_mode: ColorMode
    fps: float
    num_frames: int
    video_path: str
    annotations: tuple[ExpressionAnnotation, ...] = field(default_factory=tuple)
    biopac_path: str | None = None

    @property
    def has_rgb(self) -> bool:
        """Return True iff RGB chrominance channels are physically present."""
        return self.color_mode == "rgb"


@dataclass(frozen=True)
class LOSOFold:
    """Subject-disjoint Leave-One-Subject-Out cross-validation fold."""

    fold_index: int
    test_subject: str
    train_records: tuple[VideoRecord, ...]
    test_records: tuple[VideoRecord, ...]

    def assert_disjoint(self) -> None:
        """Verify zero subject overlap between train and test partitions."""
        train_subs = {r.subject_id for r in self.train_records}
        test_subs = {r.subject_id for r in self.test_records}
        overlap = train_subs & test_subs
        if overlap:
            raise AssertionError(f"Subject leakage detected in fold {self.fold_index}: {overlap}")


def map_emotion_3cls(emotion_raw: str) -> str | None:
    """Map dataset-specific emotion string into MEGC 3-class taxonomy (excluding 'others')."""
    mapped = CANONICAL_3CLS_MAP.get(emotion_raw.strip().lower())
    if mapped in EMOTION_TO_ID_3CLS:
        return mapped
    return None


def save_manifest(records: list[VideoRecord], out_path: str | Path) -> Path:
    """Write video manifest records to a JSONL file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")
    return path


def load_manifest(manifest_path: str | Path) -> list[VideoRecord]:
    """Load video manifest records from a JSONL file."""
    path = Path(manifest_path)
    records: list[VideoRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            anns = tuple(
                ExpressionAnnotation(
                    onset=int(a["onset"]),
                    apex=int(a["apex"]),
                    offset=int(a["offset"]),
                    expr_type=a["expr_type"],
                    emotion_raw=a["emotion_raw"],
                    emotion_3cls=a.get("emotion_3cls"),
                    au_list=tuple(a.get("au_list", ())),
                )
                for a in raw.get("annotations", [])
            )
            records.append(
                VideoRecord(
                    video_id=raw["video_id"],
                    dataset=raw["dataset"],
                    subject_id=str(raw["subject_id"]),
                    color_mode=raw["color_mode"],
                    fps=float(raw["fps"]),
                    num_frames=int(raw["num_frames"]),
                    video_path=raw["video_path"],
                    annotations=anns,
                    biopac_path=raw.get("biopac_path"),
                )
            )
    return records


def generate_loso_folds(records: list[VideoRecord]) -> list[LOSOFold]:
    """Generate deterministic subject-disjoint LOSO folds sorted by subject ID."""
    subjects = sorted({r.subject_id for r in records})
    folds: list[LOSOFold] = []
    for idx, test_sub in enumerate(subjects):
        train_recs = tuple(r for r in records if r.subject_id != test_sub)
        test_recs = tuple(r for r in records if r.subject_id == test_sub)
        fold = LOSOFold(
            fold_index=idx,
            test_subject=test_sub,
            train_records=train_recs,
            test_records=test_recs,
        )
        fold.assert_disjoint()
        folds.append(fold)
    return folds
