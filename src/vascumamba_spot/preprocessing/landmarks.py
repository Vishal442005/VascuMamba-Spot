"""Facial landmark stabilization, Procrustes alignment, and AU / vascular ROI masks."""

from __future__ import annotations

import cv2
import numpy as np

# Canonical MediaPipe FaceMesh landmark index groups for 12 Kinematic AU ROIs
KINEMATIC_AU_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left_inner_brow": (55, 107, 66, 105),
    "right_inner_brow": (285, 336, 296, 334),
    "left_outer_brow": (46, 53, 52, 65, 70),
    "right_outer_brow": (276, 283, 282, 295, 300),
    "glabella": (9, 8, 168, 107, 336),
    "left_eyelid": (159, 145, 33, 133, 158, 153),
    "right_eyelid": (386, 374, 362, 263, 385, 380),
    "left_nasolabial": (205, 36, 206, 216, 142),
    "right_nasolabial": (425, 266, 426, 436, 371),
    "left_zygomaticus": (116, 123, 187, 207, 50),
    "right_zygomaticus": (345, 352, 411, 427, 280),
    "orbicularis_oris_mentalis": (61, 291, 0, 17, 18, 200, 199),
}

# Canonical MediaPipe FaceMesh landmark index groups for 7 Low-Displacement Vascular ROIs
# Explicitly avoids eyelids and mouth corners to minimize kinematic motion contamination.
VASCULAR_ROI_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left_forehead": (103, 67, 109, 104, 68, 69),
    "right_forehead": (332, 297, 338, 333, 298, 299),
    "glabella_bridge": (10, 151, 9, 108, 337),
    "left_malar": (117, 118, 101, 50, 123, 116),
    "right_malar": (346, 347, 330, 280, 352, 345),
    "philtrum": (164, 167, 393, 97, 326),
    "lower_chin": (175, 171, 396, 152, 148, 377),
}

# Canonical reference anchor points in normalized [0, 1] space (left eye, right eye, nose tip, mouth center)
CANONICAL_ANCHORS_NORM = np.array(
    [
        [0.34, 0.36],  # left eye outer/center
        [0.66, 0.36],  # right eye outer/center
        [0.50, 0.56],  # nose tip
        [0.50, 0.74],  # mouth center
    ],
    dtype=np.float32,
)


def default_canonical_rois(size: int = 256) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build deterministic fallback elliptical/polygonal masks on a canonical size x size face grid.

    Useful for unit testing and when landmark-polygon rasterization is applied on aligned frames.
    """
    kin_boxes = {
        "left_inner_brow": (0.36, 0.27, 0.46, 0.33),
        "right_inner_brow": (0.54, 0.27, 0.64, 0.33),
        "left_outer_brow": (0.24, 0.27, 0.36, 0.33),
        "right_outer_brow": (0.64, 0.27, 0.76, 0.33),
        "glabella": (0.45, 0.27, 0.55, 0.35),
        "left_eyelid": (0.28, 0.33, 0.42, 0.41),
        "right_eyelid": (0.58, 0.33, 0.72, 0.41),
        "left_nasolabial": (0.34, 0.52, 0.44, 0.68),
        "right_nasolabial": (0.56, 0.52, 0.66, 0.68),
        "left_zygomaticus": (0.22, 0.48, 0.34, 0.62),
        "right_zygomaticus": (0.66, 0.48, 0.78, 0.62),
        "orbicularis_oris_mentalis": (0.38, 0.68, 0.62, 0.84),
    }
    vasc_boxes = {
        "left_forehead": (0.27, 0.14, 0.45, 0.24),
        "right_forehead": (0.55, 0.14, 0.73, 0.24),
        "glabella_bridge": (0.45, 0.16, 0.55, 0.26),
        "left_malar": (0.24, 0.43, 0.38, 0.53),
        "right_malar": (0.62, 0.43, 0.76, 0.53),
        "philtrum": (0.44, 0.61, 0.56, 0.67),
        "lower_chin": (0.42, 0.85, 0.58, 0.94),
    }

    def boxes_to_masks(boxes: dict[str, tuple[float, float, float, float]]) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        for name, (x0, y0, x1, y1) in boxes.items():
            m = np.zeros((size, size), dtype=bool)
            ix0, iy0 = max(0, int(round(x0 * size))), max(0, int(round(y0 * size)))
            ix1, iy1 = min(size, int(round(x1 * size))), min(size, int(round(y1 * size)))
            m[iy0:iy1, ix0:ix1] = True
            masks[name] = m
        return masks

    return boxes_to_masks(kin_boxes), boxes_to_masks(vasc_boxes)


def smooth_landmarks_ema(landmarks: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Apply causal + bidirectional EMA smoothing to (T, N, 2) landmarks to suppress detector jitter."""
    if landmarks.shape[0] <= 1:
        return landmarks.copy()
    fwd = np.empty_like(landmarks, dtype=np.float32)
    fwd[0] = landmarks[0]
    for t in range(1, landmarks.shape[0]):
        fwd[t] = alpha * landmarks[t] + (1.0 - alpha) * fwd[t - 1]
    bwd = np.empty_like(landmarks, dtype=np.float32)
    bwd[-1] = fwd[-1]
    for t in range(landmarks.shape[0] - 2, -1, -1):
        bwd[t] = alpha * fwd[t] + (1.0 - alpha) * bwd[t + 1]
    return bwd


def estimate_rigid_stabilization_matrix(
    src_landmarks: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """Estimate a 2x3 similarity transform mapping 478 facial landmarks to canonical coordinates."""
    left_eye = src_landmarks[[33, 133]].mean(axis=0)
    right_eye = src_landmarks[[362, 263]].mean(axis=0)
    nose_tip = src_landmarks[1]
    mouth_center = src_landmarks[[61, 291]].mean(axis=0)
    src_pts = np.stack([left_eye, right_eye, nose_tip, mouth_center], axis=0).astype(np.float32)
    dst_pts = CANONICAL_ANCHORS_NORM * float(target_size)
    mat, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if mat is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return mat.astype(np.float32)


def build_polygon_masks_from_landmarks(
    landmarks_aligned: np.ndarray,
    roi_spec: dict[str, tuple[int, ...]],
    image_size: int = 256,
    dilate_px: int = 4,
) -> dict[str, np.ndarray]:
    """Rasterize convex-hull ROI masks from aligned (478, 2) landmarks."""
    masks: dict[str, np.ndarray] = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    for name, indices in roi_spec.items():
        pts = np.round(landmarks_aligned[list(indices)]).astype(np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        if dilate_px > 0:
            mask = cv2.dilate(mask, kernel)
        masks[name] = mask > 0
    return masks


class FaceLandmarkStabilizer:
    """Detect 478 facial landmarks via MediaPipe FaceMesh and stabilize video frames."""

    def __init__(self, target_size: int = 256, ema_alpha: float = 0.45) -> None:
        self.target_size = target_size
        self.ema_alpha = ema_alpha
        self._face_mesh = None

    def _ensure_mediapipe(self) -> None:
        if self._face_mesh is not None:
            return
        import mediapipe as mp

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
            raise RuntimeError(
                "MediaPipe solutions.face_mesh not available in installed mediapipe build."
            )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect_sequence_landmarks(self, frames_rgb: np.ndarray) -> np.ndarray:
        """Detect (T, 478, 2) pixel landmarks with forward/backward interpolation on missed frames."""
        self._ensure_mediapipe()
        t_len, h, w, _ = frames_rgb.shape
        landmarks = np.full((t_len, 478, 2), np.nan, dtype=np.float32)
        for t in range(t_len):
            res = self._face_mesh.process(frames_rgb[t])
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                coords = np.array([[p.x * w, p.y * h] for p in lm[:478]], dtype=np.float32)
                landmarks[t] = coords

        valid_idx = np.where(~np.isnan(landmarks[:, 0, 0]))[0]
        if len(valid_idx) == 0:
            raise ValueError("No faces detected in input video sequence.")
        if len(valid_idx) < t_len:
            all_idx = np.arange(t_len)
            for n in range(478):
                for c in range(2):
                    landmarks[:, n, c] = np.interp(all_idx, valid_idx, landmarks[valid_idx, n, c])
        return smooth_landmarks_ema(landmarks, alpha=self.ema_alpha)

    def stabilize_frames(
        self,
        frames: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Align all frames onto a canonical target_size x target_size grid and compute ROI masks.

        Supports both RGB (T, H, W, 3) and grayscale (T, H, W) inputs.
        """
        is_gray = frames.ndim == 3
        if landmarks is None:
            rgb_for_mp = np.repeat(frames[..., None], 3, axis=-1) if is_gray else frames
            landmarks = self.detect_sequence_landmarks(rgb_for_mp)

        t_len = frames.shape[0]
        out_shape = (t_len, self.target_size, self.target_size) + (() if is_gray else (3,))
        aligned = np.empty(out_shape, dtype=frames.dtype)
        aligned_lms = np.empty_like(landmarks)

        for t in range(t_len):
            mat = estimate_rigid_stabilization_matrix(landmarks[t], target_size=self.target_size)
            aligned[t] = cv2.warpAffine(
                frames[t],
                mat,
                (self.target_size, self.target_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            ones = np.ones((landmarks.shape[1], 1), dtype=np.float32)
            aligned_lms[t] = (landmarks[t] @ mat[:, :2].T) + mat[:, 2]
            _ = ones

        median_lm = np.median(aligned_lms, axis=0)
        kin_masks = build_polygon_masks_from_landmarks(
            median_lm, KINEMATIC_AU_LANDMARKS, image_size=self.target_size, dilate_px=5
        )
        vasc_masks = build_polygon_masks_from_landmarks(
            median_lm, VASCULAR_ROI_LANDMARKS, image_size=self.target_size, dilate_px=4
        )
        return aligned, kin_masks, vasc_masks
