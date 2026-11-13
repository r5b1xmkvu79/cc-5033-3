"""Quantitative comparison of reconstruction strategies.

Three interpretable metrics, each tied to one of the requirements:

* ``motion_smoothness``  temporal MAD between consecutive output frames in the
  moving bands; erratic or wrongly ordered motion produces larger jumps.
* ``comb_residue``       mean comb energy left in the reconstructed frames;
  zero means no foreign-time rows are visible as teeth.
* ``detail_loss``        high-frequency energy lost in the static test
  pattern relative to the ground truth; zero means full detail retention.

The forbidden "weave then vertical blur" strategy is included as a baseline so
its comb-hiding/detail-destroying trade-off is visible in the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ffmpeg_io, fields as F
from .diagnostics import comb_energy
from .pipeline import ALL_MODES, DeinterlaceConfig, run
from .synthetic import GROUND_FPS, SOURCE_FPS, emulate_interlaced
from .fields import vertical_blur_merge

STATIC_BOX = (150, 232, 250, 310)   # top, bottom, left, right of the still pattern


@dataclass
class StrategyScore:
    name: str
    output_fps: float
    frame_count: int
    duration: float
    motion_smoothness: float
    comb_residue: float
    detail_loss_db: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.name,
            "output_fps": self.output_fps,
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration, 4),
            "motion_smoothness_mad": round(self.motion_smoothness, 4),
            "comb_residue": round(self.comb_residue, 4),
            "detail_loss_db": round(self.detail_loss_db, 3),
        }


def _luma(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32).mean(axis=2)


def motion_smoothness(frames: list[np.ndarray]) -> float:
    """Mean temporal MAD in the moving bands (everything above the static box)."""
    diffs = [
        np.abs(_luma(a[: STATIC_BOX[0]]) - _luma(b[: STATIC_BOX[0]])).mean()
        for a, b in zip(frames, frames[1:])
    ]
    return float(np.mean(diffs)) if diffs else 0.0


def static_hf_energy(frame: np.ndarray) -> float:
    """High-frequency energy inside the static test pattern."""
    top, bottom, left, right = STATIC_BOX
    patch = _luma(frame[top:bottom, left:right])
    laplacian = (
        patch[1:-1, 1:-1] * 4
        - patch[:-2, 1:-1]
        - patch[2:, 1:-1]
        - patch[1:-1, :-2]
        - patch[1:-1, 2:]
    )
    return float(np.mean(laplacian**2))


def detail_loss_db(output_frames: list[np.ndarray], truth_frames: list[np.ndarray]) -> float:
    """Loss of static high-frequency energy expressed in dB (negative = loss)."""
    ref_e = np.mean([static_hf_energy(f) for f in truth_frames])
    out_e = np.mean([static_hf_energy(f) for f in output_frames])
    if ref_e <= 0:
        return 0.0
    return float(10.0 * np.log10(max(out_e, 1e-9) / ref_e))


def _align_to_truth(
    frames: list[np.ndarray], truth: list[np.ndarray], fps: float
) -> list[np.ndarray]:
    """Nearest-time resample so all strategies share the truth time grid."""
    truth_times = np.arange(len(truth)) / GROUND_FPS
    frame_times = np.arange(len(frames)) / fps
    aligned = []
    for t in truth_times:
        idx = int(np.argmin(np.abs(frame_times - t)))
        aligned.append(frames[idx])
    return aligned


def _score(
    name: str,
    frames: list[np.ndarray],
    fps: float,
    truth: list[np.ndarray],
) -> StrategyScore:
    aligned = _align_to_truth(frames, truth, fps)
    return StrategyScore(
        name=name,
        output_fps=fps,
        frame_count=len(frames),
        duration=len(frames) / fps,
        motion_smoothness=motion_smoothness(aligned),
        comb_residue=float(np.mean([comb_energy(f) for f in aligned])),
        detail_loss_db=detail_loss_db(aligned, truth),
    )


def evaluate(
    truth: list[np.ndarray],
    field_order: str,
    diff_threshold: float = 10.0,
    neighborhood_threshold: float = 6.0,
) -> list[StrategyScore]:
    """Score every supported mode plus the blur-merge baseline on one clip."""
    interlaced = emulate_interlaced(truth, field_order)
    scores: list[StrategyScore] = []

    for mode in ALL_MODES:
        cfg = DeinterlaceConfig(
            field_order=field_order,
            mode=mode,
            diff_threshold=diff_threshold,
            neighborhood_threshold=neighborhood_threshold,
        )
        result = run(interlaced, SOURCE_FPS, cfg)
        frames = [rec.rgb for rec in result.records]
        scores.append(_score(mode, frames, result.output_fps, truth))

    # Baseline: merge both fields per frame and hide the teeth with a vertical
    # blur. Same frame count/rate as weave; expected to win nothing honestly.
    blurred = [vertical_blur_merge(fr) for fr in interlaced]
    scores.append(_score("blur_merge_baseline", blurred, SOURCE_FPS, truth))
    return scores


def evaluate_and_report(
    truth: list[np.ndarray],
    out_dir: str | Path,
    ffmpeg: str | None = None,
) -> Path:
    """Run both field orders and persist the comparison report as JSON."""
    out_dir = Path(out_dir)
    report = {"orders": {}}
    for order in (F.TFF, F.BFF):
        scores = evaluate(truth, order)
        report["orders"][order] = [s.to_dict() for s in scores]
    report["metrics"] = {
        "motion_smoothness_mad": "lower is smoother (temporal MAD in moving bands)",
        "comb_residue": "0 means no interlace comb teeth remain",
        "detail_loss_db": "0 = no static detail loss, negative values = loss in dB",
    }
    path = out_dir / "benchmark_report.json"
    ffmpeg_io.write_manifest(report, path)
    return path
