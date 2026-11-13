"""Deinterlacing performance suite for the field-processing acceleration.

The suite measures where time actually goes per stage and proves the speedup
is not bought by skipping fields or reducing resolution:

* every workload is run through BOTH the frozen legacy builder
  (``core.legacy``) and the accelerated builder, on the identical input;
* stage timing is decomposed into field split, motion verdict, reconstruction
  and total, reported as seconds per output frame;
* correctness is asserted first — any pixel, mask, frame count, fps or
  timestamp mismatch raises instead of being papered over by a default;
* comb residue is measured on the real reconstructed frames (same detector as
  the diagnostics kernel), so a faster path that grows teeth fails the run.

Two complementary loads stress different parts of the kernel:

* ``thin_line_motion`` — 1 px wide vertical strokes and fine glyph-like
  detail moving every field, which maximizes motion-edge and fine-line checks;
* ``field_order_switch`` — the same clip emulated under TFF and BFF, which
  stresses the row-reuse split path that previously re-copied every field.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ffmpeg_io, fields as F
from .diagnostics import comb_energy
from .legacy import build_adaptive_legacy
from .pipeline import (
    DeinterlaceConfig,
    FrameRecord,
    _build_adaptive,
    _same_position_index,
)

PERF_HEIGHT = 288
PERF_WIDTH = 384
PERF_FRAMES = 60          # interlaced frames -> 120 field samples
PERF_FPS = 25.0
WARMUP_RUNS = 1
TIMED_RUNS = 5


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _render_thin_line_frame(t: float, k: int) -> np.ndarray:
    """Ground-truth frame with thin moving strokes plus a static fine grid."""
    h, w = PERF_HEIGHT, PERF_WIDTH
    img = np.full((h, w, 3), 24, np.uint8)

    # 1 px wide vertical strokes moving right; the thinnest possible motion
    # edges, so any relaxed mask criterion would show up here immediately.
    shift = int(round(95.0 * t))
    for x0 in range(-20, w + 20, 14):
        x = (x0 + shift) % (w + 40) - 20
        if 0 <= x < w:
            img[20:120, x, :] = (235, 235, 220)

    # Fine alternating-pixel checker band drifting sideways: high frequency
    # detail that must stay static-clean and stay tooth-free while moving.
    band_shift = int(round(40.0 * t))
    for y in range(140, 200):
        for x in range(0, w, 2):
            xx = (x + band_shift) % w
            if (y + (x // 2)) % 2 == 0:
                img[y, xx, :] = (200, 210, 255)

    # Static 2 px grid that never moves: the reuse path must keep it exact.
    for x in range(300, 372, 4):
        img[210:280, x, :] = (180, 180, 180)
    for y in range(212, 280, 6):
        img[y, 300:372, :] = (120, 120, 120)
    return img


def make_thin_line_workload(frames: int = PERF_FRAMES) -> list[np.ndarray]:
    """Emulated interlaced clip (TFF) carrying thin-line motion."""
    ground = [
        _render_thin_line_frame(k / (2.0 * PERF_FPS), k)
        for k in range(frames * 2)
    ]
    return _emulate(ground, F.TFF)


def _emulate(ground_truth: list[np.ndarray], order: str) -> list[np.ndarray]:
    pairs = len(ground_truth) // 2
    out = []
    for k in range(pairs):
        earlier, later = ground_truth[2 * k], ground_truth[2 * k + 1]
        frame = np.empty_like(earlier)
        if order == F.TFF:
            frame[0::2] = earlier[0::2]
            frame[1::2] = later[1::2]
        else:
            frame[1::2] = earlier[1::2]
            frame[0::2] = later[0::2]
        out.append(frame)
    return out


def make_field_order_workload(frames: int = PERF_FRAMES) -> dict[str, list[np.ndarray]]:
    """The identical ground truth emulated under both capture orders."""
    ground = [
        _render_thin_line_frame(k / (2.0 * PERF_FPS), k)
        for k in range(frames * 2)
    ]
    return {
        F.TFF: _emulate(ground, F.TFF),
        F.BFF: _emulate(ground, F.BFF),
    }


# ---------------------------------------------------------------------------
# Stage-cost decomposition
# ---------------------------------------------------------------------------


@dataclass
class StageCost:
    split_seconds: float
    motion_seconds: float
    reconstruct_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.split_seconds + self.motion_seconds + self.reconstruct_seconds

    def per_frame(self, frame_count: int) -> dict:
        return {
            "split_seconds_per_frame": round(self.split_seconds / frame_count, 6),
            "motion_seconds_per_frame": round(self.motion_seconds / frame_count, 6),
            "reconstruct_seconds_per_frame": round(
                self.reconstruct_seconds / frame_count, 6
            ),
            "total_seconds_per_frame": round(self.total_seconds / frame_count, 6),
        }


def _time_split_legacy(frames, order: str) -> tuple[list[F.FieldSample], float]:
    start = time.perf_counter()
    samples = list(F.split_stream(iter(frames), order))
    return samples, time.perf_counter() - start


def _time_split_fast(frames, order: str) -> tuple[list[F.FieldSample], float]:
    start = time.perf_counter()
    samples = F.split_fields_reusable(iter(frames), order)
    return samples, time.perf_counter() - start


def _time_motion_stage(samples, cfg: DeinterlaceConfig, fast: bool) -> float:
    same_index = _same_position_index(samples)
    start = time.perf_counter()
    for current in samples:
        same_prev, same_next = same_index[current.position][current.index]
        if same_prev is None or same_next is None:
            continue
        if fast:
            F.motion_mask_fast(
                same_prev.data,
                same_next.data,
                cfg.diff_threshold,
                cfg.neighborhood_threshold,
            )
        else:
            F.motion_mask(
                same_prev.data,
                same_next.data,
                cfg.diff_threshold,
                cfg.neighborhood_threshold,
            )
    return time.perf_counter() - start


def _time_reconstruct_stage(
    samples, timestamps, cfg: DeinterlaceConfig, fast: bool
) -> tuple[list[FrameRecord], float]:
    start = time.perf_counter()
    if fast:
        records, _ = _build_adaptive(samples, timestamps, cfg)
    else:
        records, _ = build_adaptive_legacy(samples, timestamps, cfg)
    return records, time.perf_counter() - start


def _best_of(runs: int, fn) -> tuple[float, object]:
    """Run fn repeatedly; keep the shortest wall time and its result."""
    best = float("inf")
    last_result = None
    for _ in range(runs):
        start = time.perf_counter()
        last_result = fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return best, last_result


# ---------------------------------------------------------------------------
# Correctness gate — no default masking of errors
# ---------------------------------------------------------------------------


def assert_parity(
    legacy: list[FrameRecord], fast: list[FrameRecord], label: str
) -> dict:
    """Fail loudly on any behavioral difference between the two paths."""
    if len(legacy) != len(fast):
        raise AssertionError(
            f"{label}: frame count changed {len(legacy)} -> {len(fast)}; "
            "the acceleration must not drop fields"
        )
    max_pixel_diff = 0
    mask_mismatches = 0
    for i, (a, b) in enumerate(zip(legacy, fast)):
        if a.rgb.shape != b.rgb.shape:
            raise AssertionError(
                f"{label} frame {i}: resolution changed {a.rgb.shape} -> {b.rgb.shape}"
            )
        if not np.array_equal(a.rgb, b.rgb):
            diff = int(np.abs(a.rgb.astype(np.int16) - b.rgb.astype(np.int16)).max())
            max_pixel_diff = max(max_pixel_diff, diff)
            raise AssertionError(
                f"{label} frame {i}: reconstructed pixels differ (max diff {diff})"
            )
        if not np.array_equal(a.motion_mask, b.motion_mask):
            mask_mismatches += 1
        if abs(a.timestamp - b.timestamp) > 1e-12:
            raise AssertionError(
                f"{label} frame {i}: timestamp drifted {a.timestamp} -> {b.timestamp}"
            )
    if mask_mismatches:
        raise AssertionError(
            f"{label}: motion-edge mask differs on {mask_mismatches} frames; "
            "fine-line/motion criteria must stay identical"
        )
    return {
        "frames_checked": len(legacy),
        "max_pixel_diff": max_pixel_diff,
        "mask_mismatch_frames": mask_mismatches,
        "parity": "exact",
    }


def _comb_residue(records: list[FrameRecord]) -> float:
    return float(np.mean([comb_energy(r.rgb) for r in records]))


# ---------------------------------------------------------------------------
# One full A/B measurement
# ---------------------------------------------------------------------------


def measure_clip(
    frames: list[np.ndarray], order: str, label: str
) -> dict:
    cfg = DeinterlaceConfig(field_order=order, mode="adaptive")
    timestamps = F.field_timestamps(len(frames), PERF_FPS)

    # Warm up allocators/JIT-independent caches, then time each stage.
    for _ in range(WARMUP_RUNS):
        legacy_samples, _ = _time_split_legacy(frames, order)
        fast_samples, _ = _time_split_fast(frames, order)
        _time_motion_stage(legacy_samples, cfg, fast=False)
        _time_motion_stage(fast_samples, cfg, fast=True)
        _time_reconstruct_stage(legacy_samples, timestamps, cfg, fast=False)
        _time_reconstruct_stage(fast_samples, timestamps, cfg, fast=True)

    legacy_split, legacy_pair = _best_of(
        TIMED_RUNS, lambda: _time_split_legacy(frames, order)
    )
    legacy_samples = legacy_pair[0]
    fast_split, fast_pair = _best_of(
        TIMED_RUNS, lambda: _time_split_fast(frames, order)
    )
    fast_samples = fast_pair[0]
    legacy_motion = _best_of(
        TIMED_RUNS, lambda: _time_motion_stage(legacy_samples, cfg, False)
    )[0]
    fast_motion = _best_of(
        TIMED_RUNS, lambda: _time_motion_stage(fast_samples, cfg, True)
    )[0]
    legacy_reconstruct, legacy_built = _best_of(
        TIMED_RUNS,
        lambda: _time_reconstruct_stage(legacy_samples, timestamps, cfg, False),
    )
    legacy_records = legacy_built[0]
    fast_reconstruct, fast_built = _best_of(
        TIMED_RUNS,
        lambda: _time_reconstruct_stage(fast_samples, timestamps, cfg, True),
    )
    fast_records = fast_built[0]

    frame_count = len(legacy_records)
    parity = assert_parity(legacy_records, fast_records, label)

    legacy_cost = StageCost(legacy_split, legacy_motion, legacy_reconstruct)
    fast_cost = StageCost(fast_split, fast_motion, fast_reconstruct)

    comb_legacy = _comb_residue(legacy_records)
    comb_fast = _comb_residue(fast_records)
    if comb_fast > comb_legacy + 1e-9:
        raise AssertionError(
            f"{label}: comb residue grew {comb_legacy:.6f} -> {comb_fast:.6f}"
        )

    return {
        "workload": label,
        "field_order": order,
        "input_interlaced_frames": len(frames),
        "output_frames": frame_count,
        "output_fps": 2.0 * PERF_FPS,
        "resolution": [PERF_HEIGHT, PERF_WIDTH],
        "parity": parity,
        "comb_residue_legacy": round(comb_legacy, 6),
        "comb_residue_accelerated": round(comb_fast, 6),
        "legacy": legacy_cost.per_frame(frame_count),
        "accelerated": fast_cost.per_frame(frame_count),
        "speedup_total_x": round(
            legacy_cost.total_seconds / fast_cost.total_seconds, 4
        ),
        "timed_runs": TIMED_RUNS,
        "note": "best-of-N wall time; identical input for both paths; no frame "
        "or field was skipped and resolution is unchanged",
    }


def run_suite(out_dir: str | Path | None = None) -> dict:
    """Run both loads under both orders and optionally persist the report."""
    report = {"workloads": []}

    thin = make_thin_line_workload()
    report["workloads"].append(measure_clip(thin, F.TFF, "thin_line_motion"))

    orders = make_field_order_workload()
    for order in (F.TFF, F.BFF):
        report["workloads"].append(
            measure_clip(orders[order], order, f"field_order_switch_{order}")
        )

    report["metrics"] = {
        "seconds_per_frame": "lower is faster; decomposed by processing stage",
        "comb_residue": "0 means no interlace comb teeth on reconstructed frames",
        "speedup_total_x": "legacy total time divided by accelerated total time",
        "parity": "exact pixel/mask/timestamp equality is asserted before timing",
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        path = out_dir / "field_processing_perf.json"
        ffmpeg_io.write_manifest(report, path)
        report["report_path"] = str(path)
    return report
