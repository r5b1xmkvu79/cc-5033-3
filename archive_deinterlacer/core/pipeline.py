"""End-to-end deinterlacing pipeline driven by the field-order-aware kernel.

Modes
-----
``bob``        one output frame per field at 2x frame rate; purely spatial,
               so motion stays smooth and no foreign-time pixels merge.
``weave``      one output frame per source frame at the source rate; pairs
               the two fields captured together and is exact for still content.
``adaptive``   one output frame per field at 2x frame rate; static regions
               weave in neighboring fields for full detail, moving regions
               fall back to spatial reconstruction. Exposes the motion mask so
               the UI can show *why* each pixel was handled the way it was.

Every mode returns its own output timestamps; the encoder is always invoked
with the matching frame rate, so duration can never drift from frame count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ffmpeg_io, fields as F

MODE_BOB = "bob"
MODE_WEAVE = "weave"
MODE_ADAPTIVE = "adaptive"
ALL_MODES = (MODE_BOB, MODE_WEAVE, MODE_ADAPTIVE)


@dataclass
class DeinterlaceConfig:
    field_order: str                      # F.TFF or F.BFF, explicitly chosen
    mode: str = MODE_ADAPTIVE
    diff_threshold: float = 10.0
    neighborhood_threshold: float = 6.0
    crf: int = 16

    def validate(self) -> None:
        if self.field_order not in (F.TFF, F.BFF):
            raise ValueError(
                "field order must be chosen explicitly ('tff' or 'bff'); "
                f"got {self.field_order!r}"
            )
        if self.mode not in ALL_MODES:
            raise ValueError(f"unknown mode {self.mode!r}; choose one of {ALL_MODES}")


@dataclass
class FrameRecord:
    """One reconstructed frame plus the timing/context the UI needs."""

    index: int
    timestamp: float
    rgb: np.ndarray
    motion_mask: np.ndarray | None = None     # full-resolution boolean
    source_field_indices: list[int] = field(default_factory=list)


@dataclass
class PipelineResult:
    output_fps: float
    expected_duration: float
    records: list[FrameRecord]
    field_samples: list[F.FieldSample]
    field_timestamps: np.ndarray
    mode: str
    field_order: str
    motion_pixel_ratio: float

    @property
    def frame_count(self) -> int:
        return len(self.records)


def _build_adaptive(samples, timestamps, cfg: DeinterlaceConfig) -> tuple[list[FrameRecord], float]:
    by_pos: dict[str, list[F.FieldSample]] = {F.TOP: [], F.BOTTOM: []}
    for s in samples:
        by_pos[s.position].append(s)
    same_index = {F.TOP: {}, F.BOTTOM: {}}
    for pos in (F.TOP, F.BOTTOM):
        for j, s in enumerate(by_pos[pos]):
            same_index[pos][s.index] = (
                by_pos[pos][j - 1] if j > 0 else None,
                by_pos[pos][j + 1] if j + 1 < len(by_pos[pos]) else None,
            )

    records: list[FrameRecord] = []
    motion_pixels = 0
    total_pixels = 0
    for i, current in enumerate(samples):
        prev_cross = samples[i - 1] if i > 0 else None
        next_cross = samples[i + 1] if i + 1 < len(samples) else None
        same_prev, same_next = same_index[current.position][current.index]
        rgb, mask = F.adaptive_reconstruct(
            current,
            same_prev,
            same_next,
            prev_cross,
            next_cross,
            cfg.diff_threshold,
            cfg.neighborhood_threshold,
        )
        motion_pixels += int(mask.sum())
        total_pixels += mask.size
        records.append(
            FrameRecord(
                index=i,
                timestamp=float(timestamps[i]),
                rgb=rgb,
                motion_mask=mask,
                source_field_indices=[current.index],
            )
        )
    ratio = motion_pixels / total_pixels if total_pixels else 0.0
    return records, ratio


def _build_bob(samples, timestamps) -> list[FrameRecord]:
    return [
        FrameRecord(
            index=i,
            timestamp=float(timestamps[i]),
            rgb=F.bob_reconstruct(s),
            motion_mask=None,
            source_field_indices=[s.index],
        )
        for i, s in enumerate(samples)
    ]


def _build_weave(samples, timestamps) -> list[FrameRecord]:
    """Pair consecutive capture-order fields (always opposite positions)."""
    records: list[FrameRecord] = []
    pair_count = len(samples) // 2
    for k in range(pair_count):
        a, b = samples[2 * k], samples[2 * k + 1]
        if a.position == F.TOP:
            top_field, bottom_field = a.data, b.data
        else:
            top_field, bottom_field = b.data, a.data
        rgb = F.weave(top_field, bottom_field)
        # Frame timestamp is the earlier field's exposure time.
        records.append(
            FrameRecord(
                index=k,
                timestamp=float(timestamps[2 * k]),
                rgb=rgb,
                motion_mask=None,
                source_field_indices=[a.index, b.index],
            )
        )
    return records


def run(
    frames: list[np.ndarray],
    source_fps: float,
    cfg: DeinterlaceConfig,
) -> PipelineResult:
    """Run the configured reconstruction on decoded interlaced frames."""
    cfg.validate()
    if source_fps <= 0:
        raise ValueError("source fps must be positive")
    if not frames:
        raise ValueError("no input frames")
    if frames[0].shape[0] % 2:
        raise ValueError("interlaced frames must have an even height")

    samples = list(F.split_stream(iter(frames), cfg.field_order))
    f_ts = F.field_timestamps(len(frames), source_fps)

    if cfg.mode == MODE_WEAVE:
        output_fps = source_fps
        records = _build_weave(samples, f_ts)
        motion_ratio = 0.0
    elif cfg.mode == MODE_BOB:
        output_fps = 2.0 * source_fps
        records = _build_bob(samples, f_ts)
        motion_ratio = 0.0
    else:
        output_fps = 2.0 * source_fps
        records, motion_ratio = _build_adaptive(samples, f_ts, cfg)

    expected_duration = records[-1].timestamp + 1.0 / output_fps if records else 0.0
    return PipelineResult(
        output_fps=output_fps,
        expected_duration=float(expected_duration),
        records=records,
        field_samples=samples,
        field_timestamps=f_ts,
        mode=cfg.mode,
        field_order=cfg.field_order,
        motion_pixel_ratio=float(motion_ratio),
    )


def export(
    result: PipelineResult,
    out_dir: str | Path,
    base_name: str,
    ffmpeg: str | None = None,
    crf: int = 16,
) -> dict:
    """Write the real reconstructed MP4 plus timing manifest and field data."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{base_name}_{result.mode}_{result.field_order}.mp4"

    def stream():
        for rec in result.records:
            yield rec.rgb

    written = ffmpeg_io.encode_frames(
        stream(),
        video_path,
        width=result.records[0].rgb.shape[1],
        height=result.records[0].rgb.shape[0],
        fps=result.output_fps,
        ffmpeg=ffmpeg,
        crf=crf,
    )
    if written != result.frame_count:
        raise RuntimeError(
            f"encoder wrote {written} frames but pipeline produced {result.frame_count}"
        )

    manifest = {
        "video_file": video_path.name,
        "mode": result.mode,
        "field_order": result.field_order,
        "source_field_count": len(result.field_samples),
        "output_frame_count": result.frame_count,
        "output_fps": result.output_fps,
        "frame_period_seconds": 1.0 / result.output_fps,
        "expected_duration_seconds": result.expected_duration,
        "duration_check": (
            result.frame_count / result.output_fps == result.expected_duration
            or abs(
                result.frame_count / result.output_fps - result.expected_duration
            )
            < 1e-9
        ),
        "motion_pixel_ratio": result.motion_pixel_ratio,
        "frames": [
            {
                "frame": rec.index,
                "timestamp_seconds": round(rec.timestamp, 6),
                "source_field_indices": rec.source_field_indices,
                "motion": (
                    round(float(rec.motion_mask.mean()), 4)
                    if rec.motion_mask is not None
                    else None
                ),
            }
            for rec in result.records
        ],
        "fields": [
            {
                "field": s.index,
                "frame": s.frame_index,
                "position": s.position,
                "timestamp_seconds": round(float(result.field_timestamps[s.index]), 6),
            }
            for s in result.field_samples
        ],
    }
    ffmpeg_io.write_manifest(manifest, out_dir / f"{base_name}_{result.mode}_manifest.json")
    return manifest
