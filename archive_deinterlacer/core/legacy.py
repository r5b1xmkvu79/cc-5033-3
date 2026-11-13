"""Frozen pre-acceleration field-processing reference.

This module intentionally keeps the *original* adaptive pipeline behavior:
per-frame double row copies, int16/float motion differences, a full spatial
frame and a full temporal frame merged with ``np.where``. The accelerated
pipeline in :mod:`archive_deinterlacer.core.pipeline` must stay pixel-identical
to the records produced here.

It is used by:

* the deinterlacing performance suite (``core/perf.py``) as the baseline that
  measured speedups are compared against;
* the parity tests, which assert frame-by-frame equality of both paths;
* rollback: if the accelerated path ever regresses behavior beyond comb error
  and per-frame cost, switching the pipeline back to this builder restores the
  archived behavior.

Do not "optimize" this module — its value is being the stable reference.
"""

from __future__ import annotations

from ..core import fields as F
from ..core.pipeline import FrameRecord


def build_adaptive_legacy(samples, timestamps, cfg) -> tuple[list[FrameRecord], float]:
    """Verbatim behavior of the pre-acceleration ``_build_adaptive``."""
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
