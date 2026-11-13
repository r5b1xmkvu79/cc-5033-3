"""Field math: splitting interlaced frames into time-distinct fields and
reconstructing progressive frames.

Conventions used throughout the project
----------------------------------------
An interlaced frame stores two fields captured at *different* exposure times:

* the **top field** (a.k.a. field 1) holds the even pixel rows 0, 2, 4, ...;
* the **bottom field** (field 2) holds the odd pixel rows 1, 3, 5, ....

"Top/bottom" describes *pixel row position* and is independent of capture
order. Capture order is a separate concept:

* ``tff`` (top-field-first): the top field is the earlier sample;
* ``bff`` (bottom-field-first): the bottom field is the earlier sample.

For an input stream of N interlaced frames at fps F there are 2N field samples
at field rate 2F, numbered in capture order. Timestamps are derived from that
ordering, never guessed from row positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOP = "top"
BOTTOM = "bottom"
TFF = "tff"
BFF = "bff"


@dataclass(frozen=True)
class FieldSample:
    """One field exposure at one sampling instant."""

    index: int            # 0..2N-1 in capture order
    frame_index: int      # source interlaced frame
    position: str         # TOP ("top", even rows) or BOTTOM ("bottom", odd rows)
    data: np.ndarray      # H/2 x W x 3 uint8, vertically compacted rows


def split_frame(frame: np.ndarray, frame_index: int) -> tuple[FieldSample, FieldSample]:
    """Separate one interlaced frame into its top and bottom field samples."""
    top = frame[0::2, :, :].copy()
    bottom = frame[1::2, :, :].copy()
    return (
        FieldSample(frame_index * 2, frame_index, TOP, top),
        FieldSample(frame_index * 2 + 1, frame_index, BOTTOM, bottom),
    )


def split_stream(frames, field_order: str):
    """Yield all 2N field samples in capture order for the declared order.

    The sample ``index`` is its position in capture time, so under BFF the
    bottom field carries the even indices and the top field the odd ones.
    """
    if field_order not in (TFF, BFF):
        raise ValueError(f"field_order must be {TFF!r} or {BFF!r}, got {field_order!r}")
    for frame_index, frame in enumerate(frames):
        top, bottom = split_frame(frame, frame_index)
        if field_order == TFF:
            # top captured first: even time index on the even rows
            yield top
            yield bottom
        else:
            # bottom captured first; same row positions, swapped time indices
            yield FieldSample(frame_index * 2, frame_index, BOTTOM, bottom.data)
            yield FieldSample(frame_index * 2 + 1, frame_index, TOP, top.data)


def field_timestamps(frame_count: int, fps: float) -> np.ndarray:
    """Capture time of each of the 2*frame_count field samples, in seconds."""
    return np.arange(frame_count * 2, dtype=np.float64) / (2.0 * fps)


def weave(top_field: np.ndarray, bottom_field: np.ndarray) -> np.ndarray:
    """Re-interleave two compacted fields into one full-height frame."""
    half_h, width, channels = top_field.shape
    out = np.empty((half_h * 2, width, channels), dtype=np.uint8)
    out[0::2, :, :] = top_field
    out[1::2, :, :] = bottom_field
    return out


def _spatial_interpolate(field_data: np.ndarray, position: str) -> np.ndarray:
    """Line-double a single field: its own rows stay sharp, missing rows are
    averages of the two available neighbors (clamped at image borders)."""
    half_h, width, channels = field_data.shape
    full = np.empty((half_h * 2, width, channels), dtype=np.uint8)
    f = field_data.astype(np.float32)
    if position == TOP:
        full[0::2, :, :] = field_data
        inner = np.clip(((f[:-1] + f[1:]) / 2.0 + 0.5), 0, 255).astype(np.uint8)
        full[1:-1:2, :, :] = inner
        full[-1, :, :] = field_data[-1]
    else:
        full[1::2, :, :] = field_data
        inner = np.clip(((f[:-1] + f[1:]) / 2.0 + 0.5), 0, 255).astype(np.uint8)
        full[2::2, :, :] = inner
        full[0, :, :] = field_data[0]
    return full


def _dilate3(mask: np.ndarray) -> np.ndarray:
    """3x3 grayscale dilation via shifted maximums (half-resolution grid)."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out = np.maximum(out, np.roll(mask, (dy, dx), axis=(0, 1)))
    return out


def motion_mask(
    earlier: np.ndarray,
    later: np.ndarray,
    diff_threshold: float = 10.0,
    neighborhood_threshold: float = 6.0,
) -> np.ndarray:
    """Half-resolution boolean mask: True where content moved between two
    same-position fields one frame period apart.

    A pixel is "moving" when its own field difference exceeds
    ``diff_threshold`` AND the maximum difference in its 3x3 neighborhood
    exceeds ``neighborhood_threshold``. The second test rejects isolated
    sensor noise on genuinely static fine detail. Values are 0..255 luma units.
    """
    diff = np.abs(earlier.astype(np.int16) - later.astype(np.int16)).mean(axis=2)
    neighborhood = _dilate3(diff)
    return (diff > diff_threshold) & (neighborhood > neighborhood_threshold)


def _expand_mask_to_full(mask_half: np.ndarray) -> np.ndarray:
    """Place a half-resolution mask on both rows of each field line pair."""
    full = np.zeros((mask_half.shape[0] * 2, mask_half.shape[1]), dtype=bool)
    full[0::2] = mask_half
    full[1::2] = mask_half
    return full


def adaptive_reconstruct(
    current: FieldSample,
    same_prev: FieldSample | None,
    same_next: FieldSample | None,
    cross_prev: FieldSample | None,
    cross_next: FieldSample | None,
    diff_threshold: float = 10.0,
    neighborhood_threshold: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild one full-height progressive frame from one field.

    The motion decision comes from the two *same-position* fields bracketing
    the current sample (one full frame period apart), so static areas show
    near-zero difference while real movement lights up.

    Static pixels are woven with the opposite-position field nearest in time,
    retaining full vertical detail. Moving pixels are spatially interpolated
    inside the current field only, so no foreign time instant can create comb
    teeth.

    Returns (rgb_frame, motion_edge_mask) with the mask at full resolution.
    """
    spatial = _spatial_interpolate(current.data, current.position)
    if same_prev is None or same_next is None or cross_prev is None:
        edge_full = np.zeros((current.data.shape[0] * 2, current.data.shape[1]), bool)
        return spatial, edge_full

    mask_half = motion_mask(
        same_prev.data, same_next.data, diff_threshold, neighborhood_threshold
    )
    edge_full = _expand_mask_to_full(mask_half)

    temporal = np.empty_like(spatial)
    cross = cross_prev if cross_prev is not None else cross_next
    if current.position == TOP:
        temporal[0::2] = current.data
        temporal[1::2] = cross.data
    else:
        temporal[1::2] = current.data
        temporal[0::2] = cross.data

    moving = edge_full[:, :, np.newaxis]
    return np.where(moving, spatial, temporal).astype(np.uint8), edge_full


def bob_reconstruct(field: FieldSample) -> np.ndarray:
    """Pure spatial bob: one progressive frame per field, field-rate output."""
    return _spatial_interpolate(field.data, field.position)


def vertical_blur_merge(frame: np.ndarray) -> np.ndarray:
    """The deliberately bad reference strategy.

    Weaves both fields and then smears adjacent rows, hiding comb teeth with
    vertical detail loss. Kept only so the benchmark can demonstrate why this
    is not an acceptable archive output.
    """
    woven = frame.astype(np.float32)
    blurred = woven.copy()
    blurred[1:-1] = (woven[:-2] + 2 * woven[1:-1] + woven[2:]) / 4.0
    blurred[0] = (woven[0] + woven[1]) / 2.0
    blurred[-1] = (woven[-1] + woven[-2]) / 2.0
    return np.clip(blurred + 0.5, 0, 255).astype(np.uint8)
