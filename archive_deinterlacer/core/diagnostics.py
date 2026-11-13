"""Input diagnostics.

Archive metadata is frequently missing or wrong. This module never silently
"fixes" an order by guessing: it measures the signal and returns a verdict
with reasons, leaving the final field-order choice to the operator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import fields as F


@dataclass
class Diagnosis:
    """Structured, UI- and JSON-friendly diagnostic report."""

    metadata_field_order: str          # what the container claims
    content_verdict: str               # "interlaced" | "progressive" | "inconclusive"
    suggested_order: str               # "tff" | "bff" | "none"
    confidence: str                    # "high" | "low" | "none"
    comb_energy: float                 # mean comb score on the raw frames
    comb_frame_ratio: float            # fraction of frames above the comb threshold
    progressive_score: float           # 0..1, 1 means the input behaves progressive
    same_position_residual: float      # temporal residual of same-position fields
    order_score_tff: float             # temporal smoothness when read as TFF
    order_score_bff: float
    even_height: bool
    reasons: list[str] = field(default_factory=list)
    blocking: str = "none"             # "none" | "metadata_missing" | "progressive_input"

    def to_dict(self) -> dict:
        return asdict(self)


def comb_energy(frame: np.ndarray) -> float:
    """Classic interlace comb detector on one full-height frame.

    A row is combed when its center pixel differs strongly from BOTH vertical
    neighbors while those neighbors agree with each other. Returns the mean
    combed-pixel intensity (0..255 scale); a clean progressive frame is ~0.
    """
    x = frame.astype(np.float32).mean(axis=2)
    above, center, below = x[:-2], x[1:-1], x[2:]
    comb = np.abs(center - above) + np.abs(center - below) - np.abs(above - below)
    comb = np.maximum(comb, 0.0)
    return float(comb.mean() / 2.0)


def _luma(field_data: np.ndarray) -> np.ndarray:
    return field_data.astype(np.float32).mean(axis=2)


def order_temporal_score(samples: list[F.FieldSample]) -> float:
    """Mean absolute difference between consecutive fields in capture order.

    The *correct* capture order follows real motion, so adjacent samples are
    closer to each other than under a swapped order (which jumps by nearly a
    full frame period on every step). Lower is smoother.
    """
    diffs = []
    for a, b in zip(samples, samples[1:]):
        diffs.append(np.abs(_luma(a.data) - _luma(b.data)).mean())
    return float(np.mean(diffs)) if diffs else 0.0


def _samples_for_both_orders(
    raw_frames: list[np.ndarray],
) -> tuple[list[F.FieldSample], list[F.FieldSample]]:
    """Split every frame once and reuse the copied rows for both order reads.

    TFF and BFF contain the same two fields per frame in opposite capture
    order. Splitting twice re-copied every row; here the top/bottom row arrays
    are produced once and the BFF samples wrap the same arrays with swapped
    capture indices, so field order and timing stay exactly as before.
    """
    tff_samples = F.split_fields_reusable(iter(raw_frames), F.TFF)
    bff_samples: list[F.FieldSample] = []
    for frame_index in range(len(raw_frames)):
        top = tff_samples[2 * frame_index]
        bottom = tff_samples[2 * frame_index + 1]
        bff_samples.append(
            F.FieldSample(frame_index * 2, frame_index, F.BOTTOM, bottom.data)
        )
        bff_samples.append(
            F.FieldSample(frame_index * 2 + 1, frame_index, F.TOP, top.data)
        )
    return tff_samples, bff_samples


def _samples_for_order(
    raw_frames: list[np.ndarray], order: str
) -> list[F.FieldSample]:
    return list(F.split_stream(iter(raw_frames), order))


def detect_progressive(
    raw_frames: list[np.ndarray],
    samples: list[F.FieldSample],
    comb_values: list[float] | None = None,
) -> tuple[float, float]:
    """Return (progressive_score, same_position_temporal_residual).

    The decisive signal is the same-position field residual: for genuinely
    progressive content both fields depict the same instant, so a field is
    well predicted by the average of the same-position fields adjacent in
    time. Real interlaced motion makes that residual large. Comb energy alone
    is unreliable here because 4:2:0 chroma subsampling creates fake teeth on
    sharp vertical edges of an actually progressive frame.

    ``comb_values`` reuses the per-frame comb energies already computed by
    ``diagnose``; when absent they are computed locally so standalone callers
    keep working.
    """
    if len(raw_frames) < 3:
        return 0.0, 0.0

    if comb_values is None:
        comb_values = [comb_energy(fr) for fr in raw_frames]
    comb = float(np.mean(comb_values))
    comb_term = float(np.clip(1.0 - comb / 4.0, 0.0, 1.0))

    by_position: dict[str, list[F.FieldSample]] = {F.TOP: [], F.BOTTOM: []}
    for s in samples:
        by_position[s.position].append(s)

    consistency = []
    for position in (F.TOP, F.BOTTOM):
        seq = by_position[position]
        for i in range(1, len(seq) - 1):
            mid = _luma(seq[i].data)
            predicted = (_luma(seq[i - 1].data) + _luma(seq[i + 1].data)) / 2.0
            consistency.append(np.abs(mid - predicted).mean())
    residual = float(np.mean(consistency)) if consistency else 0.0
    # Residual ~4 for encoded progressive material, ~8+ under real interlace.
    temporal_term = float(np.clip(1.0 - residual / 8.0, 0.0, 1.0))
    score = float(np.clip(0.65 * temporal_term + 0.35 * comb_term, 0.0, 1.0))
    return score, residual


def diagnose(
    raw_frames: list[np.ndarray],
    metadata_field_order: str,
    comb_threshold: float = 1.5,
) -> Diagnosis:
    """Run the full diagnostic battery against decoded raw frames."""
    if not raw_frames:
        raise ValueError("cannot diagnose an empty frame stream")
    height = raw_frames[0].shape[0]
    even_height = height % 2 == 0

    comb_values = [comb_energy(fr) for fr in raw_frames]
    mean_comb = float(np.mean(comb_values))
    comb_ratio = float(np.mean([c > comb_threshold for c in comb_values]))

    # One row-copy pass; BFF reuses TFF's field row arrays.
    tff_samples, bff_samples = _samples_for_both_orders(raw_frames)
    score_tff = order_temporal_score(tff_samples)
    score_bff = order_temporal_score(bff_samples)
    progressive_score, same_position_residual = detect_progressive(
        raw_frames, tff_samples, comb_values
    )

    reasons: list[str] = []
    if not even_height:
        reasons.append(
            f"frame height {height} is odd: interlaced fields require an even "
            "height; refusing field-based processing"
        )

    if progressive_score >= 0.45 and same_position_residual <= 5.5:
        verdict = "progressive"
        suggested, confidence, blocking = "none", "high", "progressive_input"
        reasons.append(
            f"same-position field residual {same_position_residual:.2f} is low: "
            "both fields depict the same instant, so the content is already "
            "progressive and must not be deinterlaced"
        )
    elif same_position_residual >= 6.0 or mean_comb >= comb_threshold:
        verdict = "interlaced"
        # A meaningful gap is required; otherwise do not pretend to know.
        gap = abs(score_tff - score_bff) / max(score_tff, score_bff, 1e-6)
        if gap >= 0.08:
            suggested = F.TFF if score_tff < score_bff else F.BFF
            confidence = "high" if gap >= 0.2 else "low"
            reasons.append(
                f"capture-order smoothness TFF={score_tff:.2f} vs "
                f"BFF={score_bff:.2f}; lower is smoother, suggesting {suggested}"
            )
        else:
            suggested, confidence = "none", "low"
            reasons.append(
                f"TFF and BFF smoothness are nearly tied "
                f"({score_tff:.2f} vs {score_bff:.2f}); content analysis cannot "
                "settle the order, operator input required"
            )
        blocking = "none"
    else:
        verdict = "inconclusive"
        suggested, confidence, blocking = "none", "none", "none"
        reasons.append(
            f"signal is ambiguous (residual {same_position_residual:.2f}, "
            f"comb {mean_comb:.2f}); the clip may be still or previously processed"
        )

    if metadata_field_order in ("tff", "bff"):
        reasons.append(f"container metadata declares field order: {metadata_field_order}")
        if suggested != "none" and suggested != metadata_field_order:
            reasons.append(
                f"WARNING metadata says {metadata_field_order} but content "
                f"analysis favors {suggested}; metadata may be wrong"
            )
    elif metadata_field_order == "progressive":
        reasons.append("container metadata declares the stream progressive")
    else:
        reasons.append("container carries no usable field-order metadata")
        if verdict == "interlaced" and confidence != "high":
            blocking = "metadata_missing"

    return Diagnosis(
        metadata_field_order=metadata_field_order,
        content_verdict=verdict,
        suggested_order=suggested,
        confidence=confidence,
        comb_energy=round(mean_comb, 4),
        comb_frame_ratio=round(comb_ratio, 4),
        progressive_score=round(progressive_score, 4),
        same_position_residual=round(same_position_residual, 4),
        order_score_tff=round(score_tff, 4),
        order_score_bff=round(score_bff, 4),
        even_height=even_height,
        reasons=reasons,
        blocking=blocking,
    )
