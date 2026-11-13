"""In-process job store and background worker wiring for the web UI."""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..core import ffmpeg_io, fields as F
from ..core.diagnostics import diagnose
from ..core.pipeline import DeinterlaceConfig, export, run


@dataclass
class Job:
    job_id: str
    source_path: str
    work_dir: str
    status: str = "created"            # created|analyzed|processing|done|error
    error: str | None = None
    traceback_text: str | None = None
    info: dict | None = None
    diagnosis: dict | None = None
    frames: list = field(default_factory=list)          # raw interlaced frames
    field_samples: list = field(default_factory=list)   # FieldSample per order
    _field_cache: dict = field(default_factory=dict)    # order -> split samples
    active_order: str | None = None
    result: object = None
    manifest: dict | None = None
    manifest_path: str | None = None
    output_path: str | None = None

    def samples_for_order(self, order: str):
        """Split the decoded rows once per order; UI re-ordering reuses them.

        The cache wraps :func:`fields.split_fields_reusable`, so switching the
        operator's TFF/BFF selection never re-copies the field rows, while the
        returned samples remain real, freshly computed FieldSample objects.
        """
        if order not in (F.TFF, F.BFF):
            raise ValueError(f"field order must be 'tff' or 'bff', got {order!r}")
        if order not in self._field_cache:
            self._field_cache[order] = F.split_fields_reusable(
                iter(self.frames), order
            )
        return self._field_cache[order]


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, source_path: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        work_dir = self.root / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, source_path=str(source_path), work_dir=str(work_dir))
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


def analyze_job(job: Job) -> Job:
    """Decode + diagnose. Errors are captured on the job for UI feedback."""
    try:
        info = ffmpeg_io.probe(job.source_path)
        frames = list(ffmpeg_io.decode_frames(job.source_path, info))
        if not frames:
            raise ValueError("decoder produced zero frames")
        diagnosis = diagnose(frames, info.field_order)
        job.info = {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "duration": info.duration,
            "metadata_field_order": info.field_order,
        }
        job.frames = frames
        job.diagnosis = diagnosis.to_dict()
        job.status = "analyzed"
    except Exception as exc:  # surfaced back to the archive UI
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.traceback_text = traceback.format_exc()
    return job


def process_job(job: Job, order: str, mode: str, thresholds: dict) -> Job:
    """Run the chosen reconstruction and export a real video + manifest."""
    try:
        if order not in (F.TFF, F.BFF):
            raise ValueError("choose an explicit field order before processing")
        diag = job.diagnosis or {}
        if diag.get("blocking") == "progressive_input":
            raise ValueError(
                "input is progressive; deinterlacing it would invent artifacts"
            )
        cfg = DeinterlaceConfig(
            field_order=order,
            mode=mode,
            diff_threshold=float(thresholds.get("diff_threshold", 10.0)),
            neighborhood_threshold=float(
                thresholds.get("neighborhood_threshold", 6.0)
            ),
        )
        job.status = "processing"
        result = run(job.frames, job.info["fps"], cfg)
        manifest = export(result, job.work_dir, base_name="output")
        job.result = result
        job.active_order = order
        job.manifest = manifest
        job.output_path = str(Path(job.work_dir) / manifest["video_file"])
        job.manifest_path = str(
            Path(job.work_dir) / f"output_{cfg.mode}_manifest.json"
        )
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.traceback_text = traceback.format_exc()
    return job
