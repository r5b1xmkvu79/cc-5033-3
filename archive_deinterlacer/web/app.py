"""HTTP routes exposing the archive deinterlacing engine.

All heavy lifting happens synchronously per request in a worker thread; the
UI polls status. Every route returns structured error JSON so kernel-level
refusals (missing field order, progressive input, odd height) appear in the
interface instead of being silently guessed around.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)
from PIL import Image

from ..core import fields as F
from ..core.pipeline import ALL_MODES
from .jobs import JobStore, analyze_job, process_job

bp = Blueprint("deinterlace", __name__)

STORE: JobStore | None = None
UPLOAD_ROOT: Path | None = None


def init_store(work_root: Path) -> None:
    global STORE, UPLOAD_ROOT
    work_root = Path(work_root)
    UPLOAD_ROOT = work_root / "uploads"
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    STORE = JobStore(work_root / "jobs")


def _png_response(frame: np.ndarray) -> Response:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")


def _require_job(job_id: str):
    if STORE is None:
        raise RuntimeError("store not initialized")
    return STORE.get(job_id)


@bp.route("/")
def index():
    return render_template("index.html", modes=ALL_MODES)


@bp.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "no video file provided"}), 400
    safe_name = Path(f.filename).name.replace("/", "_")
    job = STORE.create(UPLOAD_ROOT / safe_name)
    f.save(job.source_path)
    if request.args.get("analyze") == "sync":
        analyze_job(job)
    else:
        thread = threading.Thread(target=analyze_job, args=(job,), daemon=True)
        thread.start()
    return jsonify({"ok": True, "job_id": job.job_id, "status": job.status})


@bp.route("/api/jobs/<job_id>")
def job_state(job_id):
    try:
        job = _require_job(job_id)
    except KeyError:
        return jsonify({"ok": False, "error": "unknown job id"}), 404
    payload = {
        "ok": job.status != "error",
        "status": job.status,
        "error": job.error,
        "info": job.info,
        "diagnosis": job.diagnosis,
        "raw_frame_count": len(job.frames),
        "active_order": job.active_order,
        "manifest": job.manifest,
    }
    if job.result is not None:
        payload["output_frame_count"] = job.result.frame_count
        payload["output_fps"] = job.result.output_fps
        payload["expected_duration"] = job.result.expected_duration
    return jsonify(payload)


@bp.route("/api/jobs/<job_id>/process", methods=["POST"])
def process(job_id):
    try:
        job = _require_job(job_id)
    except KeyError:
        return jsonify({"ok": False, "error": "unknown job id"}), 404
    body = request.get_json(force=True)
    order = body.get("field_order")
    mode = body.get("mode", "adaptive")
    if order not in (F.TFF, F.BFF):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        "field order must be explicitly selected as 'tff' or "
                        "'bff'; the tool does not guess it"
                    ),
                }
            ),
            400,
        )
    if mode not in ALL_MODES:
        return jsonify({"ok": False, "error": f"mode must be one of {ALL_MODES}"}), 400
    if job.status not in ("analyzed", "done", "error"):
        return jsonify({"ok": False, "error": f"job is {job.status}, not ready"}), 409
    job.error = None
    job.status = "analyzed"
    if request.args.get("run") == "sync":
        process_job(job, order, mode, body.get("thresholds", {}))
    else:
        thread = threading.Thread(
            target=process_job,
            args=(job, order, mode, body.get("thresholds", {})),
            daemon=True,
        )
        thread.start()
    return jsonify({"ok": True, "status": job.status})


@bp.route("/api/jobs/<job_id>/raw/<int:frame_index>")
def raw_frame(job_id, frame_index):
    job = _require_job(job_id)
    if not job.frames:
        return jsonify({"ok": False, "error": "no decoded frames yet"}), 409
    if not 0 <= frame_index < len(job.frames):
        return jsonify({"ok": False, "error": "frame index out of range"}), 404
    return _png_response(job.frames[frame_index])


@bp.route("/api/jobs/<job_id>/field/<int:field_index>")
def field_frame(job_id, field_index):
    """A single field viewed in isolation, line-doubled for display."""
    job = _require_job(job_id)
    order = request.args.get("order", job.active_order or F.TFF)
    if order not in (F.TFF, F.BFF):
        return jsonify({"ok": False, "error": "bad order parameter"}), 400
    try:
        samples = job.samples_for_order(order)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if not 0 <= field_index < len(samples):
        return jsonify({"ok": False, "error": "field index out of range"}), 404
    display = F.bob_reconstruct(samples[field_index])
    return _png_response(display)


@bp.route("/api/jobs/<job_id>/result/<int:frame_index>")
def result_frame(job_id, frame_index):
    """Final reconstructed frame; ?overlay=motion draws the motion-edge mask."""
    job = _require_job(job_id)
    if job.result is None:
        return jsonify({"ok": False, "error": "run processing first"}), 409
    records = job.result.records
    if not 0 <= frame_index < len(records):
        return jsonify({"ok": False, "error": "result frame index out of range"}), 404
    rec = records[frame_index]
    rgb = rec.rgb.copy()
    if request.args.get("overlay") == "motion" and rec.motion_mask is not None:
        rgb[rec.motion_mask] = (0.6 * rgb[rec.motion_mask] + np.array([255, 60, 0]) * 0.4).astype(np.uint8)
    return _png_response(rgb)


@bp.route("/api/jobs/<job_id>/download/video")
def download_video(job_id):
    job = _require_job(job_id)
    if not job.output_path:
        return jsonify({"ok": False, "error": "no exported video yet"}), 409
    return send_file(
        Path(job.output_path),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=Path(job.output_path).name,
    )


@bp.route("/api/jobs/<job_id>/download/manifest")
def download_manifest(job_id):
    job = _require_job(job_id)
    if not job.manifest_path:
        return jsonify({"ok": False, "error": "no manifest yet"}), 409
    manifest_path = Path(job.manifest_path)
    return send_file(
        manifest_path,
        mimetype="application/json",
        as_attachment=True,
        download_name=manifest_path.name,
    )


def create_app(work_root: Path) -> "FlaskApp":
    from flask import Flask, jsonify

    work_root = Path(work_root).resolve()
    app = Flask(__name__)
    app.register_blueprint(bp)

    @app.errorhandler(KeyError)
    def _unknown_job(exc):
        # Every unknown job id must reach the UI as a structured error, never
        # as an HTML 500 page the JS cannot render in its status area.
        return jsonify({"ok": False, "error": "unknown job id"}), 404

    init_store(work_root)
    return app
