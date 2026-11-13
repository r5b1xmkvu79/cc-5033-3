"""ffmpeg process management and container metadata parsing.

The runtime image has no ffprobe, so metadata is parsed from the banner that
ffmpeg prints on stderr. Decoding and encoding both stream raw RGB24 through
subprocess pipes so that a full video never has to reside in memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    """ffmpeg is missing or rejected the command."""


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path.

    Honors an explicit FFMPEG env var, falls back to PATH, then to the binary
    shipped with imageio-ffmpeg (which needs no system package).
    """
    import os

    env_path = os.environ.get("FFMPEG")
    if env_path and Path(env_path).exists():
        return env_path
    path_hit = shutil.which("ffmpeg")
    if path_hit:
        return path_hit
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FFmpegError(
            "ffmpeg not found on PATH and imageio-ffmpeg is unavailable"
        ) from exc


@dataclass
class VideoInfo:
    """Container-level description of an input clip."""

    width: int
    height: int
    fps: float
    frame_count: int  # estimated from duration when the container lacks nb_frames
    duration: float
    field_order: str  # "tff" | "bff" | "progressive" | "unknown"
    raw_banner: str

    @property
    def is_interlaced_metadata(self) -> bool:
        return self.field_order in ("tff", "bff")


def _parse_fps(token: str) -> float:
    token = token.strip()
    if "/" in token:
        num, den = token.split("/")
        den_f = float(den)
        if den_f == 0:
            return 0.0
        return float(num) / den_f
    return float(token)


def probe(path: str | Path, ffmpeg: str | None = None) -> VideoInfo:
    """Parse container metadata from the ffmpeg banner."""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [ffmpeg, "-hide_banner", "-i", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    banner = proc.stderr

    width = height = 0
    fps = 0.0
    duration = 0.0
    field_order = "unknown"

    for line in banner.splitlines():
        stripped = line.strip()
        if stripped.startswith("Duration:"):
            # Duration: 00:00:02.00, start: ..., bitrate: ...
            time_part = stripped.split(",")[0].replace("Duration:", "").strip()
            if time_part != "N/A":
                h, m, s = time_part.split(":")
                duration = int(h) * 3600 + int(m) * 60 + float(s)
        if "Video:" in line:
            # Stream #0:0: Video: ..., 720x480, ... 25 tbr, ...
            import re

            size = re.search(r"(\d{2,6})x(\d{2,6})", line)
            if size:
                width, height = int(size.group(1)), int(size.group(2))
            fps_match = re.search(r"([\d.]+(?:/\d+)?)\s+tbr", line)
            if fps_match:
                fps = _parse_fps(fps_match.group(1))
            low = line.lower()
            if "tff" in low:
                field_order = "tff"
            elif "bff" in low:
                field_order = "bff"
            elif "progressive" in low:
                field_order = "progressive"

    if width == 0 or height == 0:
        raise FFmpegError(
            f"could not read video stream from {path}\nffmpeg said:\n{banner[-800:]}"
        )

    frame_count = int(round(duration * fps)) if fps > 0 else 0
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        field_order=field_order,
        raw_banner=banner,
    )


def decode_frames(
    path: str | Path,
    info: VideoInfo,
    ffmpeg: str | None = None,
):
    """Yield decoded frames one at a time as HxWx3 uint8 RGB arrays."""
    ffmpeg = ffmpeg or find_ffmpeg()
    frame_bytes = info.width * info.height * 3
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                info.height, info.width, 3
            )
            yield frame.copy()
    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        proc.wait()
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg decode failed: {stderr.strip()}")


def encode_frames(
    frames,
    out_path: str | Path,
    width: int,
    height: int,
    fps: float,
    ffmpeg: str | None = None,
    crf: int = 16,
) -> int:
    """Stream RGB frames into an H.264 progressive MP4. Returns frame count."""
    ffmpeg = ffmpeg or find_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        # Output is reconstructed progressive content; never re-mark interlaced.
        "-field_order",
        "progressive",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        assert proc.stdin is not None
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"frame {count} shape {frame.shape} != {(height, width, 3)}"
                )
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            count += 1
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        proc.wait()
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg encode failed: {stderr.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        elif proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    return count


def write_frame_png(frame: np.ndarray, out_path: str | Path) -> None:
    """Persist a single RGB frame as PNG for the inspection UI."""
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8), mode="RGB").save(out_path)


def write_manifest(payload: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
