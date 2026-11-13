"""Command-line entry point for the archive deinterlacer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import ffmpeg_io
from .core.benchmark import evaluate_and_report
from .core.diagnostics import diagnose
from .core.pipeline import ALL_MODES, DeinterlaceConfig, export, run
from .core.synthetic import generate_reference_set, make_ground_truth


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_generate(args) -> int:
    out = Path(args.out)
    report = generate_reference_set(out)
    _print(report)
    print(f"reference set written to {out}", file=sys.stderr)
    return 0


def cmd_diagnose(args) -> int:
    info = ffmpeg_io.probe(args.input)
    frames = list(ffmpeg_io.decode_frames(args.input, info))
    report = diagnose(frames, info.field_order).to_dict()
    report["container"] = {
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "duration": info.duration,
    }
    _print(report)
    if args.strict and report["blocking"] != "none":
        return 2
    return 0


def cmd_deinterlace(args) -> int:
    cfg = DeinterlaceConfig(
        field_order=args.field_order,
        mode=args.mode,
        diff_threshold=args.diff_threshold,
        neighborhood_threshold=args.neighborhood_threshold,
        crf=args.crf,
    )
    info = ffmpeg_io.probe(args.input)
    frames = list(ffmpeg_io.decode_frames(args.input, info))

    report = diagnose(frames, info.field_order)
    if report.blocking == "progressive_input":
        print(
            "refusing: input is already progressive; deinterlacing would "
            "invent artifacts",
            file=sys.stderr,
        )
        return 2
    if args.field_order is None:
        print(
            "refusing: explicit --field-order tff|bff is required; diagnostics:\n"
            + json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2

    result = run(frames, info.fps, cfg)
    base = Path(args.input).stem
    manifest = export(result, args.out, base_name=base, crf=args.crf)
    _print(
        {
            "diagnosis": report.to_dict(),
            "export": manifest,
        }
    )
    return 0


def cmd_benchmark(args) -> int:
    out = Path(args.out)
    _, truth = make_ground_truth(out)
    report_path = evaluate_and_report(truth, out)
    print(f"benchmark report: {report_path}", file=sys.stderr)
    _print(json.loads(report_path.read_text()))
    return 0


def cmd_perf(args) -> int:
    from .core.perf import run_suite

    out = Path(args.out)
    report = run_suite(out)
    report_path = report.pop("report_path", None)
    print(f"field-processing perf report: {report_path}", file=sys.stderr)
    _print(report)
    return 0


def cmd_serve(args) -> int:
    from .web.app import create_app

    app = create_app(Path(args.workdir))
    print(f"serving archive deinterlacer on http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archive-deinterlacer",
        description="Field-order-aware TV archive deinterlacing toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate synthetic interlaced references")
    p.add_argument("--out", default="output/synthetic")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("diagnose", help="diagnose an input clip without modifying it")
    p.add_argument("input")
    p.add_argument("--strict", action="store_true",
                   help="exit 2 when metadata is missing/unreliable for a decision")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("deinterlace", help="reconstruct and export a progressive clip")
    p.add_argument("input")
    p.add_argument("--out", default="output/jobs")
    p.add_argument("--field-order", choices=["tff", "bff"], default=None,
                   help="explicit capture order; required, never guessed")
    p.add_argument("--mode", choices=ALL_MODES, default="adaptive")
    p.add_argument("--diff-threshold", type=float, default=10.0)
    p.add_argument("--neighborhood-threshold", type=float, default=6.0)
    p.add_argument("--crf", type=int, default=16)
    p.set_defaults(func=cmd_deinterlace)

    p = sub.add_parser("benchmark", help="score modes against synthetic ground truth")
    p.add_argument("--out", default="output/benchmark")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser(
        "perf",
        help="field-processing acceleration A/B: stage cost, comb error, per-frame cost",
    )
    p.add_argument("--out", default="output/perf")
    p.set_defaults(func=cmd_perf)

    p = sub.add_parser("serve", help="start the web correction interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workdir", default="output/web")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
