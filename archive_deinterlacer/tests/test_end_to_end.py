"""End-to-end tests through ffmpeg: synthetic generation, export, web API."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from archive_deinterlacer.core import ffmpeg_io
from archive_deinterlacer.core.diagnostics import diagnose
from archive_deinterlacer.core.pipeline import (
    MODE_ADAPTIVE,
    MODE_BOB,
    DeinterlaceConfig,
    export,
    run,
)
from archive_deinterlacer.core.synthetic import (
    GROUND_FPS,
    SOURCE_FPS,
    emulate_interlaced,
    make_ground_truth,
)


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_synthetic_clip_roundtrips_and_diagnoses_interlaced(self):
        _, truth = make_ground_truth(self.out)
        interlaced = emulate_interlaced(truth, "tff")
        clip = self.out / "tff.mp4"
        ffmpeg_io.encode_frames(iter(interlaced), clip,
                                truth[0].shape[1], truth[0].shape[0], SOURCE_FPS)
        info = ffmpeg_io.probe(clip)
        decoded = list(ffmpeg_io.decode_frames(clip, info))
        self.assertEqual(len(decoded), len(interlaced))
        report = diagnose(decoded, info.field_order)
        self.assertEqual(report.content_verdict, "interlaced")
        self.assertGreater(report.comb_energy, 0.5)

    def test_encoded_progressive_clip_is_refused(self):
        """4:2:0 fake teeth must not make a progressive clip look interlaced."""
        truth_path, _ = make_ground_truth(self.out)
        info = ffmpeg_io.probe(truth_path)
        decoded = list(ffmpeg_io.decode_frames(truth_path, info))
        report = diagnose(decoded, info.field_order)
        self.assertEqual(report.content_verdict, "progressive")
        self.assertEqual(report.blocking, "progressive_input")

    def test_export_manifest_matches_video_timing(self):
        _, truth = make_ground_truth(self.out)
        interlaced = emulate_interlaced(truth, "bff")
        result = run(interlaced, SOURCE_FPS,
                     DeinterlaceConfig(field_order="bff", mode=MODE_ADAPTIVE))
        manifest = export(result, self.out, base_name="case")
        self.assertTrue((self.out / manifest["video_file"]).exists())
        self.assertTrue(manifest["duration_check"])
        self.assertEqual(manifest["output_frame_count"], result.frame_count)
        self.assertEqual(manifest["output_fps"], result.output_fps)
        # Re-probe the exported file: it must be progressive marked.
        exported_info = ffmpeg_io.probe(self.out / manifest["video_file"])
        self.assertIn(exported_info.field_order, ("progressive", "unknown"))
        self.assertAlmostEqual(exported_info.fps, result.output_fps, places=2)

    def test_web_app_workflow(self):
        from archive_deinterlacer.web.app import create_app

        _, truth = make_ground_truth(self.out / "synth")
        interlaced = emulate_interlaced(truth, "tff")
        clip = self.out / "synth" / "up.mp4"
        ffmpeg_io.encode_frames(iter(interlaced), clip,
                                truth[0].shape[1], truth[0].shape[0], SOURCE_FPS)

        app = create_app(self.out / "webwork")
        client = app.test_client()
        with open(clip, "rb") as fh:
            resp = client.post(
                "/api/upload?analyze=sync",
                data={"video": (io.BytesIO(fh.read()), "up.mp4")},
                content_type="multipart/form-data",
            )
        job_id = resp.get_json()["job_id"]

        # Missing explicit order must be refused, not guessed.
        resp = client.post(f"/api/jobs/{job_id}/process", json={"mode": MODE_BOB})
        self.assertEqual(resp.status_code, 400)

        state = client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(state["status"], "analyzed")

        resp = client.post(
            f"/api/jobs/{job_id}/process?run=sync",
            json={"field_order": "tff", "mode": MODE_ADAPTIVE},
        )
        self.assertEqual(resp.status_code, 200)
        state = client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(state["status"], "done", state.get("error"))

        # Field and final-frame endpoints respond with PNG images.
        r = client.get(f"/api/jobs/{job_id}/field/0?order=tff")
        self.assertEqual(r.mimetype, "image/png")
        r.close()
        r = client.get(f"/api/jobs/{job_id}/result/0?overlay=motion")
        self.assertEqual(r.mimetype, "image/png")
        r.close()
        r = client.get(f"/api/jobs/{job_id}/download/manifest")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data)["field_order"], "tff")
        r.close()


if __name__ == "__main__":
    unittest.main()
