"""Kernel-level tests for field semantics, timing and reconstruction."""

import unittest

import numpy as np

from archive_deinterlacer.core import fields as F
from archive_deinterlacer.core.pipeline import (
    MODE_ADAPTIVE,
    MODE_BOB,
    MODE_WEAVE,
    DeinterlaceConfig,
    run,
)


def moving_frames(n=6, h=16, w=20):
    """Build interlaced frames from a moving block, ground truth known."""
    ground = []
    for k in range(n * 2):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = k * 2
        img[4:12, x:x + 4] = 200
        ground.append(img)
    interlaced = []
    for k in range(n):
        frame = np.empty_like(ground[0])
        frame[0::2] = ground[2 * k][0::2]
        frame[1::2] = ground[2 * k + 1][1::2]
        interlaced.append(frame)
    return interlaced, ground


class FieldSemanticsTest(unittest.TestCase):
    def test_row_positions_are_order_independent(self):
        frame = np.arange(16 * 6 * 3, dtype=np.uint8).reshape(16, 6, 3)
        top, bottom = F.split_frame(frame, 0)
        np.testing.assert_array_equal(top.data, frame[0::2])
        np.testing.assert_array_equal(bottom.data, frame[1::2])
        self.assertEqual(top.position, F.TOP)
        self.assertEqual(bottom.position, F.BOTTOM)

    def test_tff_vs_bff_only_changes_capture_order(self):
        frames, _ = moving_frames(n=3)
        tff = list(F.split_stream(iter(frames), F.TFF))
        bff = list(F.split_stream(iter(frames), F.BFF))
        # Same field data sets, opposite ordering of positions at each instant.
        self.assertEqual([s.position for s in tff], ["top", "bottom"] * 3)
        self.assertEqual([s.position for s in bff], ["bottom", "top"] * 3)
        # Capture-time index 0 is the top field under TFF but the bottom under BFF.
        self.assertEqual(tff[0].index, 0)
        self.assertEqual(bff[0].index, 0)
        self.assertEqual(bff[0].position, F.BOTTOM)
        np.testing.assert_array_equal(tff[0].data, bff[1].data)
        np.testing.assert_array_equal(tff[1].data, bff[0].data)
        # Every capture-time index 0..2N-1 appears exactly once per ordering.
        self.assertEqual(sorted(s.index for s in tff), list(range(6)))
        self.assertEqual(sorted(s.index for s in bff), list(range(6)))

    def test_field_timestamps_run_at_double_rate(self):
        ts = F.field_timestamps(frame_count=3, fps=25.0)
        np.testing.assert_allclose(ts, [0, 0.02, 0.04, 0.06, 0.08, 0.10])

    def test_weave_roundtrip(self):
        frame = np.random.default_rng(0).integers(0, 255, (16, 8, 3), np.uint8)
        top, bottom = F.split_frame(frame, 0)
        np.testing.assert_array_equal(F.weave(top.data, bottom.data), frame)


class PipelineTimingTest(unittest.TestCase):
    def test_bob_doubles_frames_and_rate_keeping_duration(self):
        frames, _ = moving_frames(n=6)
        result = run(frames, 25.0, DeinterlaceConfig(field_order=F.TFF, mode=MODE_BOB))
        self.assertEqual(result.frame_count, 12)
        self.assertEqual(result.output_fps, 50.0)
        self.assertAlmostEqual(result.frame_count / result.output_fps,
                               result.expected_duration)
        np.testing.assert_allclose(
            [r.timestamp for r in result.records],
            np.arange(12) / 50.0,
        )

    def test_weave_keeps_frame_count_and_rate(self):
        frames, _ = moving_frames(n=6)
        result = run(frames, 25.0, DeinterlaceConfig(field_order=F.TFF, mode=MODE_WEAVE))
        self.assertEqual(result.frame_count, 6)
        self.assertEqual(result.output_fps, 25.0)
        self.assertAlmostEqual(result.expected_duration, 6 / 25.0)

    def test_adaptive_masks_moving_region_and_keeps_static_clean(self):
        frames, ground = moving_frames(n=6)
        result = run(frames, 25.0,
                     DeinterlaceConfig(field_order=F.TFF, mode=MODE_ADAPTIVE))
        self.assertEqual(result.frame_count, 12)
        self.assertGreater(result.motion_pixel_ratio, 0.0)
        # Static corner never moves: its mask must stay False throughout.
        for rec in result.records:
            self.assertFalse(rec.motion_mask[0:3, 0:3].any())

    def test_explicit_order_is_required(self):
        frames, _ = moving_frames(n=2)
        with self.assertRaises(ValueError):
            run(frames, 25.0, DeinterlaceConfig(field_order="guess", mode=MODE_BOB))

    def test_odd_height_rejected(self):
        odd = [np.zeros((15, 8, 3), np.uint8)]
        with self.assertRaises(ValueError):
            run(odd, 25.0, DeinterlaceConfig(field_order=F.TFF))


class CombDetectionTest(unittest.TestCase):
    def test_clean_frame_has_no_comb(self):
        from archive_deinterlacer.core.diagnostics import comb_energy
        clean = np.full((16, 8, 3), 120, np.uint8)
        self.assertAlmostEqual(comb_energy(clean), 0.0)

    def test_teeth_frame_has_comb(self):
        from archive_deinterlacer.core.diagnostics import comb_energy
        frame = np.zeros((16, 8, 3), dtype=np.uint8)
        frame[0::2] = 255
        frame[1::2] = 0
        self.assertGreater(comb_energy(frame), 20.0)


if __name__ == "__main__":
    unittest.main()
