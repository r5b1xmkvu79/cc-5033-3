"""Tests for the field-processing acceleration.

The acceleration is only allowed to change *cost*, never behavior. These
tests pin that contract: reusable splits carry identical rows and timing,
the uint8-domain motion verdict matches the int16 original, the accelerated
reconstruction matches the frozen legacy builder pixel for pixel, and the
performance suite's correctness gate actually fails on injected regressions.
"""

import unittest

import numpy as np

from archive_deinterlacer.core import fields as F
from archive_deinterlacer.core import perf as P
from archive_deinterlacer.core.diagnostics import (
    _samples_for_both_orders,
    _samples_for_order,
    comb_energy,
    detect_progressive,
    diagnose,
)
from archive_deinterlacer.core.pipeline import (
    DeinterlaceConfig,
    run,
)


def _moving_interlaced(n=8, h=24, w=32, seed=1):
    rng = np.random.default_rng(seed)
    ground = []
    for k in range(n * 2):
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        x = (k * 3) % (w - 6)
        img[4:20, x:x + 5] = 230
        # 1 px thin strokes so the fine-line criteria are exercised.
        img[2:22, (k * 7) % w, :] = 0
        ground.append(img)
    interlaced = []
    for k in range(n):
        frame = np.empty_like(ground[0])
        frame[0::2] = ground[2 * k][0::2]
        frame[1::2] = ground[2 * k + 1][1::2]
        interlaced.append(frame)
    return interlaced


class ReusableSplitTest(unittest.TestCase):
    def test_reusable_split_matches_stream_split_for_both_orders(self):
        frames = _moving_interlaced()
        for order in (F.TFF, F.BFF):
            old = list(F.split_stream(iter(frames), order))
            new = F.split_fields_reusable(iter(frames), order)
            self.assertEqual(len(old), len(new))
            for a, b in zip(old, new):
                self.assertEqual(a.index, b.index)
                self.assertEqual(a.frame_index, b.frame_index)
                self.assertEqual(a.position, b.position)
                np.testing.assert_array_equal(a.data, b.data)

    def test_both_orders_share_one_row_copy_pass(self):
        frames = _moving_interlaced()
        tff, bff = _samples_for_both_orders(frames)
        tff_direct = _samples_for_order(frames, F.TFF)
        bff_direct = _samples_for_order(frames, F.BFF)
        for got, ref in ((tff, tff_direct), (bff, bff_direct)):
            for a, b in zip(got, ref):
                self.assertEqual((a.index, a.position), (b.index, b.position))
                np.testing.assert_array_equal(a.data, b.data)
        # Row arrays are physically shared between the two order reads: BFF
        # wraps TFF's copied rows instead of allocating a second copy.
        self.assertTrue(np.shares_memory(tff[0].data, bff[1].data))
        self.assertTrue(np.shares_memory(tff[1].data, bff[0].data))


class MotionMaskEquivalenceTest(unittest.TestCase):
    def test_uint8_mask_is_bit_identical_to_int16_original(self):
        rng = np.random.default_rng(7)
        for shape in ((40, 48, 3), (5, 7, 3), (1, 1, 3)):
            a = rng.integers(0, 256, shape, np.uint8)
            b = rng.integers(0, 256, shape, np.uint8)
            np.testing.assert_array_equal(
                F.motion_mask(a, b), F.motion_mask_fast(a, b)
            )
            np.testing.assert_array_equal(
                F.motion_mask(a, b, 3.5, 2.0),
                F.motion_mask_fast(a, b, 3.5, 2.0),
            )

    def test_thresholds_still_reject_isolated_noise_on_fine_detail(self):
        # One pixel sits between the diff threshold and the neighborhood
        # threshold: its own luma jump looks like noise, but no 3x3 neighbor
        # supports motion, so the second criterion suppresses it in both paths.
        a = np.full((10, 10, 3), 100, np.uint8)
        b = a.copy()
        b[5, 5] = 108  # diff 8 > diff_threshold 6 but <= neighborhood 9
        self.assertFalse(F.motion_mask_fast(a, b, 6.0, 9.0).any())
        self.assertFalse(F.motion_mask(a, b, 6.0, 9.0).any())


class PipelineParityTest(unittest.TestCase):
    def test_every_mode_and_order_matches_frozen_legacy(self):
        frames = _moving_interlaced()
        for order in (F.TFF, F.BFF):
            for mode in ("adaptive", "bob", "weave"):
                cfg = DeinterlaceConfig(field_order=order, mode=mode)
                legacy = run(frames, 25.0, cfg, accelerated=False)
                fast = run(frames, 25.0, cfg, accelerated=True)
                self.assertEqual(legacy.frame_count, fast.frame_count)
                self.assertEqual(legacy.output_fps, fast.output_fps)
                self.assertEqual(
                    legacy.expected_duration, fast.expected_duration
                )
                self.assertAlmostEqual(
                    legacy.motion_pixel_ratio, fast.motion_pixel_ratio
                )
                for a, b in zip(legacy.records, fast.records):
                    np.testing.assert_array_equal(a.rgb, b.rgb)
                    if a.motion_mask is None:
                        self.assertIsNone(b.motion_mask)
                    else:
                        np.testing.assert_array_equal(a.motion_mask, b.motion_mask)
                    self.assertEqual(a.timestamp, b.timestamp)

    def test_optimized_missing_rows_equal_full_spatial_interpolation(self):
        rng = np.random.default_rng(11)
        for position in (F.TOP, F.BOTTOM):
            field = rng.integers(0, 256, (12, 16, 3), np.uint8)
            sample = F.FieldSample(0, 0, position, field)
            full = F._spatial_interpolate(field, position)
            missing, sl = F._spatial_missing_rows(field, position)
            np.testing.assert_array_equal(full[sl], missing)
            own = slice(0, None, 2) if position == F.TOP else slice(1, None, 2)
            np.testing.assert_array_equal(full[own], field)


class DiagnosticsStabilityTest(unittest.TestCase):
    def test_diagnosis_unchanged_after_split_reuse(self):
        frames = _moving_interlaced(n=10)
        report = diagnose(frames, "unknown")
        self.assertEqual(report.content_verdict, "interlaced")
        # detect_progressive with reused comb values matches standalone call.
        tff = _samples_for_order(frames, F.TFF)
        comb_values = [comb_energy(f) for f in frames]
        score_a, res_a = detect_progressive(frames, tff)
        score_b, res_b = detect_progressive(frames, tff, comb_values)
        self.assertEqual(score_a, score_b)
        self.assertEqual(res_a, res_b)


class PerfSuiteTest(unittest.TestCase):
    def test_workloads_keep_full_field_count_and_resolution(self):
        thin = P.make_thin_line_workload(frames=6)
        self.assertEqual(len(thin), 6)
        self.assertEqual(thin[0].shape, (P.PERF_HEIGHT, P.PERF_WIDTH, 3))
        orders = P.make_field_order_workload(frames=6)
        self.assertEqual(len(orders[F.TFF]), len(orders[F.BFF]))

    def test_measure_clip_reports_exact_parity_and_real_comb(self):
        thin = P.make_thin_line_workload(frames=4)
        result = P.measure_clip(thin, F.TFF, "unit_thin_line")
        self.assertEqual(result["parity"]["parity"], "exact")
        self.assertEqual(result["parity"]["max_pixel_diff"], 0)
        self.assertEqual(result["parity"]["mask_mismatch_frames"], 0)
        self.assertEqual(result["output_frames"], 8)
        # Costs must be measured, never silently defaulted to zero.
        self.assertGreater(
            result["legacy"]["total_seconds_per_frame"], 0.0
        )
        self.assertGreater(
            result["accelerated"]["total_seconds_per_frame"], 0.0
        )
        self.assertGreaterEqual(result["speedup_total_x"], 0.0)
        self.assertGreaterEqual(result["comb_residue_legacy"], 0.0)

    def test_parity_gate_fails_on_injected_pixel_regression(self):
        records_legacy = run(
            _moving_interlaced(n=4),
            25.0,
            DeinterlaceConfig(field_order=F.TFF),
            accelerated=False,
        ).records
        records_fast = run(
            _moving_interlaced(n=4),
            25.0,
            DeinterlaceConfig(field_order=F.TFF),
            accelerated=True,
        ).records
        P.assert_parity(records_legacy, records_fast, "clean")
        original = int(records_fast[3].rgb[0, 0, 0])
        records_fast[3].rgb[0, 0] = np.uint8(original + 1)
        with self.assertRaises(AssertionError):
            P.assert_parity(records_legacy, records_fast, "injected")

    def test_parity_gate_fails_on_dropped_frame(self):
        records_legacy = run(
            _moving_interlaced(n=4),
            25.0,
            DeinterlaceConfig(field_order=F.BFF),
            accelerated=False,
        ).records
        records_fast = run(
            _moving_interlaced(n=4),
            25.0,
            DeinterlaceConfig(field_order=F.BFF),
            accelerated=True,
        ).records[:-1]
        with self.assertRaises(AssertionError):
            P.assert_parity(records_legacy, records_fast, "dropped")


if __name__ == "__main__":
    unittest.main()
