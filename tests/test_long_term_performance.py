import unittest

from feedsummary_core.long_term import (
    PerformanceBenchmarkSettings,
    run_long_term_performance_benchmark,
)


class LongTermPerformanceTests(unittest.TestCase):
    def test_90_and_180_day_datasets_stay_within_ci_capacity_budget(self):
        for days in (90, 180):
            with self.subTest(days=days):
                result = run_long_term_performance_benchmark(
                    PerformanceBenchmarkSettings(
                        days=days,
                        events_per_day=2,
                        embedding_dimensions=16,
                        candidate_probes=20,
                    )
                )

                expected_events = days * 2
                self.assertEqual(expected_events, result["dataset"]["event_count"])
                self.assertEqual(expected_events, result["counts"]["cluster_count"])
                self.assertEqual(expected_events, result["counts"]["membership_count"])
                self.assertLess(result["timings_ms"]["total"], 10_000)
                self.assertLess(result["peak_memory_mb"], 256)
                self.assertGreater(result["counts"]["selected_snapshot_count"], 0)
                self.assertGreater(result["counts"]["reduce_input_chars"], 0)


if __name__ == "__main__":
    unittest.main()
