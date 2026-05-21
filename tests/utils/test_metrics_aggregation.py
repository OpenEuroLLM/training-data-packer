import unittest

from training_data_packer.utils import metrics


class TestMetricsAggregation(unittest.TestCase):
    def test_aggregate_metrics(self):
        m1 = {"input": {"lines_read": 10}, "output": {"lines_written": 8}}
        m2 = {"input": {"lines_read": 20}, "output": {"lines_written": 15}}

        result = metrics.aggregate_metrics([m1, m2])

        expected = {"input": {"lines_read": 30}, "output": {"lines_written": 23}}
        self.assertEqual(result, expected)

    def test_aggregate_metrics_with_missing_keys(self):
        m1 = {
            "input": {"lines_read": 10},
        }
        m2 = {"output": {"lines_written": 15}}

        result = metrics.aggregate_metrics([m1, m2])

        expected = {"input": {"lines_read": 10}, "output": {"lines_written": 15}}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
