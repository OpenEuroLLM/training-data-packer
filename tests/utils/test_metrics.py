import unittest
from typing import Dict, Any

from training_data_packer.utils import metrics

class DummyClassWithMetrics:
    def __init__(self, test_metrics: Dict[str, Any]):
        self.my_metrics = test_metrics

    def get_metrics(self) -> Dict[str, Any]:
        return self.my_metrics


class TestMetrics(unittest.TestCase):
    def test_collect_metrics_with_only_nones(self):
        result = metrics.collect_metrics(None, None)
        self.assertEqual(result, {})

    def test_collect_metrics_with_objects(self):
        object1 = DummyClassWithMetrics({"foo": "bar"})
        object2 = DummyClassWithMetrics({"gazonk": {"lines": 5}})
        result = metrics.collect_metrics(object1, None, object2)
        self.assertEqual(result, {"foo": "bar", "gazonk": {"lines": 5}})


if __name__ == '__main__':
    unittest.main()
