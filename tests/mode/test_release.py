import unittest

from training_data_packer.mode.release import parallel_package_pipeline


class TestParallelPackagePipeline(unittest.TestCase):
    def test_first(self):
        src_data = [
            {
                "source_text": "Antarctica is the coldest place on Earth.",
                "target_text": "Antarktida është kontinent më i ftohtë në tokë.",
                "src_lang": "eng_Latn",
                "tgt_lang": "als_Latn",
            }
        ]
        metadata = {
            "_internal": {"parallel": True},
            "text": "text",
            "parallel": {
                "input": "source",
                "source": {
                    "text": "source_text",
                    "language": "src_lang",
                },
                "target": {
                    "text": "target_text",
                    "language": "tgt_lang",
                },
            },
        }
        part_config = {}
        piis = iter([])
        contaminations = iter([])
        expect_on_of = [
            {
                "id": "14882f09150ccfb56e01519d5913ae8c869d910c018d69b3b01eba884595473f",
                "text": "English: Antarctica is the coldest place on Earth.\n"
                "Tosk Albanian: Antarktida është kontinent më i ftohtë në tokë.",
            },
            {
                "id": "9b27af5e0a5e7ed3630448a0b04731b7a986d9f899873ffdff8694c7359896bb",
                "text": "Tosk Albanian: Antarktida është kontinent më i ftohtë në tokë.\n"
                "English: Antarctica is the coldest place on Earth.",
            },
        ]

        result, metrics = parallel_package_pipeline(src_data, metadata, part_config, piis, contaminations)
        result = list(result)
        found = False
        for e in expect_on_of:
            if e == result[0]:
                found = True
                break
        self.assertTrue(found)
