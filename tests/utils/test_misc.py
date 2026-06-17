import unittest

from parameterized import parameterized

from training_data_packer.utils.misc import lang_to_name


class TestLangToName(unittest.TestCase):
    @parameterized.expand(
        [
            [
                "basic_eng",
                "eng",
                "English",
            ],
            [
                "basic_spa",
                "spa_Latn",
                "Spanish",
            ],
            [
                "basic_fra",
                "fra_Latn",
                "French",
            ],
            [
                "basic_deu",
                "deu_Latn",
                "German",
            ],
            [
                "basic_ita",
                "ita_Latn",
                "Italian",
            ],
            [
                "locale_eng_Latn",
                "eng_Latn",
                "English",
            ],
        ]
    )
    def test_lang_to_name(self, name, lang_code, expected_name):
        result = lang_to_name(lang_code)
        self.assertEqual(expected_name, result)


if __name__ == "__main__":
    unittest.main()
