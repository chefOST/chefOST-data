import unittest

from pipeline import parse_label, smooth


class PipelineTest(unittest.TestCase):
    def test_labels_and_smoothing(self):
        self.assertEqual(parse_label("Assistant: toasted"), "toasted")
        self.assertEqual(parse_label("partially_toasted"), "partially_toasted")
        self.assertEqual(parse_label("unclear"), "not_visible")
        self.assertEqual(
            smooth(["raw"] * 3 + ["toasted"] * 3 + ["raw"] * 3),
            ["not_visible", "not_visible", "raw", "raw", "raw", "toasted", "toasted", "toasted", "toasted"],
        )
        self.assertEqual(smooth(["raw", "raw", "not_visible", "raw"]), ["not_visible"] * 4)


if __name__ == "__main__":
    unittest.main()
