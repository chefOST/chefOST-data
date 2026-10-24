import unittest

from full_video_run import response_text, schema


class FullVideoRunTest(unittest.TestCase):
    def test_transition_contract_and_response_parsing(self):
        fields = schema()["properties"]["transitions"]["items"]["required"]
        self.assertEqual(fields, ["frame", "object", "state_before", "state_after"])
        self.assertEqual(response_text({"output": [{"content": [{"type": "output_text", "text": "{}"}]}]}), "{}")


if __name__ == "__main__":
    unittest.main()
