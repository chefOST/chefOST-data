import csv
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import consensus, enforce_consistency, finalize, find_changes, keyframes, windows


class PipelineTest(unittest.TestCase):
    def test_windows_overlap_without_tiny_duplicate_tail(self):
        self.assertEqual(windows(20), [(0, 12), (8, 20)])
        self.assertEqual(windows(13), [(0, 12), (8, 13)])

    def test_consensus_abstains_on_disagreement(self):
        proposals = {
            0: [
                {"state": "pale", "confidence": 0.95, "evidence": "pale"},
                {"state": "browned", "confidence": 0.95, "evidence": "brown"},
            ]
        }
        self.assertEqual(consensus(proposals, 1)[0]["state"], "uncertain")

    def test_consistency_and_changes(self):
        timeline = [
            {"sample_index": i, "timestamp_sec": i, "state": state, "confidence": 0.95, "evidence": "visible"}
            for i, state in enumerate(["pale", "pale", "browned", "pale", "browned", "browned"])
        ]
        transitions = {"pale": ["browned"], "browned": []}
        result = enforce_consistency(timeline, transitions)
        self.assertEqual([row["state"] for row in result], ["pale", "pale", "uncertain", "uncertain", "browned", "browned"])
        self.assertEqual(find_changes(result, transitions)[0]["frame"], 4)

    def test_keyframes_skip_uncertain_and_collapse_repeated_states(self):
        timeline = [
            {"sample_index": 0, "state": "uncertain"},
            {"sample_index": 2, "state": "whole"},
            {"sample_index": 3, "state": "whole"},
            {"sample_index": 4, "state": "occluded"},
            {"sample_index": 7, "state": "halved"},
        ]
        self.assertEqual(keyframes(timeline), [{"frame": 2, "state": "whole"}, {"frame": 7, "state": "halved"}])

    def test_finalize_requires_every_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "review.csv"
            output = Path(directory) / "changes.jsonl"
            fields = ["frame", "approved"]
            with review.open("w", newline="") as file:
                writer = csv.DictWriter(file, fields)
                writer.writeheader()
                writer.writerow({"frame": 2, "approved": ""})
            with self.assertRaises(RuntimeError):
                finalize(review, output)
            review.write_text("frame,approved\n2,yes\n3,no\n")
            self.assertEqual(finalize(review, output), 1)
            self.assertEqual(json.loads(output.read_text())["frame"], "2")


if __name__ == "__main__":
    unittest.main()
