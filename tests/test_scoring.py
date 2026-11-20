import copy

import pytest

from chefost_eval.embeddings import CachedEmbeddings, QWEN_MODEL, cosine, label_text
from chefost_eval.labels import AssistedMapping, Vocabulary
from chefost_eval.scoring import THRESHOLDS, evaluate, match_events


@pytest.fixture
def vocabulary():
    return Vocabulary({"s2i": {"raw": 0, "chopped": 1, "cooked": 2},
                       "all2one": {"cut": "chopped", "": "cooked"}})


def video(gt, windows, *, name="egg", key="CMU/a", available=0, present=None, visible=None):
    n = len(gt)
    return {"key": key, "dataset": "CMU", "fps": 1, "frames": list(range(n)), "ingredients": {
        name: {"gt": gt, "present": present or [True] * n, "visible": visible or [True] * n,
               "available_from_frame": available, "windows": windows}}}


def window(start, end, labels=None, probabilities=None):
    return {"start_frame": start, "end_frame": end,
            **({"labels": labels} if labels is not None else {"probabilities": probabilities})}


def score(videos, vocabulary, **kwargs):
    return evaluate(videos, {"CMU": vocabulary}, method=kwargs.pop("method", "claude"), **kwargs)


def test_frame_ratios_not_micro_counts(vocabulary):
    row = score([video([["raw"], ["raw", "cooked"]],
                      [window(0, 1, ["raw"]), window(1, 2, ["wrong", "other"])])], vocabulary)["scores"][0]
    assert row["precision"] == .5  # pooled precision would be 1/3
    assert row["recall"] == .5


def test_unannotated_frames_and_initial_support_are_neutral(vocabulary):
    v = video([["raw"], ["raw"], [], ["cooked"], ["cooked"]],
              [window(1, 3, ["raw"])], available=1)
    row = score([v], vocabulary)["scores"][0]
    assert row["precision"] == 1
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["videos"]["CMU/a"]["ingredients"]["egg"]["eligible_frames"] == 3


def test_outer_presence_bounds(vocabulary):
    v = video([["raw"]] * 4, [window(0, 4, ["wrong"])], present=[False, True, True, False])
    row = score([v], vocabulary)["scores"][0]["videos"]["CMU/a"]["ingredients"]["egg"]
    assert row["eligible_frames"] == 2


def test_equal_ingredients_then_equal_videos(vocabulary):
    a = video([["raw"]] * 10, [window(0, 10, ["raw"])])
    second = video([["raw"]] * 10, [window(0, 10, ["wrong"])], name="bread")
    a["ingredients"].update(second["ingredients"])
    b = video([["raw"]], [window(0, 1, ["wrong"])], key="CMU/b")
    assert score([a, b], vocabulary)["scores"][0]["precision"] == .25


def test_thresholds_select_multiple_before_mapping_keep_unknowns(vocabulary):
    v = video([["chopped"]], [window(0, 1, probabilities={"cut": .8, "chopped": .5, "invented": .3})])
    rows = score([v], vocabulary, method="mosp")["scores"]
    assert [r["threshold"] for r in rows] == list(THRESHOLDS)
    assert [r["precision"] for r in rows] == [.5, 1, 1]
    assert [r["recall"] for r in rows] == [1, 1, 1]


def test_threshold_abstention_never_extends_initial_exclusion(vocabulary):
    v = video([["raw"]] * 2, [window(0, 2, probabilities={"raw": .4})])
    row = score([v], vocabulary, method="vidosc")["scores"][2]
    assert row["precision"] is None
    assert row["recall"] == 0
    assert row["videos"]["CMU/a"]["ingredients"]["egg"]["eligible_frames"] == 2


def test_replacement_does_not_accumulate_states(vocabulary):
    v = video([["raw"], ["cooked"]], [window(0, 1, ["raw"]), window(1, 2, ["cooked"])])
    row = score([v], vocabulary)["scores"][0]
    assert row["precision"] == row["recall"] == 1
    assert row["timing"]["f1"] == 1


def test_timing_resets_at_gaps_and_initial_frame(vocabulary):
    v = video([["raw"], [], ["cooked"]], [window(0, 1, ["raw"]), window(1, 3, ["cooked"])])
    timing = score([v], vocabulary)["scores"][0]["timing"]
    assert timing["tp"] == timing["fp"] == timing["fn"] == 0
    assert timing["f1"] is None


def test_tolerance_uses_gt_before_initial_exclusion(vocabulary):
    gt = [["raw"], ["cooked"], ["raw"], ["raw"], ["raw"], ["cooked"]]
    v = video(gt, [window(5, 6, ["cooked"])], available=5)
    output = score([v], vocabulary)
    assert output["gt_interevent_gap_count"] == 2
    assert output["timing_tolerance_seconds"] == 1


def test_matching_maximizes_count_then_minimizes_error():
    assert match_events([0, 2], [1, 3], 1) == (2, 2)
    assert match_events([0, 2], [0, 1, 2], 2) == (2, 0)
    assert match_events([1], [0, 1, 2], 1) == (1, 0)


def test_unmatched_nearby_event_is_not_merged(vocabulary):
    v = video([["raw"], ["cooked"], ["cooked"]],
              [window(0, 1, ["raw"]), window(1, 2, ["cooked"]), window(2, 3, ["raw"])])
    timing = score([v], vocabulary)["scores"][0]["timing"]
    assert timing["tp"] == timing["fp"] == 1
    assert timing["f1"] == pytest.approx(2 / 3)


def test_assisted_mapping_preserves_unmapped_remainder(vocabulary):
    cache = {"model": "gpt-6.2-sol", "prompt_sha256": "a" * 64,
             "vocabulary_sha256": {"CMU": "b" * 64}, "entries": {
                 "CMU": {"cut and purple": {"states": ["chopped"], "unmatched": True}}}}
    mapper = AssistedMapping(cache, {"CMU": vocabulary}, {"CMU": "b" * 64})
    rows = score([video([["chopped"]], [window(0, 1, ["cut and purple"])])],
                 vocabulary, assisted=mapper)["scores"]
    assert rows[0]["precision"] == 0
    assert rows[1]["precision"] == .5
    assert rows[1]["recall"] == 1
    with pytest.raises(ValueError, match="Missing cached"):
        mapper.map("CMU", ["raw"])


def test_cosine_is_per_frame_whole_raw_set_and_macro(vocabulary):
    raw = {"model": QWEN_MODEL, "revision": "test", "text_format": "sorted_unique_labels_json",
           "instruction": "", "vectors": {label_text(["raw"]): [1, 0], label_text(["cut"]): [0, 1],
                                            label_text(["chopped"]): [0, 1]}}
    provider = CachedEmbeddings(raw)
    v = video([["raw"], ["chopped"], ["raw"]], [window(0, 1, ["raw"]), window(1, 2, ["cut"])])
    assert score([v], vocabulary, embeddings=provider)["scores"][0]["cosine_similarity"] == pytest.approx(2 / 3)
    assert label_text(["raw", "cut", "raw"]) == '["cut", "raw"]'
    with pytest.raises(ValueError, match="dimensions"):
        cosine([1, 0], [1])


def test_blank_alias_cannot_become_a_state(vocabulary):
    with pytest.raises(ValueError, match="nonempty"):
        vocabulary.strict([""])
