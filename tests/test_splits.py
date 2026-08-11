"""Offline split-policy tests: no dataset download required.

These guard the split policy structurally — determinism from seed 42, 80/20
proportions, disjointness of the committed files, and the rule that no code path can
address `beir/scifact/test`. The dataset-dependent versions live in `tests/test_data.py`.
"""

import pytest

from retrieval.data import (
    DEV_FRACTION,
    SEED,
    generate_calibration_split,
    read_split_file,
    resolve_split,
    validate_calibration_splits,
)

COMMITTED_TRAIN = "splits/calibration_train.txt"
COMMITTED_DEV = "splits/calibration_dev.txt"


def test_committed_split_files_are_disjoint_and_deduplicated():
    train_ids = read_split_file(COMMITTED_TRAIN)
    dev_ids = read_split_file(COMMITTED_DEV)

    assert set(train_ids).isdisjoint(dev_ids)
    assert len(train_ids) == len(set(train_ids))
    assert len(dev_ids) == len(set(dev_ids))


def test_committed_split_proportions_are_80_20():
    train_ids = read_split_file(COMMITTED_TRAIN)
    dev_ids = read_split_file(COMMITTED_DEV)
    total = len(train_ids) + len(dev_ids)

    assert len(dev_ids) == round(total * DEV_FRACTION)
    assert len(dev_ids) / total == pytest.approx(DEV_FRACTION, abs=0.01)


def test_split_generation_is_deterministic_and_order_independent():
    ids = [f"q{i}" for i in range(100)]
    train_a, dev_a = generate_calibration_split(ids, seed=SEED)
    train_b, dev_b = generate_calibration_split(list(reversed(ids)), seed=SEED)

    # generate_calibration_split sorts before shuffling, so input order cannot change
    # the split — this is what makes the files independent of ir_datasets iteration order.
    assert (train_a, dev_a) == (train_b, dev_b)
    assert len(dev_a) == 20
    assert set(train_a).isdisjoint(dev_a)


def test_different_seed_produces_a_different_split():
    ids = [f"q{i}" for i in range(100)]
    assert generate_calibration_split(ids, seed=SEED) != generate_calibration_split(ids, seed=7)


def test_validate_calibration_splits_rejects_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        validate_calibration_splits(["a", "b"], ["b"], ["a", "b"])


def test_validate_calibration_splits_rejects_incomplete_coverage():
    with pytest.raises(ValueError, match="does not equal"):
        validate_calibration_splits(["a"], ["b"], ["a", "b", "c"])


@pytest.mark.parametrize("split_arg", ["test", "beir/scifact/test", "train", "calibration_dev"])
def test_resolve_split_refuses_anything_but_the_two_calibration_splits(split_arg):
    # Held-out data is not reachable by passing a different string to the calibration loader.
    # It has its own named function, so opening it is always an explicit, greppable act.
    train_data = {"corpus": {}, "queries": {"q1": "a"}, "qrels": {"q1": {"d1": 1}}}
    with pytest.raises(ValueError, match="Unknown split"):
        resolve_split(split_arg, train_data)


def test_final_test_split_has_a_separate_deliberate_loader():
    # The guard above is only meaningful if the held-out path is a distinct, named entry point
    # rather than a branch inside the calibration loader.
    from retrieval import data as data_module

    assert data_module.FINAL_TEST_SPLIT == "test"
    assert callable(data_module.load_final_test_split)
    assert "test" not in data_module._SPLIT_FILES


def test_resolve_split_filters_queries_and_qrels_to_the_split(tmp_path):
    (tmp_path / "calibration_dev.txt").write_text("q1\nq3\n")
    train_data = {
        "corpus": {"d1": "doc one", "d2": "doc two"},
        "queries": {"q1": "a", "q2": "b", "q3": "c"},
        "qrels": {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d2": 1}},
    }

    resolved = resolve_split("calibration-dev", train_data, splits_dir=tmp_path)

    assert set(resolved["queries"]) == {"q1", "q3"}
    assert set(resolved["qrels"]) == {"q1", "q3"}
    assert resolved["corpus"] == train_data["corpus"]  # corpus is shared across splits
