"""Integration tests against the real SciFact dataset via ir_datasets.

These run as part of the default `uv run pytest` (plan §16 requires the split and
leakage checks); they skip, rather than fail, when the dataset is unreachable — see
`tests/conftest.py`. The offline structural split checks live in `tests/test_splits.py`.
"""

import pytest

from retrieval.data import (
    evaluable_query_ids,
    generate_calibration_split,
    load_or_build_calibration_splits,
    read_split_file,
    write_split_file,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def train_data(scifact):
    return scifact["train"]


@pytest.fixture(scope="module")
def test_data(scifact):
    return scifact["test"]


@pytest.fixture(scope="module")
def calibration_train_ids():
    return read_split_file("splits/calibration_train.txt")


@pytest.fixture(scope="module")
def calibration_dev_ids():
    return read_split_file("splits/calibration_dev.txt")


def test_scifact_loads_and_counts_are_sane(train_data, test_data):
    assert len(train_data["queries"]) > 0
    assert len(train_data["corpus"]) > 0
    assert len(train_data["qrels"]) > 0
    assert len(test_data["queries"]) > 0
    assert len(test_data["corpus"]) > 0
    assert len(test_data["qrels"]) > 0


def test_split_generation_disjoint_and_reproducible(train_data):
    ids = evaluable_query_ids(train_data["qrels"])
    train_ids, dev_ids = generate_calibration_split(ids, seed=42)
    assert set(train_ids).isdisjoint(dev_ids)

    train_ids2, dev_ids2 = generate_calibration_split(ids, seed=42)
    assert train_ids == train_ids2
    assert dev_ids == dev_ids2


def test_committed_split_files_match_seed_42_regeneration(
    train_data, calibration_train_ids, calibration_dev_ids
):
    # The committed splits/*.txt must be exactly what seed 42 produces today; if this
    # fails, either the files were hand-edited or the split policy changed silently.
    ids = evaluable_query_ids(train_data["qrels"])
    expected_train, expected_dev = generate_calibration_split(ids, seed=42)
    assert calibration_train_ids == expected_train
    assert calibration_dev_ids == expected_dev


def test_split_ids_have_qrels_and_docs_exist(train_data, calibration_train_ids, calibration_dev_ids):
    for query_id in calibration_train_ids + calibration_dev_ids:
        assert query_id in train_data["qrels"]
        for doc_id in train_data["qrels"][query_id]:
            assert doc_id in train_data["corpus"]


def test_no_test_split_leakage(test_data, calibration_train_ids, calibration_dev_ids):
    calibration_ids = set(calibration_train_ids) | set(calibration_dev_ids)
    assert not (set(test_data["queries"]) & calibration_ids)


def test_split_only_covers_evaluable_queries(train_data, calibration_train_ids, calibration_dev_ids):
    evaluable = set(evaluable_query_ids(train_data["qrels"]))
    covered = set(calibration_train_ids) | set(calibration_dev_ids)
    assert covered == evaluable


def test_load_or_build_rejects_partial_split_state(train_data, tmp_path):
    write_split_file(tmp_path / "calibration_train.txt", ["1", "2"])
    # calibration_dev.txt intentionally left missing.
    with pytest.raises(FileNotFoundError):
        load_or_build_calibration_splits(train_data, splits_dir=tmp_path)
