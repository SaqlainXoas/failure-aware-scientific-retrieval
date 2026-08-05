"""SciFact loading, splits, config loading, and device selection."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SEED = 42
DEV_FRACTION = 0.2
SPLITS_DIR = "splits"
STATS_PATH = "results/tables/dataset_stats.json"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_scifact_split(split: str) -> dict[str, Any]:
    """Loads one BEIR SciFact split (train/test) into corpus/queries/qrels dicts."""
    import ir_datasets

    ds = ir_datasets.load(f"beir/scifact/{split}")
    corpus = {d.doc_id: f"{d.title}\n{d.text}" for d in ds.docs_iter()}
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def load_scifact() -> dict[str, Any]:
    return {"train": load_scifact_split("train"), "test": load_scifact_split("test")}


def evaluable_query_ids(qrels: dict[str, dict[str, int]]) -> list[str]:
    """Query IDs with at least one positive (grade > 0) qrel — only these carry a defined evidence label."""
    return sorted(
        query_id
        for query_id, doc_relevances in qrels.items()
        if any(relevance > 0 for relevance in doc_relevances.values())
    )


def generate_calibration_split(
    evaluable_ids: list[str], seed: int = SEED, dev_frac: float = DEV_FRACTION
) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 split: sort ids (iteration order isn't guaranteed stable), then seeded shuffle."""
    ids = sorted(evaluable_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_dev = round(len(ids) * dev_frac)
    dev_ids, train_ids = ids[:n_dev], ids[n_dev:]
    return sorted(train_ids), sorted(dev_ids)


def write_split_file(path: str | Path, query_ids: list[str]) -> None:
    Path(path).write_text("\n".join(query_ids) + "\n")


def read_split_file(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def validate_calibration_splits(
    train_ids: list[str], dev_ids: list[str], evaluable_ids: list[str]
) -> None:
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("calibration_train contains duplicate query IDs")
    if len(dev_ids) != len(set(dev_ids)):
        raise ValueError("calibration_dev contains duplicate query IDs")
    if set(train_ids) & set(dev_ids):
        raise ValueError("calibration_train and calibration_dev are not disjoint")
    if set(train_ids) | set(dev_ids) != set(evaluable_ids):
        raise ValueError(
            "calibration_train ∪ calibration_dev does not equal the evaluable training query IDs"
        )


def build_calibration_splits(
    train_data: dict[str, Any], splits_dir: str | Path = SPLITS_DIR, seed: int = SEED
) -> tuple[list[str], list[str]]:
    ids = evaluable_query_ids(train_data["qrels"])
    train_ids, dev_ids = generate_calibration_split(ids, seed=seed)
    validate_calibration_splits(train_ids, dev_ids, ids)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_split_file(splits_dir / "calibration_train.txt", train_ids)
    write_split_file(splits_dir / "calibration_dev.txt", dev_ids)
    return train_ids, dev_ids


def load_or_build_calibration_splits(
    train_data: dict[str, Any], splits_dir: str | Path = SPLITS_DIR, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Neither file exists -> generate both; both exist -> load+validate; exactly one -> error (never silently patched)."""
    splits_dir = Path(splits_dir)
    train_path = splits_dir / "calibration_train.txt"
    dev_path = splits_dir / "calibration_dev.txt"
    train_exists, dev_exists = train_path.exists(), dev_path.exists()

    if not train_exists and not dev_exists:
        logger.info("No calibration split files found; generating from seed=%s", seed)
        return build_calibration_splits(train_data, splits_dir=splits_dir, seed=seed)

    if train_exists != dev_exists:
        missing = dev_path if train_exists else train_path
        raise FileNotFoundError(
            f"Partial calibration split state: {missing} is missing while its counterpart exists. "
            "Refusing to silently regenerate or continue — restore or delete both files."
        )

    train_ids = read_split_file(train_path)
    dev_ids = read_split_file(dev_path)
    validate_calibration_splits(train_ids, dev_ids, evaluable_query_ids(train_data["qrels"]))
    return train_ids, dev_ids


def compute_dataset_stats(
    data: dict[str, Any], calib_train_ids: list[str], calib_dev_ids: list[str]
) -> dict[str, Any]:
    def split_counts(split_data: dict[str, Any]) -> dict[str, int]:
        return {
            "n_queries": len(split_data["queries"]),
            "n_docs": len(split_data["corpus"]),
            "n_qrels": sum(len(v) for v in split_data["qrels"].values()),
        }

    train_qrels = data["train"]["qrels"]
    n_evaluable = len(evaluable_query_ids(train_qrels))
    n_excluded = len(data["train"]["queries"]) - n_evaluable

    return {
        "seed": SEED,
        "train": split_counts(data["train"]),
        "test": split_counts(data["test"]),
        "train_evaluable_queries": n_evaluable,
        "train_excluded_queries": n_excluded,
        "calibration_train": {"n_queries": len(calib_train_ids)},
        "calibration_dev": {"n_queries": len(calib_dev_ids)},
    }


def save_stats(stats: dict[str, Any], path: str | Path = STATS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    setup_logging()
    data = load_scifact()
    train_ids, dev_ids = load_or_build_calibration_splits(data["train"])
    stats = compute_dataset_stats(data, train_ids, dev_ids)
    logger.info("Dataset statistics: %s", json.dumps(stats, indent=2))
    save_stats(stats)
    logger.info("Saved dataset statistics to %s", STATS_PATH)


if __name__ == "__main__":
    main()
