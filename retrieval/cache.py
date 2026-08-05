"""Disk cache for BM25 index / dense embeddings / run outputs, keyed by content fingerprint."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

CACHE_ROOT = ".cache"


def cache_key(namespace: str, **fields: Any) -> str:
    """Stable hash of sorted (namespace, **fields) for use as a cache directory name."""
    payload = json.dumps({"namespace": namespace, **fields}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(namespace: str, key: str, cache_dir: str | Path = CACHE_ROOT) -> Path:
    return Path(cache_dir) / namespace / key


def corpus_fingerprint(corpus: dict[str, str]) -> str:
    """Sha256 over sorted (doc_id, text) pairs — cheap at SciFact's scale (a few thousand docs)."""
    payload = json.dumps(sorted(corpus.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_or_compute(
    path: Path,
    compute_fn: Callable[[], Any],
    force: bool,
    loader: Callable[[Path], Any],
    saver: Callable[[Path, Any], None],
) -> tuple[Any, bool]:
    """Generic cache get-or-set: force or missing path -> compute+save; else load. Returns (value, was_cached)."""
    if not force and path.exists():
        return loader(path), True
    value = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(path, value)
    return value, False
