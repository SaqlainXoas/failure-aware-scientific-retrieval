from retrieval.cache import cache_key, corpus_fingerprint, load_or_compute


def test_cache_key_deterministic():
    assert cache_key("bm25", dataset="beir/scifact", version="v1") == cache_key(
        "bm25", dataset="beir/scifact", version="v1"
    )


def test_cache_key_sensitive_to_namespace():
    assert cache_key("bm25", dataset="x") != cache_key("dense", dataset="x")


def test_cache_key_sensitive_to_each_field():
    base = cache_key("bm25", dataset="beir/scifact", model="a", version="v1")
    assert base != cache_key("bm25", dataset="beir/scifact", model="b", version="v1")
    assert base != cache_key("bm25", dataset="other", model="a", version="v1")
    assert base != cache_key("bm25", dataset="beir/scifact", model="a", version="v2")


def test_corpus_fingerprint_sensitive_to_content():
    a = corpus_fingerprint({"d1": "hello"})
    b = corpus_fingerprint({"d1": "world"})
    assert a != b
    assert corpus_fingerprint({"d1": "hello"}) == a


def test_load_or_compute_cold_cache_computes_and_saves(tmp_path):
    path = tmp_path / "sub" / "value.txt"
    calls = []

    def compute_fn():
        calls.append(1)
        return "computed"

    value, was_cached = load_or_compute(
        path,
        compute_fn,
        force=False,
        loader=lambda p: p.read_text(),
        saver=lambda p, v: p.write_text(v),
    )

    assert value == "computed"
    assert was_cached is False
    assert len(calls) == 1
    assert path.read_text() == "computed"


def test_load_or_compute_warm_cache_skips_compute(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("cached")
    calls = []

    value, was_cached = load_or_compute(
        path,
        lambda: calls.append(1) or "recomputed",
        force=False,
        loader=lambda p: p.read_text(),
        saver=lambda p, v: p.write_text(v),
    )

    assert value == "cached"
    assert was_cached is True
    assert len(calls) == 0


def test_load_or_compute_force_recomputes_even_when_warm(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("cached")
    calls = []

    value, was_cached = load_or_compute(
        path,
        lambda: calls.append(1) or "recomputed",
        force=True,
        loader=lambda p: p.read_text(),
        saver=lambda p, v: p.write_text(v),
    )

    assert value == "recomputed"
    assert was_cached is False
    assert len(calls) == 1
    assert path.read_text() == "recomputed"
