import subprocess
import sys

from retrieval.data import load_config


def test_package_imports():
    import retrieval  # noqa: F401


def test_load_config_roundtrips(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline: bm25\nseed: 42\n")

    config = load_config(config_path)

    assert config == {"pipeline": "bm25", "seed": 42}


def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "retrieval.run", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
