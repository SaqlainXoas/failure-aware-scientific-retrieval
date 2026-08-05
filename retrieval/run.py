"""Thin entrypoint: retrieve -> rerank -> evaluate for a single pipeline config."""

import argparse
import logging

from retrieval.data import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a retrieval pipeline end to end (retrieve, rerank, evaluate)."
    )
    parser.add_argument("--config", required=True, help="Path to a pipeline config YAML.")
    parser.add_argument("--split", default="calibration-dev", help="Query split to run on.")
    parser.add_argument("--force", action="store_true", help="Bypass cache and recompute.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    logger.info("Loaded config from %s for split %s: %s", args.config, args.split, config)
    raise NotImplementedError("Pipeline execution is not implemented yet (Phase 0 stub).")


if __name__ == "__main__":
    main()
