#!/usr/bin/env python
"""Lightweight CLI that runs the class-based pipeline in `src`.

Usage examples:
  train:        docker-compose run --rm app pipeline train
  run all:      docker-compose run --rm app pipeline run
  customize:    docker-compose run --rm app pipeline --config config/config.yaml run
"""
import argparse
from pathlib import Path
import sys
from omegaconf import OmegaConf

# Ensure project root is on sys.path so imports from `src` work when running as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ml_project.pipelines.pipeline import TaxiPipeline


def load_config(path: str):
    cfg = OmegaConf.load(path)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run the class-basedPipeline from `src`.")
    parser.add_argument("action", nargs="?", default="run", choices=["run", "train"],
                        help="Action to perform (train or run which trains + runs inference).")
    parser.add_argument("--config", "-c", default="config/config.yaml",
                        help="Path to the config YAML file (default: config/config.yaml)")

    args = parser.parse_args()

    cfg = load_config(args.config)

    pipeline = TaxiPipeline(cfg)

    if args.action in ("run", "train"):
        pipeline.run()


if __name__ == "__main__":
    main()
