#!/usr/bin/env python3
"""
Entry point for executing the NYC Taxi Trip Duration ML pipeline.

This script:
 - Loads a config dictionary from a YAML file
 - Instantiates TaxiPipeline
 - Runs training, model persistence, and batch inference
"""

import os
import yaml
from src.ml_project.pipelines.pipeline import TaxiPipeline  # update path if needed

def load_config(yaml_path: str) -> dict:
    """Load YAML configuration file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Path to your YAML config
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Ensure directories exist
    os.makedirs(config["paths"]["artifact_dir"], exist_ok=True)
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)

    # Initialize pipeline
    pipeline = TaxiPipeline(config)

    # Execute full run
    model, best_model_name, output_df = pipeline.run()

    print("Training complete.")
    print(f"Best model selected: {best_model_name}")
    print(f"Inference output size: {len(output_df)} rows")
    print(f"Artifacts saved to: {config['paths']['artifact_dir']}")
    print(f"Predictions saved to: {config['paths']['output_dir']}")

if __name__ == "__main__":
    main()
