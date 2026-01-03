#!/usr/bin/env python3
"""
Entry point for executing the NYC Taxi Trip Duration ML pipeline.

This script:
 - Loads a config dictionary
 - Instantiates TaxiPipeline
 - Runs training, model persistence, and batch inference
"""

import os
from src.ml_project.pipelines.pipeline import TaxiPipeline  # update if pipeline file path differs

def main():
    # Example configuration dictionary; update paths as needed
    config = {
    "paths": {
        "train_csv": "src/ml_project/data/train.csv",
        "test_csv": "src/ml_project/data/test.csv",
        "artifact_dir": "artifacts/models",
        "output_dir": "artifacts/predictions"
    },
    "train": {
        "metric": "rmse",
        "test_size": 0.2,
        "seed": 42
    }
}


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
