#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from typing import List

import pytest

# ... (keep all your existing imports)


def run_tests() -> bool:
    """
    Run all pytest tests and return True if all tests pass, False otherwise.
    """
    test_files: List[str] = ["test_models.py", "test_training_target_consolidator.py"]

    exit_code = pytest.main(test_files)

    return exit_code == 0  # pytest.ExitCode.OK is 0


def main() -> None:
    """
    Run OGL pipeline, from data parsing to graph construction to GNN
    training with checks to avoid redundant computation.
    First runs all unit tests and only proceeds if they all pass.
    """
    print("Running unit tests...")
    tests_passed = run_tests()

    if not tests_passed:
        print("Unit tests failed. Aborting pipeline execution.")
        sys.exit(1)

    print("All unit tests passed. Proceeding with pipeline execution.")

    args = parse_pipeline_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    pipe_runner = PipelineRunner(config=experiment_config, args=args)
    pipe_runner.run_pipeline()


if __name__ == "__main__":
    main()
