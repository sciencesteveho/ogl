#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handles stored and shared data from omics graph learning configs."""

from dataclasses import dataclass
from dataclasses import field
import pathlib
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore

from utils import parse_yaml


@dataclass
class ExperimentConfig:

    # Params
    baseloops: str
    experiment_name: str
    feat_window: int
    gene_gene: bool
    interaction_types: Optional[List[str]]
    loop_resolution: float
    nodes: Optional[List[str]]
    regulatory: str
    tissues: List[str]

    # Files for training split
    average_activity_df: pathlib.Path
    config_dir: pathlib.Path
    expression_median_across_all: pathlib.Path
    expression_median_matrix: pathlib.Path
    expression_all_matrix: pathlib.Path
    gencode_gtf: pathlib.Path
    protein_abundance_matrix: pathlib.Path
    protein_abundance_medians: pathlib.Path
    test_chrs: List[str]
    val_chrs: List[str]
    working_directory: pathlib.Path
    node_types: List[str] = field(
        default_factory=lambda: ["dyadic", "enhancers", "gencode", "promoters"]
    )

    # Directories
    root_dir: pathlib.Path = field(default_factory=pathlib.Path)
    raw_data_dir: pathlib.Path = field(default_factory=pathlib.Path)
    shared_data_dir: pathlib.Path = field(default_factory=pathlib.Path)
    interaction_dir: pathlib.Path = field(default_factory=pathlib.Path)
    local_data_dir: pathlib.Path = field(default_factory=pathlib.Path)
    reference_dir: pathlib.Path = field(default_factory=pathlib.Path)
    regulatory_dir: pathlib.Path = field(default_factory=pathlib.Path)
    expression_dir: pathlib.Path = field(default_factory=pathlib.Path)
    tpm_dir: pathlib.Path = field(default_factory=pathlib.Path)
    matrix_dir: pathlib.Path = field(default_factory=pathlib.Path)
    baseloop_dir: pathlib.Path = field(default_factory=pathlib.Path)

    def _resolve_directories(self, directories: Dict[str, str]) -> None:
        """Retrieves the root dir from the config and sets up all other required
        directories as resolved absolute paths."""
        self.root_dir = pathlib.Path(directories["root_dir"]).resolve()
        # Use setattr to update dataclass fields based on the dictionary entries
        for key, relative_path in directories.items():
            if key != "root_dir":
                absolute_path = self.root_dir / relative_path
                setattr(self, f"{key}", absolute_path.resolve())

    @classmethod
    def from_yaml(cls, yaml_file: pathlib.Path) -> "ExperimentConfig":
        """Load the configuration from a yaml and set all configs."""
        params = cls._load_yaml(yaml_file)

        cls._update_with_training_targets(params)
        cls._ensure_lists(
            params, ["interaction_types", "nodes", "test_chrs", "val_chrs", "tissues"]
        )

        return cls(**params)

    @staticmethod
    def _load_yaml(yaml_file: pathlib.Path) -> Dict[str, Any]:
        with open(yaml_file, "r") as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def _update_with_training_targets(params: dict) -> None:
        training_targets = params.pop("training_targets", {})
        for key, path in training_targets.items():
            params[key] = pathlib.Path(path)

    @staticmethod
    def _ensure_lists(params: dict, keys: List[str]) -> None:
        """Ensure the given keys have a list value."""
        for key in keys:
            params[key] = params.get(key, [])


@dataclass
class TissueConfig:
    # Nested configuration dictionaries
    dirs: Dict[str, pathlib.Path]
    features: Dict[str, pathlib.Path]
    interaction: Dict[str, pathlib.Path]
    local: Dict[str, pathlib.Path]
    methylation: Dict[str, Union[pathlib.Path, str, bool]]
    references: Dict[str, pathlib.Path]
    resources: Dict[str, Union[pathlib.Path, str]]
    tissue_specific_nodes: Dict[str, pathlib.Path]

    @staticmethod
    def _load_yaml(yaml_file: pathlib.Path) -> dict:
        with open(yaml_file, "r") as stream:
            return yaml.safe_load(stream)

    @classmethod
    def _process_directory_paths(cls, params: Dict[str, Any]) -> None:
        if "dirs" in params:
            cls._convert_str_to_path(params["dirs"])

    @classmethod
    def _process_feature_paths(cls, params: Dict[str, Any]) -> None:
        if "features" in params:
            cls._convert_str_to_path(params["features"])

    # Add similar methods for other configuration sections:
    # _process_interaction_paths
    # _process_local_paths
    # ... (and so on for each dictionary-type attribute)

    @staticmethod
    def _convert_str_to_path(config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if isinstance(value, str):
                config_dict[key] = pathlib.Path(value)

    @classmethod
    def from_yaml(cls, yaml_file: pathlib.Path) -> "TissueConfig":
        params = cls._load_yaml(yaml_file)

        cls._process_directory_paths(params)
        cls._process_feature_paths(params)
        # Call the corresponding processing methods for each section:
        # cls._process_interaction_paths(params)
        # cls._process_local_paths(params)
        # ... (and so on for each dictionary-type attribute)

        return cls(**params)
