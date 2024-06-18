# sourcery skip: snake-case-variable-declarations
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handle stored and shared data from omics graph learning configs."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml  # type: ignore


def load_yaml(yaml_file: Path) -> Dict[str, Any]:
    """Load a YAML file and return the contents as a dictionary."""
    with open(yaml_file, "r") as stream:
        return yaml.safe_load(stream)


@dataclass
class ExperimentConfig:
    """Class representing the configuration for an experiment.

    Arguments:
        yaml_file: Path to the YAML file containing the experiment
        configuration.

    Methods
    --------
    from_yaml(yaml_file: Path) -> ExperimentConfig:
        Loads the configuration YAML, resolves directories, updates param
        values, and ensures required lists exist as lists.

    Returns:
        ExperimentConfig: An instance of ExperimentConfig initialized with the
        configuration loaded from the YAML file.

    Examples:
    --------
    >>> from config_handlers import ExperimentConfig
    >>> config = ExperimentConfig.from_yaml("k562_combined_contacts.yaml")
    """

    VALID_LOG_TRANSFORMS = {"log2", "log1p", "log10"}

    # Params
    attribute_references: Dict[str, str]
    baseloops: str
    blacklist: str
    chromfile: str
    config_dir: Path
    differentiate_tf: bool
    experiment_name: str
    fasta: str
    feat_window: int
    gene_gene: bool
    interaction_types: List[str]
    liftover: str
    liftover_chain: str
    log_transform: str
    loop_resolution: float
    nodes: List[str]
    regulatory_schema: str
    root_dir: Path
    tissues: List[str]

    # Params from directories dict
    baseloop_dir: Path
    expression_dir: Path
    interaction_dir: Path
    local_data_dir: Path
    matrix_dir: Path
    raw_data_dir: Path
    reference_dir: Path
    regulatory_dir: Path
    shared_data_dir: Path
    target_dir: Path
    tpm_dir: Path

    # Params from training_targets dict
    average_activity_df: str
    expression_median_across_all: str
    expression_median_matrix: str
    expression_all_matrix: str
    gencode_gtf: str
    protein_abundance_matrix: str
    protein_abundance_medians: str
    test_chrs: List[str]
    val_chrs: List[str]

    # Params from instantiation
    working_directory: Path
    graph_dir: Path

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "ExperimentConfig":
        """Load the configuration from a yaml and set all configs."""

        # load the yaml file
        params = load_yaml(yaml_file)

        # update the params with the params hidden in dictionaries
        cls._add_training_target_params(params=params, key="training_targets")
        cls._add_attribute_references(params=params, key="attribute_references")
        cls._resolve_directories(params=params, root_dir=params["root_dir"])
        cls._ensure_lists(
            params=params,
            keys=["interaction_types", "nodes", "test_chrs", "val_chrs", "tissues"],
        )

        # update str dirs to Path objects
        params["root_dir"] = Path(params["root_dir"])
        params["config_dir"] = Path(params["config_dir"])

        # add the default node types
        cls._update_node_types(params=params)

        # add working dir and graph_dir
        params["working_directory"] = params["root_dir"] / params["experiment_name"]
        params["graph_dir"] = params["working_directory"] / "graphs"

        # validate log_transform
        params["log_transform"] = cls.validate_log_transform(params["log_transform"])

        return cls(**params)

    @classmethod
    def validate_log_transform(cls, value: str) -> str:
        """Check if the log_transform value is valid"""
        if value not in cls.VALID_LOG_TRANSFORMS:
            raise ValueError(
                f"log_transform value must be one of {cls.VALID_LOG_TRANSFORMS}, got '{value}'."
            )
        return value

    @staticmethod
    def _add_training_target_params(params: dict, key: str) -> None:
        training_targets = params.pop(key, {})
        for key, value in training_targets.items():
            params[key] = value

    @staticmethod
    def _add_attribute_references(params: dict, key: str) -> None:
        """Add attribute references by prepending the reference directory to the
        file paths. Additionally, add the regulatory elements attr based on the
        regulatory schema."""
        ref_dir = params[key].pop("ref_dir")
        for subkey, value in params[key].items():
            params[key][subkey] = f"{ref_dir}/{value}"

        # add regulatory ref
        reg_attr_file = (
            f"regulatory_elements_{params['regulatory_schema']}_node_attr.bed"
        )
        params[key]["regulatory_elements"] = f"{ref_dir}/{reg_attr_file}"

    @staticmethod
    def _resolve_directories(params: dict, root_dir: str) -> None:
        """Sets up all required directories as resolved absolute paths, based on a given root directory."""
        root_path = Path(root_dir).resolve()
        directories = params.pop("derived_directories")
        for key, relative_path in directories.items():
            absolute_path = root_path / relative_path
            params[key] = absolute_path.resolve()

    @staticmethod
    def _ensure_lists(params: dict, keys: List[str]) -> None:
        """Ensure the given keys have a list value."""
        for key in keys:
            if params.get(key) is None:
                params[key] = []

    @staticmethod
    def _update_node_types(params: Dict[str, Any]) -> None:
        """Update the node_types by adding the values from the YAML file to the default node_types."""
        default_node_types = ["dyadic", "enhancers", "gencode", "promoters"]
        yaml_node_types = params.get("nodes", [])
        params["nodes"] = list(set(yaml_node_types + default_node_types))


@dataclass
class TissueConfig:
    """Class representing the configuration for an experiment.

    Arguments:
        Lorem: Ipsum

    Methods
    --------

    Returns:
        Lorem: Ipsum

    Examples:
    --------
    >>> from config_handlers import TissueConfig
    >>> config = TissueConfig.from_yaml("left_ventricle.yaml")
    """

    features: Dict[str, str]
    interaction: Dict[str, str]
    local: Dict[str, str]
    methylation: Dict[str, Any]
    resources: Dict[str, str]
    tissue_specific_nodes: Dict[str, str]

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "TissueConfig":
        """Load the configuration from a yaml and set all configs."""

        # load the yaml file
        params = load_yaml(yaml_file)
        return cls(**params)

        # # update the params with the params hidden in dictionaries
        # unpacked_params = TissueConfig.unpack_nested_dict(params)
        # return TissueConfig(**unpacked_params)

    @staticmethod
    def unpack_nested_dict(nested: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """A helper function to unpack a nested dictionary into a flat dictionary."""
        unpacked = {}
        for key, value in nested.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    unpacked[f"{subkey}"] = subvalue
            else:
                unpacked[key] = value
        return unpacked

    # # Params from feature dictionary
    # ATAC: str
    # CTCF: str
    # DNase: str
    # H3K27ac: str
    # H3K27me3: str
    # H3K36me3: str
    # H3K4me1: str
    # H3K4me2: str
    # H3K4me3: str
    # H3K79me2: str
    # H3K9ac: str
    # H3K9me3: str
    # POLR2A: str
    # RAD21: str
    # SMC3: str

    # # params from interaction dictionary
    # gct: str
    # id_lookup: str
    # tf_binding: str
    # tf_marker: str

    # # params from local dictionary
    # cnv: str
    # cpgislands: str
    # ctcfccre: str
    # gencode: str
    # indels: str
    # line: str
    # ltr: str
    # microsatellites: str
    # phastcons: str
    # polyasites: str
    # rbpbindingsites: str
    # recombination: str
    # repg1b: str
    # repg2: str
    # reps1: str
    # reps2: str
    # reps3: str
    # reps4: str
    # rnarepeat: str
    # simplerepeats: str
    # sine: str
    # snp: str
    # tss: str

    # # params from methylation dictionary
    # cpg: str
    # cpg_filetype: str
    # cpg_liftover: bool

    # # params from resources dictionary
    # key_protein_abundance: str
    # key_tpm: str
    # marker_name: str
    # ppi_tissue: str
    # rna: str
    # tissue: str
    # tissue_name: str
    # tpm: str

    # # params from tissue_specific_nodes
    # crms: str
    # super_enhancer: str
    # tads: str
    # tf_footprints: str
