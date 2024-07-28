#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""_summary_ of project"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.config_handlers import TissueConfig
from omics_graph_learning.data_splitter import GeneTrainTestSplitter
from omics_graph_learning.gene_filter import TPMFilter
from omics_graph_learning.target_assembler import TargetAssembler
from omics_graph_learning.target_consolidator import (
    TrainingTargetConsolidator,
)  # Assuming this is the name of the file
import pytest


@pytest.fixture
def mock_experiment_config() -> Mock:
    """Create a mock ExperimentConfig object."""
    config = Mock(spec=ExperimentConfig)
    config.sample_config_dir = Path("/mock/sample/config")
    config.graph_dir = Path("/mock/graph")
    config.working_directory = Path("/mock/working")
    config.expression_all_matrix = Path("/mock/expression_all_matrix.csv")
    config.interaction_types = ["rbp_network"]
    config.interaction_dir = Path("/mock/interaction")
    config.experiment_name = "mock_experiment"
    return config


@pytest.fixture
def mock_tissue_config() -> Mock:
    """Create a mock TissueConfig object."""
    config = Mock(spec=TissueConfig)
    config.resources = {
        "tissue": "mock_tissue",
        "rna": Path("/mock/rna_seq.csv"),
        "tpm": Path("/mock/tpm.csv"),
    }
    return config


@pytest.fixture
def consolidator(
    mock_experiment_config: Mock, mock_tissue_config: Mock
) -> TrainingTargetConsolidator:
    """Create a TrainingTargetConsolidator instance for testing."""
    return TrainingTargetConsolidator(
        experiment_config=mock_experiment_config,
        tissue_config=mock_tissue_config,
        tpm_filter=1.0,
        percent_of_samples_filter=0.5,
        filter_mode="within",
        split_name="test_split",
        target="rna_seq",
    )


@pytest.fixture
def mock_filtered_genes() -> List[str]:
    """Create a list of mock filtered genes."""
    return ["gene1", "gene2", "gene3"]


@pytest.fixture
def mock_split() -> Dict[str, List[str]]:
    """Create a mock split dictionary."""
    return {
        "train": ["gene1_mock_tissue", "gene2_mock_tissue"],
        "test": ["gene3_mock_tissue"],
        "val": [],
    }


@pytest.fixture
def mock_targets() -> Dict[str, Dict[str, np.ndarray]]:
    """Create mock targets."""
    return {
        "gene1_mock_tissue": {"target": np.array([1.0, 2.0])},
        "gene2_mock_tissue": {"target": np.array([3.0, 4.0])},
        "gene3_mock_tissue": {"target": np.array([5.0, 6.0])},
    }


def test_init(consolidator: TrainingTargetConsolidator) -> None:
    """Test the initialization of TrainingTargetConsolidator."""
    assert consolidator.tissue == "mock_tissue"
    assert consolidator.split_name == "test_split"
    assert consolidator.target == "rna_seq"
    assert isinstance(consolidator.split_path, Path)


@patch("training_target_consolidator.dir_check_make")
def test_prepare_split_directories(
    mock_dir_check_make: Mock, consolidator: TrainingTargetConsolidator
) -> None:
    """Test the _prepare_split_directories method."""
    split_path = consolidator._prepare_split_directories()
    assert split_path == consolidator.graph_dir / consolidator.split_name
    mock_dir_check_make.assert_called_once_with(split_path)


@patch("training_target_consolidator._save_pickle")
def test_save_splits(
    mock_save_pickle: Mock,
    consolidator: TrainingTargetConsolidator,
    mock_split: Dict[str, List[str]],
) -> None:
    """Test the _save_splits method."""
    consolidator._save_splits(mock_split)
    expected_path = (
        consolidator.split_path / f"training_split_{consolidator.tissue}.pkl"
    )
    mock_save_pickle.assert_called_once_with(mock_split, expected_path)


@patch("training_target_consolidator._save_pickle")
def test_save_targets(
    mock_save_pickle: Mock,
    consolidator: TrainingTargetConsolidator,
    mock_targets: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Test the _save_targets method."""
    consolidator._save_targets(mock_targets)
    expected_path = (
        consolidator.split_path / f"training_targets_{consolidator.tissue}.pkl"
    )
    mock_save_pickle.assert_called_once_with(mock_targets, expected_path)

    consolidator._save_targets(mock_targets, scaled=True)
    expected_path = (
        consolidator.split_path / f"training_targets_{consolidator.tissue}_scaled.pkl"
    )
    mock_save_pickle.assert_called_with(mock_targets, expected_path)


@patch.object(TPMFilter, "filtered_genes_from_encode_rna_data")
@patch.object(TPMFilter, "filter_genes")
def test_filter_genes(
    mock_filter_genes: Mock,
    mock_filtered_genes_from_encode: Mock,
    consolidator: TrainingTargetConsolidator,
    mock_filtered_genes: List[str],
) -> None:
    """Test the filter_genes method."""
    mock_filtered_genes_from_encode.return_value = mock_filtered_genes
    mock_filter_genes.return_value = mock_filtered_genes

    # Test RNA-seq target
    genes = consolidator.filter_genes()
    assert genes == [f"{gene}_{consolidator.tissue}" for gene in mock_filtered_genes]
    mock_filtered_genes_from_encode.assert_called_once()

    # Test non-RNA-seq target
    consolidator.target = "other"
    consolidator.filter_mode = "within"
    genes = consolidator.filter_genes()
    assert genes == [f"{gene}_{consolidator.tissue}" for gene in mock_filtered_genes]
    mock_filter_genes.assert_called_once()


def test_remove_active_rbp_genes(
    consolidator: TrainingTargetConsolidator, mock_filtered_genes: List[str]
) -> None:
    """Test the remove_active_rbp_genes method."""
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '["gene2"]'
        with patch("pickle.load", return_value=["gene2"]):
            result = consolidator.remove_active_rbp_genes(mock_filtered_genes)
            assert result == ["gene1", "gene3"]


@patch.object(TargetAssembler, "assemble_rna_targets")
@patch.object(TargetAssembler, "assemble_matrix_targets")
def test_assemble_targets(
    mock_assemble_matrix: Mock,
    mock_assemble_rna: Mock,
    consolidator: TrainingTargetConsolidator,
    mock_targets: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Test the assemble_targets method."""
    mock_assembler = Mock(spec=TargetAssembler)
    mock_assemble_rna.return_value = mock_targets
    mock_assemble_matrix.return_value = mock_targets

    # Test RNA-seq target
    result = consolidator.assemble_targets(mock_assembler)
    assert result == mock_targets
    mock_assemble_rna.assert_called_once()

    # Test non-RNA-seq target
    consolidator.target = "other"
    result = consolidator.assemble_targets(mock_assembler)
    assert result == mock_targets
    mock_assemble_matrix.assert_called_once()


@patch.object(TrainingTargetConsolidator, "filter_genes")
@patch.object(TrainingTargetConsolidator, "remove_active_rbp_genes")
@patch.object(GeneTrainTestSplitter, "train_test_val_split")
@patch.object(TrainingTargetConsolidator, "assemble_targets")
@patch.object(TargetAssembler, "scale_targets")
@patch.object(TrainingTargetConsolidator, "_save_splits")
@patch.object(TrainingTargetConsolidator, "_save_targets")
def test_consolidate_training_targets(
    mock_save_targets: Mock,
    mock_save_splits: Mock,
    mock_scale_targets: Mock,
    mock_assemble_targets: Mock,
    mock_train_test_val_split: Mock,
    mock_remove_active_rbp_genes: Mock,
    mock_filter_genes: Mock,
    consolidator: TrainingTargetConsolidator,
    mock_filtered_genes: List[str],
    mock_split: Dict[str, List[str]],
    mock_targets: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Test the consolidate_training_targets method."""
    mock_filter_genes.return_value = mock_filtered_genes
    mock_remove_active_rbp_genes.return_value = mock_filtered_genes
    mock_train_test_val_split.return_value = mock_split
    mock_assemble_targets.return_value = mock_targets
    mock_scale_targets.return_value = mock_targets

    result = consolidator.consolidate_training_targets()

    assert result == mock_filtered_genes
    mock_filter_genes.assert_called_once()
    mock_remove_active_rbp_genes.assert_called_once()
    mock_train_test_val_split.assert_called_once()
    mock_assemble_targets.assert_called_once()
    mock_scale_targets.assert_called_once()
    assert mock_save_splits.call_count == 1
    assert mock_save_targets.call_count == 2


def test_append_tissue_to_genes(
    consolidator: TrainingTargetConsolidator, mock_filtered_genes: List[str]
) -> None:
    """Test the _append_tissue_to_genes static method."""
    result = consolidator._append_tissue_to_genes(mock_filtered_genes, "mock_tissue")
    expected = [f"{gene}_mock_tissue" for gene in mock_filtered_genes]
    assert result == expected
