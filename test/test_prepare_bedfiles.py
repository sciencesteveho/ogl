import os
from unittest.mock import MagicMock
from unittest.mock import patch

from prepare_bedfiles import GenomeDataPreprocessor
import pytest

# Constants for tests
TEST_WORKING_DIR = "/test/working/dir"
TEST_EXPERIMENT_NAME = "test_experiment"
TEST_INTERACTION_TYPES = ["circuits", "ppis", "mirna"]
TEST_NODES = ["cpgislands", "ctcfccre", "tss"]
TEST_PARAMS = {
    "dirs": {
        "root_dir": "/test/root/dir",
        "circuit_dir": "/test/circuit/dir"
    },
    "interaction": {
        "circuits": "test_circuits.bed",
        "ppis": "test_ppis.bed",
        "tf_marker": "test_tf_marker.bed",
        "tf_binding": "test_tf_binding.bed",
        "mirdip": "test_mirdip.bed",
        "mirnatargets": "test_mirnatargets.bed"
    },
    "methylation": {
        "cpg": "test_cpg.bed",
        "cpg_liftover": False,
        "cpg_filetype": "ENCODE"
    },
    "resources": {
        "tissue": "test_tissue",
        "tf_motifs": "test_tf_motifs.bed",
        "liftover": "test_liftover",
        "liftover_chain": "test_liftover_chain"
    },
    "local": {
        "cpgislands": "test_cpgislands.bed",
        "ctcfccre": "test_ctcfccre.bed",
        "tss": "test_tss.bed"
    },
    "features": {
        "histone": "test_histone.bed"
    },
    "tissue_specific_nodes": {
        "crms": "test_crms.bed",
        "tads": "test_tads.bed",
        "super_enhancer": "test_super_enhancer.bed",
        "tf_binding": "test_tf_binding.bed"
    }
}

# Helper function to create a preprocessor instance
def create_preprocessor():
    return GenomeDataPreprocessor(
        TEST_EXPERIMENT_NAME,
        TEST_INTERACTION_TYPES,
        TEST_NODES,
        TEST_WORKING_DIR,
        TEST_PARAMS
    )

# Parametrized test for the happy path
@pytest.mark.parametrize("experiment_name, interaction_types, nodes, working_directory, params, expected", [
    # Test ID: #1
    (
        TEST_EXPERIMENT_NAME,
        TEST_INTERACTION_TYPES,
        TEST_NODES,
        TEST_WORKING_DIR,
        TEST_PARAMS,
        True  # Expected result (success)
    ),
    # Add more test cases here for different combinations of inputs
])
def test_happy_path(experiment_name, interaction_types, nodes, working_directory, params, expected):
    # Arrange
    preprocessor = GenomeDataPreprocessor(experiment_name, interaction_types, nodes, working_directory, params)
    
    # Act
    # Assuming the prepare_data_files method is the main pipeline function to be tested
    with patch('utils.dir_check_make') as mock_dir_check_make, \
         patch('utils.check_and_symlink') as mock_check_and_symlink, \
         patch('subprocess.run') as mock_run_cmd:
        preprocessor.prepare_data_files()
    
    # Assert
    # Verify that the directory check and make function is called
    assert mock_dir_check_make.called
    # Verify that the check and symlink function is called
    assert mock_check_and_symlink.called
    # Verify that the run command function is called
    assert mock_run_cmd.called

# Parametrized test for edge cases
@pytest.mark.parametrize("experiment_name, interaction_types, nodes, working_directory, params, expected_exception", [
    # Test ID: #2
    (
        TEST_EXPERIMENT_NAME,
        [],  # Empty interaction types
        TEST_NODES,
        TEST_WORKING_DIR,
        TEST_PARAMS,
        None  # No exception expected
    ),
    # Add more edge cases here
])
def test_edge_cases(experiment_name, interaction_types, nodes, working_directory, params, expected_exception):
    # Arrange
    preprocessor = GenomeDataPreprocessor(experiment_name, interaction_types, nodes, working_directory, params)
    
    # Act & Assert
    if expected_exception:
        with pytest.raises(expected_exception):
            preprocessor.prepare_data_files()
    else:
        preprocessor.prepare_data_files()

# Parametrized test for error cases
@pytest.mark.parametrize("experiment_name, interaction_types, nodes, working_directory, params, expected_exception", [
    # Test ID: #3
    (
        TEST_EXPERIMENT_NAME,
        TEST_INTERACTION_TYPES,
        TEST_NODES,
        None,  # None working directory
        TEST_PARAMS,
        TypeError  # Expected exception
    ),
    # Add more error cases here
])
def test_error_cases(experiment_name, interaction_types, nodes, working_directory, params, expected_exception):
    # Act & Assert
    with pytest.raises(expected_exception):
        GenomeDataPreprocessor(experiment_name, interaction_types, nodes, working_directory, params)
