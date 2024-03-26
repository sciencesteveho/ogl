import os
from unittest.mock import MagicMock
from unittest.mock import patch

from local_context_parser import LocalContextParser
import pytest

# Constants for tests
TEST_WORKING_DIR = "/test/working/dir"
TEST_EXPERIMENT_NAME = "test_experiment"
TEST_NODES = ["node1", "node2"]
TEST_BEDFILES = ["bedfile1", "bedfile2"]
TEST_PARAMS = {
    "resources": {
        "tissue": "test_tissue",
        "chromfile": "/test/chromfile",
        "fasta": "/test/fasta",
        "tpm": "/test/tpm.gct",
    },
    "local": {"gencode": "test_gencode"},
    "dirs": {"root_dir": "/test/root/dir"},
}


# Helper function to create a fake bed file
def create_fake_bed_file(path):
    with open(path, "w") as f:
        f.write("chr1\t100\t200\tfeature1\n")


# Parametrized test for the happy path
@pytest.mark.parametrize(
    "experiment_name, nodes, working_directory, bedfiles, params, expected",
    [
        # Test ID: HP-1
        (
            TEST_EXPERIMENT_NAME,
            TEST_NODES,
            TEST_WORKING_DIR,
            TEST_BEDFILES,
            TEST_PARAMS,
            None,
        ),
        # Add more test cases with different combinations of inputs
    ],
    ids=["HP-1"],
)
def test_happy_path(
    experiment_name, nodes, working_directory, bedfiles, params, expected
):
    # Arrange
    create_fake_bed_file(f"{TEST_WORKING_DIR}/{experiment_name}/tpm_filtered_genes.bed")
    create_fake_bed_file(
        f"{TEST_WORKING_DIR}/{experiment_name}/tpm_filtered_gene_regions.bed"
    )
    create_fake_bed_file(
        f"{TEST_WORKING_DIR}/{experiment_name}/local/basenodes_hg38.txt"
    )
    create_fake_bed_file(f"{TEST_PARAMS['resources']['tpm']}")
    os.makedirs(f"{TEST_WORKING_DIR}/{experiment_name}/local", exist_ok=True)

    # Act
    parser = LocalContextParser(
        experiment_name, nodes, working_directory, bedfiles, params
    )

    # Assert
    assert parser.experiment_name == experiment_name
    assert parser.nodes == nodes
    assert parser.working_directory == working_directory
    assert parser.bedfiles == bedfiles
    assert parser.resources == params["resources"]
    assert parser.gencode == params["local"]["gencode"]
    # Add more assertions to cover all lines and branches


# Parametrized test for edge cases
@pytest.mark.parametrize(
    "experiment_name, nodes, working_directory, bedfiles, params, expected_exception",
    [
        # Test ID: EC-1
        (
            TEST_EXPERIMENT_NAME,
            [],
            TEST_WORKING_DIR,
            TEST_BEDFILES,
            TEST_PARAMS,
            ValueError,
        ),
        # Add more test cases for edge cases
    ],
    ids=["EC-1"],
)
def test_edge_cases(
    experiment_name, nodes, working_directory, bedfiles, params, expected_exception
):
    # Arrange
    # (Omitted if all input values are provided via the test parameters)

    # Act & Assert
    with pytest.raises(expected_exception):
        LocalContextParser(experiment_name, nodes, working_directory, bedfiles, params)


# Parametrized test for error cases
@pytest.mark.parametrize(
    "experiment_name, nodes, working_directory, bedfiles, params, expected_exception",
    [
        # Test ID: ER-1
        (
            TEST_EXPERIMENT_NAME,
            TEST_NODES,
            TEST_WORKING_DIR,
            TEST_BEDFILES,
            {},
            KeyError,
        ),
        # Add more test cases for error cases
    ],
    ids=["ER-1"],
)
def test_error_cases(
    experiment_name, nodes, working_directory, bedfiles, params, expected_exception
):
    # Arrange
    # (Omitted if all input values are provided via the test parameters)

    # Act & Assert
    with pytest.raises(expected_exception):
        LocalContextParser(experiment_name, nodes, working_directory, bedfiles, params)


# Cleanup created files and directories after tests
def teardown_module(module):
    os.remove(f"{TEST_WORKING_DIR}/{TEST_EXPERIMENT_NAME}/tpm_filtered_genes.bed")
    os.remove(
        f"{TEST_WORKING_DIR}/{TEST_EXPERIMENT_NAME}/tpm_filtered_gene_regions.bed"
    )
    os.remove(f"{TEST_WORKING_DIR}/{TEST_EXPERIMENT_NAME}/local/basenodes_hg38.txt")
    os.remove(f"{TEST_PARAMS['resources']['tpm']}")
    os.rmdir(f"{TEST_WORKING_DIR}/{TEST_EXPERIMENT_NAME}/local")
    os.rmdir(f"{TEST_WORKING_DIR}/{TEST_EXPERIMENT_NAME}")
    os.rmdir(TEST_WORKING_DIR)
