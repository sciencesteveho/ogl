from unittest.mock import mock_open
from unittest.mock import patch

from edge_parser import EdgeParser
import pybedtools
import pytest
import utils

# Test IDs are structured as follows:
# "test_<method_name>_<scenario>_<expected_outcome>_<ID>"


# Happy path tests
@pytest.mark.parametrize(
    "experiment_name, interaction_types, working_directory, loop_file, params, expected",
    [
        # Test ID: HP-1
        (
            "exp1",
            ["ppis", "mirna"],
            "/work",
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            None,
        ),
        # Test ID: HP-2
        (
            "exp2",
            ["tfbinding"],
            "/work",
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            None,
        ),
        # Test ID: HP-3
        (
            "exp3",
            ["circuits"],
            "/work",
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            None,
        ),
    ],
    ids=["HP-1", "HP-2", "HP-3"],
)
def test_EdgeParser_init_happy_path(
    experiment_name, interaction_types, working_directory, loop_file, params, expected
):
    # Arrange
    # Act
    edge_parser = EdgeParser(
        experiment_name, interaction_types, working_directory, loop_file, params
    )
    # Assert
    assert edge_parser.experiment_name == experiment_name
    assert edge_parser.interaction_types == interaction_types
    assert edge_parser.working_directory == working_directory
    assert edge_parser.loop_file == loop_file


# Edge cases
@pytest.mark.parametrize(
    "experiment_name, interaction_types, working_directory, loop_file, params, expected_exception",
    [
        # Test ID: EC-1
        (
            None,
            ["ppis"],
            "/work",
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            TypeError,
        ),
        # Test ID: EC-2
        (
            "exp1",
            None,
            "/work",
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            TypeError,
        ),
        # Test ID: EC-3
        (
            "exp1",
            ["ppis"],
            None,
            "loops.bed",
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            TypeError,
        ),
        # Test ID: EC-4
        (
            "exp1",
            ["ppis"],
            "/work",
            None,
            {"local": {}, "interaction": {}, "resources": {}, "dirs": {}},
            TypeError,
        ),
    ],
    ids=["EC-1", "EC-2", "EC-3", "EC-4"],
)
def test_EdgeParser_init_edge_cases(
    experiment_name,
    interaction_types,
    working_directory,
    loop_file,
    params,
    expected_exception,
):
    # Arrange
    # Act / Assert
    with pytest.raises(expected_exception):
        EdgeParser(
            experiment_name, interaction_types, working_directory, loop_file, params
        )


# Error cases
@pytest.mark.parametrize(
    "experiment_name, interaction_types, working_directory, loop_file, params, expected_exception",
    [
        # Test ID: ER-1
        (
            "exp1",
            ["ppis"],
            "/work",
            "loops.bed",
            {
                "local": {"gencode": "missing_file.bed"},
                "interaction": {},
                "resources": {},
                "dirs": {},
            },
            FileNotFoundError,
        ),
        # Test ID: ER-2
        (
            "exp1",
            ["ppis"],
            "/work",
            "loops.bed",
            {
                "local": {},
                "interaction": {},
                "resources": {"reftss_genes": "missing_file.bed"},
                "dirs": {},
            },
            FileNotFoundError,
        ),
    ],
    ids=["ER-1", "ER-2"],
)
def test_EdgeParser_init_error_cases(
    experiment_name,
    interaction_types,
    working_directory,
    loop_file,
    params,
    expected_exception,
):
    # Arrange
    # Act / Assert
    with pytest.raises(expected_exception):
        EdgeParser(
            experiment_name, interaction_types, working_directory, loop_file, params
        )


# Mocking utils and pybedtools for further tests
@pytest.fixture
def mock_utils():
    with patch("edge_parser.utils") as mock:
        yield mock


@pytest.fixture
def mock_pybedtools():
    with patch("edge_parser.pybedtools") as mock:
        yield mock


# Test data preparation
@pytest.fixture
def genesymbol_to_gencode():
    return {
        "geneA": "GENCODE1",
        "geneB": "GENCODE2",
        # Add more mappings as needed for testing
    }


@pytest.fixture
def edge_parser_instance(genesymbol_to_gencode):
    instance = YourClassNameHere()
    instance.genesymbol_to_gencode = genesymbol_to_gencode
    return instance


# Parametrized test cases
@pytest.mark.parametrize(
    "interaction_file, tissue, expected_output, test_id",
    [
        # Happy path tests
        (
            "interaction_file_valid.tsv",
            "brain",
            [("GENCODE1", "GENCODE2", -1, "ppi")],
            "happy_path_1",
        ),
        # Add more happy path test cases with different files and tissues
        # Edge cases
        ("interaction_file_empty.tsv", "brain", [], "edge_case_empty_file"),
        # Add more edge cases as needed
        # Error cases
        pytest.param(
            "interaction_file_missing.tsv",
            "brain",
            [],
            "error_case_missing_file",
            marks=pytest.mark.xfail,
        ),
        # Add more error cases as needed
    ],
)
def test_iid_ppi(
    edge_parser_instance, interaction_file, tissue, expected_output, test_id, tmpdir
):
    # Arrange
    # Create a temporary directory and file for the test case
    file_path = tmpdir.join(interaction_file)
    file_path.write(
        "content based on test_id"
    )  # Replace with actual content needed for the test

    # Act
    result = list(edge_parser_instance._iid_ppi(str(file_path), tissue))

    # Assert
    assert result == expected_output


# More tests would be needed to cover each method and their branches.
# Due to the complexity of the code, it's not feasible to cover all cases in this format.
# The tests would need to mock file I/O, external library calls, and more.
# This would be done using the mock_open, patch, and other features from the unittest.mock module.
