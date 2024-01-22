from io import StringIO
from unittest.mock import mock_open
from unittest.mock import patch

from edge_parser import EdgeParser
import numpy as np
import pandas as pd
import pybedtools
import pytest

# Constants for tests
EXPERIMENT_NAME = "test_experiment"
INTERACTION_TYPES = ["mirna", "tf_marker", "circuits", "tfbinding"]
WORKING_DIRECTORY = "/test_working_dir"
LOOP_FILE = "test_loops.bed"
PARAMS = {
    "local": {
        "gencode": "test_gencode.bed",
        "enhancers": "test_enhancers.bed",
        "promoters": "test_promoters.bed",
        "dyadic": "test_dyadic.bed",
    },
    "interaction": {
        "mirnatargets": "test_mirnatargets.tsv",
        "mirdip": "test_mirdip.tsv",
        "tf_marker": "test_tf_marker.tsv",
        "circuits": "test_circuits.tsv",
        "tfbinding": "test_tfbinding.bed",
    },
    "resources": {
        "tissue": "test_tissue",
        "tissue_name": "test_tissue_name",
        "marker_name": "test_marker_name",
        "ppi_tissue": "test_ppi_tissue",
        "reftss_genes": "test_reftss_genes.bed",
        "gencode_attr": "test_gencode_attr.tsv",
        "reg_ref": "test_reg_ref.tsv",
        "se_ref": "test_se_ref.tsv",
    },
    "dirs": {
        "root_dir": "/test_root_dir",
    },
}

# Mock data for files
GENCODE_FILE_CONTENT = """chr1\t11868\t14409\tENSG00000223972.5\t0\t+\nchr1\t14403\t29570\tENSG00000227232.5\t0\t-\n"""
MIRNA_TARGETS_CONTENT = (
    """hsa-miR-123\tENSG00000123456\nhsa-miR-789\tENSG00000234567\n"""
)
MIRNA_ACTIVE_CONTENT = """hsa-miR-123\t0.9\nhsa-miR-789\t0.1\n"""
TF_MARKER_CONTENT = """chr1\t11868\t14409\tTF\ttest_marker_name\tENSG00000123456\nchr1\t14403\t29570\tI Marker\ttest_marker_name\tENSG00000234567\n"""
CIRCUITS_CONTENT = (
    """ENSG00000123456\tENSG00000234567\t0.8\nENSG00000123456\tENSG00000345678\t0.2\n"""
)
TFBINDING_CONTENT = """chr1\t11868\t14409\tENSG00000123456\t0\t+\nchr1\t14403\t29570\tENSG00000234567\t0\t-\n"""


# Helper functions for tests
def create_bedtool_from_string(content):
    return pybedtools.BedTool(StringIO(content), from_string=True)


@pytest.mark.parametrize(
    "test_id, interaction_file, tissue, expected_output",
    [
        # Happy path tests
        (
            "HP_1",
            "test_ppi.tsv",
            "test_tissue",
            [("ENSG00000123456", "ENSG00000234567", -1, "ppi")],
        ),
        # Edge cases
        ("EC_1", "test_ppi_empty.tsv", "test_tissue", []),
        # Error cases
        (
            "ER_1",
            "test_ppi_nonexistent.tsv",
            "test_tissue",
            pytest.raises(FileNotFoundError),
        ),
    ],
)
def test_iid_ppi(test_id, interaction_file, tissue, expected_output):
    # Arrange
    with patch("builtins.open", mock_open(read_data=GENCODE_FILE_CONTENT)) as mock_file:
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {
                    "symbol1": ["ENSG00000123456"],
                    "symbol2": ["ENSG00000234567"],
                    "evidence_type": ["exp"],
                    "n_methods": [3],
                    tissue: [1],
                }
            )
            edge_parser = EdgeParser(
                EXPERIMENT_NAME, INTERACTION_TYPES, WORKING_DIRECTORY, LOOP_FILE, PARAMS
            )

    # Act
    if isinstance(expected_output, type) and issubclass(expected_output, BaseException):
        with expected_output:
            edge_parser._iid_ppi(interaction_file, tissue)
    else:
        result = edge_parser._iid_ppi(interaction_file, tissue)

    # Assert
    assert result == expected_output


# Additional test functions would follow the same pattern:
# - test_mirna_targets
# - test_tf_markers
# - test_marbach_regulatory_circuits
# - test_tfbinding_footprints
# - test_get_loop_edges
# - test_process_graph_edges
# - test_add_node_coordinates
# - test_parse_edges

# Note: The above test cases are examples and should be expanded upon to achieve 100% line and branch coverage.
# This includes testing different branches within each method, such as different conditions in if statements,
# different branches of try/except blocks, and so on.
