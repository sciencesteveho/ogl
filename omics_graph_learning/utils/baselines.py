#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""This module contains code for severable baseline experiments to understand
the nature of our regression targets. Code is included for:
    (1) Estimating the correlation between experimental replicates, i.e. gives
        an upper bound of "experimental-level accuracy"
    (2) Providing an average expected fold change (combining all genes) between
        experimental replicates
    (3) Building cell-line specific baseline predictors using average activity
    
EXPERIMENT LEVEL ACCURACY:
    We use replicate 1 and replicate 2 of rna-seq experiments as `predicted` vs
    `expected`
    
AVERAGE FOLD CHANGE:
    We take the fold change between all genes of replicate 1 and replicate 2 and
    return an average across all the fold changes
    
AVERAGE ACTIVITY BASELINE (cell lines):
    For the seven cell lines in OGL (and two replicates each), we take the
    Log2TPM of the protein coding genes for each experiment, then take the
    average (mean) across the experiments, and use them to predict expression in
    Rep1 of K562 (average activity predictor as in Schreiber et al., 2020).
"""


from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from matplotlib.colors import LogNorm  # type: ignore
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from pybedtools import BedTool  # type: ignore
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore

PSEUDOCOUNT = 0.25


@dataclass
class RNASeqReplicate:
    """Class to hold RNA-seq replicate information."""

    tissue: str
    replicate_1: str
    replicate_2: Optional[str] = None


class BaselineAnalyzer:
    """Class to handle baseline analysis with optimized gene handling.
    Initializing the analyzer with "chromosomes" will force the analysis to only
    consider genes on those chromosomes.

    Attributes:
        chromosomes: List of chromosomes to consider in the analysis.
        gencode_bed: Path to the gencode bed file.
        genes: List of protein-coding genes from the gencode bed file.
        chr_map: Dictionary of chromosome: gene list.

    Methods
    --------
        analyze_replicates:
            Analyze correlation between rna-seq replicates.
        get_average_fold_change:
            Calculate average fold change between replicates.
        calculate_average_activity_baseline:
            Calculate average activity baseline for a given cell line.
    """

    def __init__(self, gencode_bed: Path, chromosomes: Optional[List[str]] = None):
        """Initialize the BaselineAnalyzer class."""
        self.chromosomes = chromosomes
        self.gencode_bed = gencode_bed
        self.genes = self._get_genes()
        self.chr_map = self._gene_chr_map()
        self._set_matplotlib_params()

    def _set_matplotlib_params(self) -> None:
        """Set matplotlib parameters for publication quality plots."""
        plt.rcParams.update(
            {
                "font.size": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.titlesize": 7,
                "figure.dpi": 450,
                "font.sans-serif": "Arial",
            }
        )

    def _get_genes(self) -> List[str]:
        """Get genes from a gencode bed file, only including the test_set genes."""
        gencode = BedTool(self.gencode_bed)
        if self.chromosomes:
            genes = [
                feature[3]
                for feature in gencode
                if "protein_coding" in str(feature) and feature[0] in self.chromosomes
            ]
        else:
            genes = [
                feature[3]
                for feature in gencode
                if "protein_coding" in str(feature)
                and feature[0] not in ["chrX", "chrY", "chrM"]
            ]
        return [gene for gene in genes if "_" not in gene]

    def _gene_chr_map(self) -> Dict[str, List[str]]:
        """Create a dictionary of chromosome: gene list."""
        gencode = BedTool(self.gencode_bed)
        chr_map: Dict[str, List[str]] = {}
        for feature in gencode:
            gene = feature[3]
            chrom = feature[0]
            if chrom not in chr_map:
                chr_map[chrom] = []
            chr_map[chrom].append(gene)
        return chr_map

    @staticmethod
    def read_encode_rna_seq_data(rna_seq_file: str) -> pd.DataFrame:
        """Read an ENCODE rna-seq tsv, keep only ENSG genes."""
        df = pd.read_table(rna_seq_file, index_col=0, header=[0])
        return df[df.index.str.contains("ENSG")]

    def _get_rna_quantifications(self, rna_matrix: str) -> Dict[str, float]:
        """Returns a dictionary of gene: log transformed + pseudocount TPM
        values.
        """
        df = self.read_encode_rna_seq_data(rna_matrix)
        return (
            (df["TPM"] + PSEUDOCOUNT)
            .apply(lambda x: np.log2(pd.DataFrame([x]))[0][0])
            .to_dict()
        )

    def match_quantification(
        self, gene: str, rna_quantifications: Dict[str, float]
    ) -> Union[float, int]:
        """Retrieve the RNA quantification for a given gene."""
        key = gene.split("_")[0]
        try:
            return rna_quantifications[key]
        except KeyError:
            base_key = key.split(".")[0]
            possible_matches = [
                k for k in rna_quantifications if k.startswith(base_key + ".")
            ]
            if not possible_matches:
                print(f"Gene '{gene}' not found in rna_quantifications.")
                return -1
            matched_key = possible_matches[0]
            return rna_quantifications[matched_key]

    def get_target_genes(self, replicate: str) -> Dict[str, float]:
        """Get log2 transformed TPM values for protein-coding genes."""
        quantifications = self._get_rna_quantifications(replicate)
        return {
            gene: self.match_quantification(gene, quantifications)
            for gene in self.genes
        }

    def filter_targets_by_chromosome(
        self, rep_1: Dict[str, Any], rep_2: Dict[str, Any], chromosomes: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Filter genes by chromosome."""
        common_genes = set(rep_1.keys()) & set(rep_2.keys())
        filtered_genes_1 = {
            gene: rep_1[gene]
            for gene in common_genes
            if gene.split("_")[1] in chromosomes
        }
        filtered_genes_2 = {
            gene: rep_2[gene]
            for gene in common_genes
            if gene.split("_")[1] in chromosomes
        }
        return filtered_genes_1, filtered_genes_2

    def calculate_pearson_correlation(
        self,
        rep_1: Dict[str, Any],
        rep_2: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Calculate pearson correlation between two replicates."""
        return stats.pearsonr(list(rep_1.values()), list(rep_2.values()))

    def plot_predicted_versus_expected(
        self,
        predicted: np.ndarray,
        expected: np.ndarray,
        tissue: str,
        comparison: str = "replication",
    ) -> None:
        """Plot predicted versus expected values"""
        pearson_r, _ = stats.pearsonr(expected, predicted)
        spearman_r, _ = stats.spearmanr(expected, predicted)

        plot = sns.jointplot(
            x=predicted,
            y=expected,
            kind="hex",
            height=3,
            ratio=4,
            space=0.1,
            joint_kws={
                "gridsize": 30,
                "cmap": "viridis",
                "norm": LogNorm(),
                "edgecolors": "white",
                "linewidths": 0.025,
            },
            marginal_kws={
                "element": "step",
                "color": "lightsteelblue",
                "edgecolors": "lightslategray",
                "linewidth": 0.5,
            },
        )

        # set labels for compairison type
        if comparison == "replication":
            x_label = "Replicate 1 Log2 Expression"
            y_label = "Replicate 2 Log2 Expression"
            plot_title = f"{tissue}\nExperimental Replication"
            figure_title = f"{tissue}_experimental_accuracy.png"

        elif comparison == "average_activity":
            x_label = "Average Activity Predicted Log2 Expression"
            y_label = "Expected Log2 Expression"
            plot_title = f"{tissue}\nAverage Activity Baseline"
            figure_title = f"{tissue}_average_activity.png"

        else:
            raise ValueError(
                f"Comparison type '{comparison}' not recognized. "
                "Must be 'replication' or 'average_activity'."
            )

        plot.ax_joint.set_xlabel(x_label)
        plot.ax_joint.set_ylabel(y_label)
        plot.figure.suptitle(
            f"{plot_title}\n" rf"Spearman's $\rho$: {spearman_r:.4f}",
            y=0.95,
        )

        plot.figure.colorbar(
            plot.ax_joint.collections[0],
            ax=plot.ax_joint,
            aspect=5,
            shrink=0.35,
        )

        plot.ax_joint.set_xlim(np.min(expected) - 0.5, np.max(expected) + 0.5)
        plot.ax_joint.set_ylim(np.min(predicted) - 0.5, np.max(predicted) + 0.5)
        plot.ax_joint.text(
            0.025,
            1.05,
            r"$\mathit{r}$ = " + f"{pearson_r:.4f}",
            transform=plot.ax_joint.transAxes,
            fontsize=7,
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(figure_title, dpi=450, bbox_inches="tight")

    def calculate_percent_fold_change(
        self, baseline_log2_tpm: dict, perturbation_log2_tpm: dict
    ) -> dict:
        """Calculate percent fold change for each gene."""
        percent_changes = {}
        for gene, baseline_value in baseline_log2_tpm.items():
            if gene in perturbation_log2_tpm:
                perturbation_value = perturbation_log2_tpm[gene]
                log2_fc = perturbation_value - baseline_value
                percent_changes[gene] = (2**log2_fc - 1) * 100
        return percent_changes

    def analyze_replicates(
        self, replicates: List[RNASeqReplicate]
    ) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Analyze correlation between replicates."""
        correlations = []
        all_predicted = []
        all_expected = []

        for replicate in replicates:
            if not replicate.replicate_2:
                continue

            print(f"Comparing replicates for {replicate.tissue}")
            rep_1 = self.get_target_genes(replicate.replicate_1)
            rep_2 = self.get_target_genes(replicate.replicate_2)

            correlation, p_value = self.calculate_pearson_correlation(
                rep_1=rep_1,
                rep_2=rep_2,
            )

            self.plot_predicted_versus_expected(
                predicted=np.array(list(rep_2.values())),
                expected=np.array(list(rep_1.values())),
                tissue=replicate.tissue,
            )

            correlations.append(correlation)
            all_predicted.extend(list(rep_2.values()))
            all_expected.extend(list(rep_1.values()))

        return correlations, np.array(all_predicted), np.array(all_expected)

    def get_average_fold_change(
        self, replicates: List[RNASeqReplicate]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate average fold change between replicates for each tissue."""
        all_fold_changes = {}

        for replicate in replicates:
            if not replicate.replicate_2:
                continue

            print(f"Calculating fold change for {replicate.tissue}")
            rep_1 = self.get_target_genes(replicate.replicate_1)
            rep_2 = self.get_target_genes(replicate.replicate_2)

            fold_changes = self.calculate_percent_fold_change(rep_1, rep_2)
            all_fold_changes[replicate.tissue] = fold_changes

        return all_fold_changes

    def calculate_average_activity_baseline(
        self,
        replicates: List[RNASeqReplicate],
        target_tissue: str,
        target_replicate: str,
    ) -> Dict[str, Any]:
        """Calculate average activity baseline for a given cell line"""
        # remove target tissue from analysis set
        analysis_replicates = [r for r in replicates if r.tissue != target_tissue]

        all_expression: Dict[str, List[float]] = {gene: [] for gene in self.genes}
        for replicate in analysis_replicates:
            rep_1 = self.get_target_genes(replicate.replicate_1)
            for gene in rep_1:
                all_expression[gene].append(rep_1[gene])

            if replicate.replicate_2:
                rep_2 = self.get_target_genes(replicate.replicate_2)
                for gene in rep_2:
                    all_expression[gene].append(rep_2[gene])

        averaged_activity = {
            gene: np.mean(values) for gene, values in all_expression.items()
        }
        target_expression = self.get_target_genes(target_replicate)

        self.plot_predicted_versus_expected(
            predicted=np.array(list(averaged_activity.values())),
            expected=np.array(list(target_expression.values())),
            tissue=target_tissue,
            comparison="average_activity",
        )

        return averaged_activity

    def _sort_differentially_expressed_genes(
        self, rep_1: Dict[str, float], rep_2: Dict[str, float], top_n: int = 500
    ) -> Dict[str, float]:
        """Get the top N differentially expressed genes between two replicates
        based on absolute log2 fold change.
        """
        fold_changes = {
            gene: rep_2[gene] - rep_1[gene]
            for gene in self.genes
            if gene in rep_1 and gene in rep_2
        }

        # sort genes by absolute fold change in descending order
        sorted_genes = {
            gene: fold_changes[gene]
            for gene in sorted(fold_changes, key=lambda gene: fold_changes[gene])
        }

        return {gene: sorted_genes[gene] for gene in list(sorted_genes.keys())[:top_n]}

    def get_top_differentially_expressed_genes(
        self, replicates: List[RNASeqReplicate], top_n: int = 500
    ) -> Dict[str, Dict[str, float]]:
        """Get the top N differentially expressed genes between replicates."""
        top_genes = {}
        for replicate in replicates:
            if not replicate.replicate_2:
                continue

            rep_1 = self.get_target_genes(replicate.replicate_1)
            rep_2 = self.get_target_genes(replicate.replicate_2)

            top_genes[replicate.tissue] = self._sort_differentially_expressed_genes(
                rep_1, rep_2, top_n=top_n
            )

        return top_genes


def main() -> None:
    """Run baseline analysis."""
    replicates = [
        RNASeqReplicate("K562", "ENCFF384BFE.tsv", "ENCFF611MXW.tsv"),
        RNASeqReplicate("IMR90", "ENCFF325KTI.tsv"),
        RNASeqReplicate("GM12878", "ENCFF362RMV.tsv", "ENCFF723ICA.tsv"),
        RNASeqReplicate("HepG2", "ENCFF103FSL.tsv", "ENCFF692QVJ.tsv"),
        RNASeqReplicate("H1", "ENCFF910OBU.tsv", "ENCFF174OMR.tsv"),
        RNASeqReplicate("HMEC", "ENCFF292FVY.tsv", "ENCFF219EZH.tsv"),
        RNASeqReplicate("NHEK", "ENCFF662VMK.tsv", "ENCFF219QXK.tsv"),
    ]

    avg_baseline_configs = [
        ("K562", "ENCFF611MXW.tsv"),
        ("IMR90", "ENCFF325KTI.tsv"),
        ("GM12878", "ENCFF362RMV.tsv"),
        ("HepG2", "ENCFF103FSL.tsv"),
        ("H1", "ENCFF910OBU.tsv"),
        ("HMEC", "ENCFF292FVY.tsv"),
        ("NHEK", "ENCFF662VMK.tsv"),
    ]

    gencode_bed = Path(
        "/Users/steveho/ogl/development/recap/gencode_v26_genes_only_with_GTEx_targets.bed"
    )

    # create analyzer instance
    # uses all chromosomes for these analyses
    all_analyzer = BaselineAnalyzer(gencode_bed)

    correlations, all_predicted, all_expected = all_analyzer.analyze_replicates(
        replicates
    )

    # plot combined results for all cell lines
    all_analyzer.plot_predicted_versus_expected(
        predicted=all_predicted, expected=all_expected, tissue="Cell lines"
    )

    # get average fold changes
    average_fold_changes = all_analyzer.get_average_fold_change(replicates)
    for tissue, fold_changes in average_fold_changes.items():
        avg_fold_change = np.mean(list(fold_changes.values()))
        print(f"Average fold change for {tissue}: {avg_fold_change}")

    # Average fold change for K562: 4.3313392094563605
    # Average fold change for GM12878: -2.2501806729209903
    # Average fold change for HepG2: 38.167682929021474
    # Average fold change for H1: -1.6475826642098936
    # Average fold change for HMEC: -7.137802643740976
    # Average fold change for NHEK: 6.272474981974544

    # get top differentially expressed genes
    diff_genes = all_analyzer.get_top_differentially_expressed_genes(replicates)
    with open("top_differentially_expressed_genes.txt", "wb") as f:
        pickle.dump(diff_genes, f)

    # use an analyzer with only test set chrs
    activity_analyzer = BaselineAnalyzer(gencode_bed, chromosomes=["chr8", "chr9"])

    # calculate average activity baseline
    for tissue, replicate in avg_baseline_configs:
        activity_analyzer.calculate_average_activity_baseline(
            replicates=replicates, target_tissue=tissue, target_replicate=replicate
        )


if __name__ == "__main__":
    main()
