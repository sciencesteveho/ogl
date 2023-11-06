#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] PRIORITY ** Fix memory leak!
# - [ ] Fix filepaths. They are super ugly!
# - [ ] one-hot encode node_feat type?
#

"""Parse edges from interaction-type omics data"""

from collections import defaultdict
import csv
import itertools
from itertools import chain
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Generator

import numpy as np
import pandas as pd
import pybedtools

from utils import genes_from_gencode
from utils import time_decorator


class EdgeParser:
    """Object to construct tensor based graphs from parsed bedfiles

    The baseline graph structure is build from the following in order:
        Curated protein-protein interactions from the integrated interactions
        database V 2021-05
        TF-gene circuits from Marbach et al.
        TF-gene interactions from TFMarker. We keep "TF" and "I Marker" type relationships
        Enhancer-gene networks from FENRIR
        Enhancer-enhancer networks from FENRIR

        Alternative polyadenylation targets from APAatlas

    Args:
        params: configuration vals from yaml

    Methods
    ----------
    _genes_from_gencode:
        Lorem
    _base_graph:
        Lorem
    _iid_ppi:
        Lorem
    _mirna_targets:
        Lorem
    _tf_markers:
        Lorem
    _marchbach_regulatory_circuits:
        Lorem
    _enhancer_index:
        Lorem
    _format_enhancer:
        Lorem
    _process_graph_edges:
        Lorem
    """

    def __init__(
        self,
        experiment_name: str,
        interaction_types: List[str],
        working_directory: str,
        loop_file: str,
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.experiment_name = experiment_name
        self.interaction_types = interaction_types
        self.working_directory = working_directory
        self.loop_file = loop_file

        self.gencode = params["local"]["gencode"]
        self.interaction_files = params["interaction"]
        self.tissue = params["resources"]["tissue"]
        self.tissue_name = params["resources"]["tissue_name"]
        self.marker_name = params["resources"]["marker_name"]
        self.ppi_tissue = params["resources"]["ppi_tissue"]
        self.tss = params["resources"]["reftss_genes"]
        self.shared = params["local"]

        self.root_dir = params["dirs"]["root_dir"]
        self.shared_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = (
            f"{self.working_directory}/{self.experiment_name}/{self.tissue}"
        )
        self.local_dir = f"{self.tissue_dir}/local"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.interaction_dir = f"{self.tissue_dir}/interaction"
        self.shared_interaction_dir = f"{self.shared_dir}/interaction"

        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_attr_ref = self._create_reference_dict(
            params["resources"]["gencode_attr"]
        )
        self.regulatory_attr_ref = self._create_reference_dict(
            params["resources"]["reg_ref"]
        )
        self.se_ref = self._create_reference_dict(params["resources"]["se_ref"])
        self.mirna_ref = self._create_reference_dict(
            f"{self.interaction_dir}/{params['interaction']['mirdip']}"
        )
        self.footprint_ref = self._create_reference_dict(
            f"{self.tissue_dir}/unprocessed/tfbindingsites_ref.bed"
        )

    def _create_reference_dict(self, file: str) -> List[str]:
        """Reads a file and stores its lines in a dictionary"""
        try:
            return {
                line[3]: line[0:4]
                for line in csv.reader(open(file, newline=""), delimiter="\t")
            }
        except FileNotFoundError:
            return {}

    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
    ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        df = pd.read_csv(interaction_file, delimiter="\t")
        df = df[["symbol1", "symbol2", "evidence_type", "n_methods", tissue]]
        t_spec_filtered = df[
            (df[tissue] > 0)
            & (df["n_methods"] >= 3)
            & (df["evidence_type"].str.contains("exp"))
        ]
        edges = list(
            zip(
                *map(t_spec_filtered.get, ["symbol1", "symbol2"]),
                itertools.repeat(-1),
                itertools.repeat("ppi"),
            )
        )
        return [
            (
                f"{self.genesymbol_to_gencode[edge[0]]}",
                f"{self.genesymbol_to_gencode[edge[1]]}",
                edge[2],
                edge[3],
            )
            for edge in edges
            if edge[0] in self.genesymbol_to_gencode.keys()
            and edge[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _mirna_targets(
        self,
        target_list: str,
        tissue_active_mirnas: str,
    ) -> List[Tuple[str, str]]:
        """Filters all miRNA -> target interactions from miRTarBase and only
        keeps the miRNAs that are active in the given tissue from mirDIP.
        """
        active_mirna = [
            line[3]
            for line in csv.reader(
                open(tissue_active_mirnas, newline=""), delimiter="\t"
            )
        ]

        return [
            (
                line[0],
                self.genesymbol_to_gencode[line[1]],
                -1,
                "mirna",
            )
            for line in csv.reader(open(target_list, newline=""), delimiter="\t")
            if line[0] in active_mirna and line[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _tf_markers(self, interaction_file: str) -> List[Tuple[str, str]]:
        tf_keep = ["TF", "I Marker", "TFMarker"]
        tf_markers = []
        with open(interaction_file, newline="") as file:
            file_reader = csv.reader(file, delimiter="\t")
            next(file_reader)
            for line in file_reader:
                if line[2] in tf_keep and line[5] == self.marker_name:
                    try:
                        if ";" in line[10]:
                            genes = line[10].split(";")
                            for gene in genes:
                                if line[2] == "I Marker":
                                    tf_markers.append((gene, line[1]))
                                else:
                                    tf_markers.append((line[1], gene))
                        else:
                            if line[2] == "I Marker":
                                tf_markers.append((line[10], line[1]))
                            else:
                                tf_markers.append((line[1], line[10]))
                    except IndexError:
                        pass
        return [
            (
                f"{self.genesymbol_to_gencode[tup[0]]}_tf",
                self.genesymbol_to_gencode[tup[1]],
                -1,
                "tf_marker",
            )
            for tup in tf_markers
            if tup[0] in self.genesymbol_to_gencode.keys()
            and tup[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str,
        score_filter: int,
    ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
        network is in the following format:
            col_1   TF
            col_2   Target gene
            col_3   Edge weight
        """
        tf_g, scores = [], []
        with open(interaction_file, newline="") as file:
            file_reader = csv.reader(file, delimiter="\t")
            for line in file_reader:
                scores.append(float(line[2]))
                if (
                    line[0] in self.genesymbol_to_gencode.keys()
                    and line[1] in self.genesymbol_to_gencode.keys()
                ):
                    tf_g.append((line[0], line[1], float(line[2])))

        cutoff = np.percentile(scores, score_filter)

        return [
            (
                f"{self.genesymbol_to_gencode[line[0]]}_tf",
                self.genesymbol_to_gencode[line[1]],
                line[2],
                "circuits",
            )
            for line in tf_g
            if line[2] >= cutoff
        ]

    @time_decorator(print_args=True)
    def _tfbinding_footprints(
        self,
        tfbinding_file: str,
        footprint_file: str,
    ) -> List[Tuple[str, str, float, str]]:
        """Create edges based on whether or not known TF binding from Meuleman
        et al. overlap footprints from Vierstra et al.

        Args:
            tfbinding_file (str): shared tf binding locations from Meuleman
            footprint_file (str): tissue-specific footprints from Vierstra

        Returns:
            List[Tuple[str, str, float, str]]
        """
        tf_binding = pybedtools.BedTool(tfbinding_file)
        tf_edges = tf_binding.intersect(footprint_file, wb=True)
        return [
            (
                f"{self.genesymbol_to_gencode[line[3]]}_tf",
                f"{line[5]}_{line[6]}_{line[3]}",
                -1,
                "tf_binding_footprint",
            )
            for line in tf_edges
        ]

    def _load_tss(self) -> pybedtools.BedTool:
        """Load TSS file and ignore any TSS that do not have a gene target.

        Returns:
            pybedtools.BedTool - TSS w/ target genes
        """
        tss = pybedtools.BedTool(f"{self.tss}")
        return tss.filter(lambda x: x[3].split("_")[3] != "").saveas()

    @time_decorator(print_args=True)
    def get_loop_edges(
        self,
        chromatin_loops: str,
        feat_1: str,
        feat_2: str,
        edge_type: str,
        tss: bool = False,
    ) -> Generator[Tuple[str, str, float, str], None, None]:
        """Connects nodes if they are linked by chromatin loops. Can specify if
        the loops should only be done for direct overlaps or if they should
        be within 2mb of a loop anchor for TSS. If using TSS, make sure to
        specify the TSS as the second feature!

        Args:
            feat_1 (str): _description_
            feat_2 (str): _description_
            type (str): _description_

        Returns:
            Generator[Tuple[str, str, float, str], None, None]: _description_
        """

        def _loop_direct_overlap(
            loops: pybedtools.BedTool, features: pybedtools.BedTool
        ) -> pybedtools.BedTool:
            """Get features that directly overlap with loop anchor"""
            return loops.intersect(features, wo=True)

        def _loop_within_distance(
            loops: pybedtools.BedTool,
            features: pybedtools.BedTool,
            distance: int,
        ) -> pybedtools.BedTool:
            """Get features 2kb within loop anchor

            Args:
                loops (pybedtools.BedTool): _description_
                features (pybedtools.BedTool): _description_
                distance (int): _description_
            """
            return loops.window(features, w=distance)

        def _split_chromatin_loops(
            chromatin_loops: str,
        ) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
            """_summary_
            Returns:
                Tuple[pybedtools.BedTool, pybedtools.BedTool]: _description_
            """
            first_anchor = pybedtools.BedTool(chromatin_loops)
            second_anchor = pybedtools.BedTool(
                "\n".join(
                    [
                        "\t".join([x[3], x[4], x[5], x[0], x[1], x[2]])
                        for x in first_anchor
                    ]
                ),
                from_string=True,
            )
            return first_anchor.cut([0, 1, 2, 3, 4, 5]), second_anchor.cut(
                [0, 1, 2, 3, 4, 5]
            )

        def _flatten_anchors(*beds: pybedtools.BedTool) -> Dict[str, List[str]]:
            """Creates a dict to store each anchor and its overlaps. Adds the feature by
            ignoring the first 7 columns of the bed file and adding whatever is left."""
            anchor = defaultdict(list)
            for feature in itertools.chain(*beds):
                anchor["_".join(feature[:3])].append(
                    "_".join([feature[6], feature[7], feature[9]])
                )
            return anchor

        def _loop_edges(
            loops: pybedtools.BedTool,
            first_anchor_edges: Dict[str, List[str]],
            second_anchor_edges: Dict[str, List[str]],
        ) -> Generator[Tuple[str, str], None, None]:
            """Return a generator of edges that are connected by their overlap over chromatin
            loop anchors by matching the anchor names across dicts"""
            for loop in loops:
                first_anchor = "_".join(loop[0:3])
                second_anchor = "_".join(loop[3:6])
                try:
                    for edge in itertools.product(
                        first_anchor_edges[first_anchor],
                        second_anchor_edges[second_anchor],
                    ):
                        yield edge
                except KeyError:
                    continue

        def _check_tss_gene_in_gencode(tss: str) -> bool:
            """Simple check to see if gene has gencode ID"""
            gene = tss.split("_")[5]
            return self.genesymbol_to_gencode.get(gene, False)

        # split loops into anchors
        first_anchor, second_anchor = _split_chromatin_loops(chromatin_loops)

        if tss:
            first_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(first_anchor, feat_1),
                _loop_within_distance(first_anchor, feat_2, 2000),
            )
            second_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(second_anchor, feat_1),
                _loop_within_distance(second_anchor, feat_2, 2000),
            )
            for edge in _loop_edges(
                first_anchor, first_anchor_edges, second_anchor_edges
            ):
                if "tss" in edge[0] and "tss" in edge[1]:
                    if _check_tss_gene_in_gencode(
                        edge[0]
                    ) and _check_tss_gene_in_gencode(edge[1]):
                        yield (
                            _check_tss_gene_in_gencode(edge[0]),
                            _check_tss_gene_in_gencode(edge[1]),
                            -1,
                            "g_g",
                        )
                    else:
                        pass
                elif "tss" in edge[0] and "tss" not in edge[1]:
                    if _check_tss_gene_in_gencode(edge[0]):
                        yield (
                            _check_tss_gene_in_gencode(edge[0]),
                            edge[1],
                            -1,
                            edge_type,
                        )
                    else:
                        pass
                elif "tss" not in edge[0] and "tss" in edge[1]:
                    if _check_tss_gene_in_gencode(edge[1]):
                        yield (
                            edge[0],
                            _check_tss_gene_in_gencode(edge[1]),
                            -1,
                            edge_type,
                        )
                    else:
                        pass
                else:
                    yield (
                        edge[0],
                        edge[1],
                        -1,
                        edge_type,
                    )
        else:
            first_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(first_anchor, feat_1),
                _loop_direct_overlap(first_anchor, feat_2),
            )
            second_anchor_edges = _flatten_anchors(
                _loop_direct_overlap(second_anchor, feat_1),
                _loop_direct_overlap(second_anchor, feat_2),
            )
            for edge in _loop_edges(
                first_anchor, first_anchor_edges, second_anchor_edges
            ):
                yield (edge[0], edge[1], -1, edge_type)

    @time_decorator(print_args=True)
    def _process_graph_edges(self) -> None:
        """Retrieve all interaction edges and saves them to a text file. Edges
        will be loaded from the text file for subsequent runs to save
        processing time.

        Returns:
            A list of deduplicated nodes (separate lists for genes and for miRNAs)
        """
        tss = self._load_tss()
        distal_enhancers = (
            pybedtools.BedTool(f"{self.local_dir}/{self.shared['enhancers']}")
            .filter(lambda x: x[3] == "dELS")
            .saveas()
        )
        promoters = pybedtools.BedTool(f"{self.local_dir}/{self.shared['promoters']}")
        dyadic = pybedtools.BedTool(f"{self.local_dir}/{self.shared['dyadic']}")

        gene_overlaps = [
            (distal_enhancers, "g_e"),
            (promoters, "g_p"),
            (dyadic, "g_d"),
        ]
        promoters_overlaps = [
            (distal_enhancers, "p_e"),
            (dyadic, "p_d"),
        ]

        try:
            if "superenhancers" in self.interaction_types:
                super_enhancers = pybedtools.BedTool(
                    f"{self.local_dir}/superenhancers_{self.tissue}.bed"
                )
                gene_overlaps = gene_overlaps + [(super_enhancers, "g_se")]
                promoter_overlaps = promoter_overlaps + [(super_enhancers, "p_se")]
        except TypeError:
            pass

        chrom_loop_edges = list(
            chain(
                (
                    self.get_loop_edges(
                        chromatin_loops=self.loop_file,
                        feat_1=element[0],
                        feat_2=tss,
                        tss=True,
                        edge_type=element[1],
                    )
                    for element in gene_overlaps
                ),
                (
                    self.get_loop_edges(
                        chromatin_loops=self.loop_file,
                        feat_1=element[0],
                        feat_2=promoters,
                        tss=False,
                        edge_type=element[1],
                    )
                    for element in promoters_overlaps
                ),
            )
        )

        # only parse edges specified in experiment
        ppi_edges, mirna_targets, tf_markers, circuit_edges, tfbinding_edges = (
            [],
            [],
            [],
            [],
            [],
        )
        if self.interaction_types is not None:
            if "mirna" in self.interaction_types:
                mirna_targets = self._mirna_targets(
                    target_list=f"{self.interaction_dir}/{self.interaction_files['mirnatargets']}",
                    tissue_active_mirnas=f"{self.interaction_dir}/{self.interaction_files['mirdip']}",
                )
            if "tf_marker" in self.interaction_types:
                tf_markers = self._tf_markers(
                    interaction_file=f"{self.interaction_dir}/{self.interaction_files['tf_marker']}",
                )
            if "circuits" in self.interaction_types:
                circuit_edges = self._marbach_regulatory_circuits(
                    f"{self.interaction_dir}/{self.interaction_files['circuits']}",
                    score_filter=30,
                )
            if "tfbinding" in self.interaction_types:
                tfbinding_edges = self._tfbinding_footprints(
                    tfbinding_file=f"{self.shared_interaction_dir}/{self.interaction_files['tfbinding']}",
                    footprint_file=f"{self.local_dir}/{self.shared['footprints']}",
                )
            # if "ppis" in self.interaction_types:
            #     ppi_edges = self._iid_ppi(
            #         interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
            #         tissue=self.ppi_tissue,
            #     )

        self.interaction_edges = list(
            chain(ppi_edges, mirna_targets, tf_markers, circuit_edges, tfbinding_edges)
        )
        self.chrom_edges = list(set(chrom_loop_edges))
        self.all_edges = list(chain(self.chrom_edges, self.interaction_edges))

        chrom_loops_regulatory_nodes = set(
            chain.from_iterable(
                (
                    (edge[0], edge[1])
                    for edge in self.chrom_edges
                    if "ENSG" not in edge[0] and "superenhancer" not in edge[0]
                )
            )
        )

        chrom_loops_se_nodes = set(
            chain.from_iterable(
                (
                    (edge[0], edge[1])
                    for edge in self.chrom_edges
                    if "superenhancer" in edge[0]
                )
            )
        )

        gencode_nodes = set(
            chain(
                (tup[0] for tup in ppi_edges),
                (tup[1] for tup in ppi_edges),
                (tup[1] for tup in mirna_targets),
                (tup[0] for tup in tf_markers),
                (tup[1] for tup in tf_markers),
                (tup[0] for tup in circuit_edges),
                (tup[1] for tup in circuit_edges),
                (edge[0] for edge in self.chrom_edges if "ENSG" in edge[0]),
                (edge[1] for edge in self.chrom_edges if "ENSG" in edge[1]),
                (tup[0] for tup in tfbinding_edges),
            )
        )

        return (
            set(gencode_nodes),
            set(chrom_loops_regulatory_nodes),
            set(chrom_loops_se_nodes),
            set([tup[0] for tup in mirna_targets]),
            set(tup[1] for tup in tfbinding_edges),
        )

    @time_decorator(print_args=False)
    def _add_node_coordinates(
        self,
        nodes,
        node_ref,
    ) -> None:
        """_summary_

        Args:
            nodes:
            node_ref:
        """
        if len(nodes) == 0:
            return []
        else:
            return [node_ref[node] for node in nodes]

    @time_decorator(print_args=True)
    def parse_edges(self) -> None:
        """Constructs tissue-specific interaction base graph"""

        print("Parsing edges...")
        # retrieve interaction-based edges
        (
            gencode_nodes,
            regulatory_nodes,
            se_nodes,
            mirnas,
            footprints,
        ) = self._process_graph_edges()

        print("Parsing edges complete!")
        print("Adding coordinates to nodes...")
        # add coordinates to nodes in parallel
        pool = Pool(processes=5)
        nodes_for_attr = list(
            chain.from_iterable(
                pool.starmap(
                    self._add_node_coordinates,
                    zip(
                        [gencode_nodes, regulatory_nodes, se_nodes, mirnas, footprints],
                        [
                            self.gencode_attr_ref,
                            self.regulatory_attr_ref,
                            self.se_ref,
                            self.mirna_ref,
                            self.footprint_ref,
                        ],
                    ),
                )
            )
        )
        pool.close()
        nodes_for_attr = sum(nodes_for_attr, [])  # flatten list of lists

        print("Adding coordinates to nodes complete!")
        print("Writing nodes and edges to file...")
        # write nodes to file
        with open(f"{self.tissue_dir}/local/basenodes_hg38.txt", "w+") as output:
            csv.writer(output, delimiter="\t").writerows(nodes_for_attr)

        print("Writing nodes to file complete!")
        print("Writing edges to file...")
        # add coordinates to edges
        full_edges = []
        nodes_with_coords = {node[3]: node[0:3] for node in nodes_for_attr}
        for edge in self.all_edges:
            if edge[0] in nodes_with_coords and edge[1] in nodes_with_coords:
                full_edges.append(
                    [edge[0]]
                    + nodes_with_coords[edge[0]]
                    + [edge[1]]
                    + nodes_with_coords[edge[1]]
                    + [edge[2], edge[3]]
                )

        # write edges to file
        all_interaction_file = f"{self.interaction_dir}/interaction_edges.txt"

        with open(all_interaction_file, "w+") as output:
            csv.writer(output, delimiter="\t").writerows(self.all_edges)

        # write edges with coordinates to file
        with open(f"{self.interaction_dir}/full_edges.txt", "w+") as output:
            csv.writer(output, delimiter="\t").writerows(full_edges)
