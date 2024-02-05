#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - []

"""Parse edges from interaction-type omics data"""


import contextlib
import csv
from typing import Dict, Generator, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
import pybedtools

import utils


class EdgeParser:
    """Object to construct tensor based graphs from parsed bedfiles

    Attributes:
        experiment_name (str): The name of the experiment.
        interaction_types (List[str]): The types of interactions to consider.
        working_directory (str): The working directory.
        loop_file (str): The loop file.
        params (Dict[str, Dict[str, str]]): Configuration values from YAML.

    Methods
    ----------
    _initialize_directories:
        Initialize directory paths.
    _initialize_references:
        Initialize reference dictionaries.
    _create_reference_dict:
        Reads a file and stores its lines in a dictionary.
    _iid_ppi:
        Protein-protein interactions from the Integrated Interactions Database v
        2021-05.
    _mirna_targets:
        Filters all miRNA -> target interactions from miRTarBase and only keeps
        the miRNAs that are active in the given tissue from mirDIP.
    _tf_markers:
        Filters TF markers based on specific conditions.
    _marbach_regulatory_circuits:
        Regulatory circuits from Marbach et al., Nature Methods, 2016.
    _tfbinding_footprints:
        Create edges based on whether or not known TF binding from Meuleman et
        al. overlap footprints from Vierstra et al.
    _load_tss:
        Load TSS file and ignore any TSS that do not have a gene target.
    _process_graph_edges:
        Retrieve all interaction edges and saves them to a text file.
    _add_node_coordinates:
        Add coordinates to nodes.
    get_loop_edges:
        Connects nodes if they are linked by chromatin loops.
    parse_edges:
        Constructs tissue-specific interaction base graph.
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

        self._initialize_directories(params)
        self._initialize_references(params)

    def _initialize_directories(self, params: Dict[str, Dict[str, str]]) -> None:
        """Initialize directory paths"""
        self.gencode = params["local"]["gencode"]
        self.interaction_files = params["interaction"]
        self.tissue = params["resources"]["tissue"]
        self.chromfile = params["resources"]["chromfile"]
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

    def _initialize_references(self, params: Dict[str, Dict[str, str]]) -> None:
        """Initialize reference dictionaries"""
        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = utils.genes_from_gencode(
            gencode_ref=self.gencode_ref
        )
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
                line[3]: line[:4]
                for line in csv.reader(open(file, newline=""), delimiter="\t")
            }
        except FileNotFoundError:
            return {}

    def _read_csv_wrapper(
        self, file_path: str, with_header: bool = True
    ) -> Union[Tuple[List[str], Iterator[List[str]]], Iterator[List[str]]]:
        """Wrapper function to read a CSV file."""
        with open(file_path, newline="") as file:
            reader = csv.reader(file, delimiter="=\t")
            if with_header:
                header = next(reader)
                return header, reader
            return reader

    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
    ) -> Generator[Tuple[str, str, float, str], None, None]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        header, reader = self._read_csv_with_header(interaction_file, with_header=True)
        idxs = {
            key: header.index(key)
            for key in ["symbol1", "symbol2", "evidence_type", "n_methods", tissue]
        }

        for line in reader:
            if (
                float(line[idxs[tissue]]) > 0
                and int(line[idxs["n_methods"]]) >= 3
                and "exp" in line[idxs["evidence_type"]]
                and line[idxs["symbol1"]] in self.genesymbol_to_gencode
                and line[idxs["symbol2"]] in self.genesymbol_to_gencode
            ):
                yield (
                    self.genesymbol_to_gencode[line[idxs["symbol1"]]],
                    self.genesymbol_to_gencode[line[idxs["symbol2"]]],
                    # -1,
                    "ppi",
                )

    def _mirna_targets(
        self,
        target_list: str,
        tissue_active_mirnas: str,
    ) -> Generator[Tuple[str, str, float, str], None, None]:
        """Filters all miRNA -> target interactions from miRTarBase and only
        keeps the miRNAs that are active in the given tissue from mirDIP.
        """
        active_mirna_reader = self._read_csv_with_header(tissue_active_mirnas)
        active_mirna = {line[3] for line in active_mirna_reader}

        target_reader = self._read_csv_with_header(target_list)
        for line in target_reader:
            if line[0] in active_mirna and line[1] in self.genesymbol_to_gencode:
                yield (
                    line[0],
                    self.genesymbol_to_gencode[line[1]],
                    # -1,
                    "mirna",
                )

    def _tf_markers(
        self, interaction_file: str
    ) -> Generator[Tuple[str, str, float, str], None, None]:
        """Filters tf markers based on specified conditions.

        Args:
            interaction_file (str): The path to the interaction file.

        Returns:
            List[Tuple[str, str]]: A list of filtered tf marker interactions.
        """
        tf_keep = ["TF", "I Marker", "TFMarker"]
        _, reader = self._read_csv_with_header(interaction_file)

        tf_markers = []
        for line in reader:
            if line[2] in tf_keep and line[5] == self.marker_name:
                with contextlib.suppress(IndexError):
                    if ";" in line[10]:
                        genes = line[10].split(";")
                        for gene in genes:
                            if line[2] == "I Marker":
                                tf_markers.append((gene, line[1]))
                            else:
                                tf_markers.append((line[1], gene))
                    elif line[2] == "I Marker":
                        tf_markers.append((line[10], line[1]))
                    else:
                        tf_markers.append((line[1], line[10]))

        for tup in tf_markers:
            if (
                tup[0] in self.genesymbol_to_gencode.keys()
                and tup[1] in self.genesymbol_to_gencode.keys()
            ):
                yield (
                    f"{self.genesymbol_to_gencode[tup[0]]}_tf",
                    self.genesymbol_to_gencode[tup[1]],
                    # -1,
                    "tf_marker",
                )

    def _marbach_regulatory_circuits(
        self,
        interaction_file: str,
        score_filter: int,
    ) -> Generator[Tuple[str, str, float, str], None, None]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
        network is in the following format:
            col_1   TF
            col_2   Target gene
            col_3   Edge weight
        """
        reader = self._read_csv_with_header(interaction_file)

        scores = [
            float(line[2])
            for line in reader
            if (
                line[0] in self.genesymbol_to_gencode
                and line[1] in self.genesymbol_to_gencode
            )
        ]
        cutoff = np.percentile(scores, score_filter)

        reader = self._read_csv_with_header(interaction_file)  # Re-read the file
        for line in reader:
            tf, target_gene, weight = line[0], line[1], float(line[2])
            if (
                tf in self.genesymbol_to_gencode
                and target_gene in self.genesymbol_to_gencode
                and weight >= cutoff
            ):
                yield (
                    f"{self.genesymbol_to_gencode[tf]}_tf",
                    self.genesymbol_to_gencode[target_gene],
                    # weight,
                    "circuits",
                )

    def _tfbinding_footprints(
        self,
        tfbinding_file: str,
        footprint_file: str,
    ) -> Generator[Tuple[str, str, float, str], None, None]:
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
        for line in tf_edges:
            yield (
                f"{self.genesymbol_to_gencode[line[3]]}_tf",
                f"{line[5]}_{line[6]}_{line[3]}",
                # -1,
                "tf_binding_footprint",
            )

    def _check_tss_gene_in_gencode(self, tss: str) -> bool:
        """Check if gene associated with TSS is in gencode v26"""
        gene = tss.split("_")[5]
        return self.genesymbol_to_gencode.get(gene, False)

    def _write_node_list(self, node: Tuple[Union[str, int]]) -> None:
        """Write gencode nodes to file"""
        with open(f"{self.local_dir}/gencode_nodes.txt", "a") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(node)

    def _write_noderef_combination(self, node: str) -> None:
        """Writes chr, start, stop, node to a file. Gets coords from ref
        dict."""
        ref_mapping = {"ENGS": self.gencode_attr_ref, "superenhancer": self.se_ref}
        ref = ref_mapping.get(node, self.regulatory_attr_ref)
        self._write_node_list(self._add_node_coordinates(node, ref))

    def _write_edges(self, edge: Tuple[Union[str, int]]) -> None:
        """Write edge to file"""
        with open(f"{self.interaction_dir}/interaction_edges.txt", "a") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(edge)

    def _prepare_interaction_generators(self):
        """Initiate interaction type generators"""
        if "ppis" in self.interaction_types:
            ppi_generator = self._iid_ppi(
                interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
                tissue=self.ppi_tissue,
            )
        if "mirna" in self.interaction_types:
            mirna_generator = self._mirna_targets(
                target_list=f"{self.interaction_dir}/{self.interaction_files['mirnatargets']}",
                tissue_active_mirnas=f"{self.interaction_dir}/{self.interaction_files['mirdip']}",
            )
        if "tf_marker" in self.interaction_types:
            tf_generator = self._tf_markers(
                interaction_file=f"{self.interaction_dir}/{self.interaction_files['tf_marker']}",
            )
        if "circuits" in self.interaction_types:
            circuit_generator = self._marbach_regulatory_circuits(
                f"{self.interaction_dir}/{self.interaction_files['circuits']}",
                score_filter=30,
            )
        if "tfbinding" in self.interaction_types:
            tfbinding_generator = self._tfbinding_footprints(
                tfbinding_file=f"{self.shared_interaction_dir}/{self.interaction_files['tfbinding']}",
                footprint_file=f"{self.local_dir}/{self.shared['footprints']}",
            )

        return (
            ppi_generator,
            mirna_generator,
            tf_generator,
            circuit_generator,
            tfbinding_generator,
        )

    def _add_node_coordinates(
        self,
        node: str,
        node_ref: Dict[str, List[str]],
    ) -> Tuple[str, str, str, str]:
        """Add coordinates to nodes based on the given node reference."""
        return node_ref[node]

    def _run_generator_common(
        self,
        generator: Generator,
        attr_refs: List[Dict[str, List[str]], Dict[str, List[str]]],
    ) -> None:
        """Runs a generator and processes its results. Returns nothing.

        Args:
            generator (Generator): The generator to run.
            attr_refs (List[Dict[str, List[str]], Dict[str, List[str]]]): The attribute references.
        """
        for result in generator:
            self._write_edges(result)
            for element, attr_ref in zip(result, attr_refs):
                self._write_node_list(self._add_node_coordinates(element, attr_ref))

    @utils.time_dectorator(print_args=True)
    def _process_interaction_edges(self) -> None:
        """Retrieve all interaction edges and saves them to a text file. Edges
        will be loaded from the text file for subsequent runs to save
        processing time.

        Returns:
            A list of deduplicated nodes (separate lists for genes and for miRNAs)
        """

        def _run_generator(generator):
            attr_refs = [self.gencode_attr_ref, self.gencode_attr_ref]
            self._run_generator_common(generator, attr_refs)

        def _run_mirna_generator(generator):
            attr_refs = [self.mirna_ref, self.gencode_attr_ref]
            self._run_generator_common(generator, attr_refs)

        def _run_tfbinding_generator(generator):
            attr_refs = [self.gencode_attr_ref, self.footprint_ref]
            self._run_generator_common(generator, attr_refs)

        # get generators
        (
            ppi_generator,
            mirna_generator,
            tf_generator,
            circuit_generator,
            tfbinding_generator,
        ) = self._prepare_interaction_generators()

        # run generators!
        for gen in [ppi_generator, tf_generator, circuit_generator]:
            _run_generator(gen)
        _run_mirna_generator(mirna_generator)
        _run_tfbinding_generator(tfbinding_generator)

    def _overlap_groupby(
        self,
        anchor: pybedtools.BedTool,
        features: pybedtools.BedTool,
        overlap_func: callable,
    ) -> pybedtools.BedTool:
        """Gets specific overlap, renames regulatory features, and uses groupby to
        collapse overlaps into the same anchor"""
        return (
            overlap_func(anchor, features)
            .each(self._add_feat_names)
            .groupby(
                g=[1, 2, 3, 4, 5, 6],
                c=7,
                o="collapse",
            )
        )

    def _write_loop_edges(
        self,
        edges_df: pd.DataFrame,
        file_path: str,
        tss=False,
    ) -> None:
        """Write the edges to a file in bulk."""

        def _process_edge_nodes(row):
            for edge in ("edge_0", "edge_1"):
                if "tss" in row[edge]:
                    checked_edge = self._check_tss_gene_in_gencode(row[edge])
                    if checked_edge:
                        row[edge] = checked_edge

        if tss:
            edges_df = edges_df.apply(_process_edge_nodes, axis=1)
        edges_df.to_csv(file_path, sep="\t", mode="a", header=False, index=False)
        return set(edges_df["edge_0"].unique() + edges_df["edge_1"].unique())

    @utils.time_dectorator(print_args=True)
    def _process_loop_edges(
        self,
        first_feature: pybedtools.BedTool,
        second_feature: pybedtools.BedTool,
        edge_type: str,
    ) -> Generator[Tuple[str, str, float, str]]:
        """Connects nodes if they are linked by chromatin loops. Can specify if
        the loops should only be done for direct overlaps or if they should
        be within 2mb of a loop anchor for TSS. If using TSS, make sure to
        specify the TSS as the second feature!

        The overlap function arg only affects the first anchor. The second
        anchor will always use direct overlap.
        """
        first_anchor_overlaps = self._overlap_groupby(self.first_anchor, first_feature)
        second_anchor_overlaps = self._overlap_groupby(
            self.second_anchor, second_feature
        )
        second_anchor_overlaps = self._reverse_anchors(second_anchor_overlaps)

        # if TSS, set TSS bool
        tss = self.first_anchor is self.tss or self.second_anchor is self.tss

        # get edges and write to file
        return self._write_loop_edges(
            edges_df=self._generate_edge_combinations(
                df1=first_anchor_overlaps.to_dataframe(),
                df2=second_anchor_overlaps.to_dataframe(),
                edge_type=edge_type,
            ),
            file_path=f"{self.interaction_dir}/interaction_edges.txt",
            tss=tss,
        )

    def _prepare_regulatory_elements(self):
        """Simple wrapper to load regulatory elements and return BedTools"""
        all_enhancers = pybedtools.BedTool(
            f"{self.local_dir}/{self.shared['enhancers']}"
        )
        distal_enhancers = all_enhancers.filter(lambda x: x[3] == "dELS").saveas()
        promoters = pybedtools.BedTool(f"{self.local_dir}/{self.shared['promoters']}")
        dyadic = pybedtools.BedTool(f"{self.local_dir}/{self.shared['dyadic']}")
        return all_enhancers, distal_enhancers, promoters, dyadic

    def _load_tss(self) -> pybedtools.BedTool:
        """Load TSS file and ignore any TSS that do not have a gene target.
        Additionally, adds a slop of 2kb to the TSS for downstream overlap.

        Returns:
            pybedtools.BedTool - TSS w/ target genes
        """
        tss = pybedtools.BedTool(f"{self.tss}")
        return (
            tss.filter(lambda x: x[3].split("_")[3] != "")
            .slop(g=self.chromfile, b=2000)
            .saveas()
        )

    @utils.time_dectorator(print_args=True)
    def _parse_chromloop_basegraph(self, gene_gene: bool = False) -> None:
        """Performs overlaps and write edges for regulatory features connected
        by chromatin loops. For gene overlaps, anchors are connected if anchors
        are within 2kb of the TSS. For regulatory features, anchors are
        connected if they directly overlap.

        Optional boolian 'gene_gene' specifies if gene_gene interactions will be
        parsed. If not, only gene_regulatory interactions will be parsed.
        Defaults to no.

        tss vs de, p , d
        p vs e, d, p
        e vs e, d
        """
        # Load elements
        all_enhancers, distal_enhancers, promoters, dyadic = (
            self._prepare_regulatory_elements()
        )
        self.tss = self._load_tss()
        self.first_anchor, self.second_anchor = self._split_chromatin_loops(
            self.loop_file
        )

        # Helpers - types of elements connected by loops
        overlaps = {
            self.tss: (distal_enhancers, promoters, dyadic),
            promoters: (all_enhancers, dyadic, promoters),
            all_enhancers: (all_enhancers, dyadic),
        }
        overlaps = [
            (self.tss, distal_enhancers, "g_de"),
            (self.tss, promoters, "g_p"),
            (self.tss, dyadic, "g_d"),
            (promoters, all_enhancers, "p_e"),
            (promoters, dyadic, "p_d"),
            (promoters, promoters, "p_p"),
            (all_enhancers, all_enhancers, "e_e"),
            (all_enhancers, dyadic, "e_d"),
        ]

        if "superenhancers" in self.interaction_types:
            super_enhancers = pybedtools.BedTool(
                f"{self.local_dir}/superenhancers_{self.tissue}.bed"
            )
            overlaps += [
                (self.tss, super_enhancers, "g_se"),
                (promoters, super_enhancers, "p_se"),
                (dyadic, super_enhancers, "d_se"),
                (all_enhancers, super_enhancers, "e_se"),
                (super_enhancers, super_enhancers, "se_se"),
            ]

        # Add gene_gene interactions if specified
        if gene_gene:
            overlaps.append((self.tss, self.tss, "g_g"))

        # perform two sets of overlaps
        basenodes = set()
        for first_feature, second_feature, edge_type in overlaps:
            basenodes |= self._process_loop_edges(
                first_feature=first_feature,
                second_feature=second_feature,
                edge_type=edge_type,
            ) | self._process_loop_edges(
                first_feature=second_feature,
                second_feature=first_feature,
                edge_type=edge_type,
            )

        for _, _, edge_type in overlaps:
            print(f"Processed chrom_loop {edge_type} edges")

        return basenodes

    @utils.time_dectorator(print_args=True)
    def parse_edges(self) -> None:
        """This method parses edges from various sources and constructs the
        interaction base graph. It then adds coordinates to nodes and writes the
        nodes and edges to files.

        Returns:
            None
        """
        print("Parsing interaction edges...")
        self._process_interaction_edges()
        print("Interaction edges complete!")

        print("Parsing chrom loop edges...")
        basenodes = set()
        self._parse_chromloop_basegraphs(gene_gene=self.gene_gene)
        print("Chrom loop edges complete!")

        print("Writing node references...")
        for node in basenodes:
            self._write_noderef_combination(node)
        print("Node references complete!")

    @staticmethod
    def _generate_edge_combinations(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        edge_type: str = "",
    ) -> pd.DataFrame:
        """Merging two DataFrames, generating edge combinations, and writing to
        a file."""
        # merge df
        df = pd.merge(
            df1,
            df2,
            on=["chrom", "start", "end", "name", "score", "strand"],
            how="outer",
        ).dropna()

        # get anchored regulatory features
        df["thickStart_x"] = df["thickStart_x"].str.split(",")
        df["thickStart_y"] = df["thickStart_y"].str.split(",")

        # explode and remove self loops
        df = (
            df.explode("thickStart_x")
            .explode("thickStart_y")
            .query("thickStart_x != thickStart_y")
        )

        return (
            df[["thickStart_x", "thickStart_y"]]
            .drop_duplicates()
            .rename(columns={"thickStart_x": "edge_1", "thickStart_y": "edge_2"})
            .assign(type=edge_type)
        )

    @staticmethod
    def _split_chromatin_loops(
        chromatin_loops: str,
    ) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
        """Split chromatin loops into two separate BedTools.

        Args:
            chromatin_loops (str): The path to the chromatin loops file.

        Returns:
            Tuple[pybedtools.BedTool, pybedtools.BedTool]: The first and
            second anchor BedTools.
        """
        first_anchor = pybedtools.BedTool(chromatin_loops).sort()
        second_anchor = first_anchor.cut([3, 4, 5, 0, 1, 2]).sort()
        return first_anchor.cut([0, 1, 2, 3, 4, 5]), second_anchor

    @staticmethod
    def _reverse_anchors(bed: pybedtools.BedTool) -> pybedtools.BedTool:
        """Reverse the anchors of a BedTool, only meant to be used after overlaps
        are added as it returns seven columns"""
        return bed.cut([3, 4, 5, 0, 1, 2, 6])

    @staticmethod
    def _add_feat_names(feature: str) -> str:
        """Add feature name in the format of chr_start_type"""
        feature[6] = f"{feature[6]}_{feature[7]}_{feature[9]}"
        return feature

    @staticmethod
    def _loop_direct_overlap(
        loops: pybedtools.BedTool, features: pybedtools.BedTool
    ) -> pybedtools.BedTool:
        """Get features that directly overlap with loop anchor"""
        return loops.intersect(features, wo=True, stream=True)

    # @staticmethod
    # def _loop_within_distance(
    #     loops: pybedtools.BedTool,
    #     features: pybedtools.BedTool,
    #     distance: int = 2000,
    # ) -> pybedtools.BedTool:
    #     """Get features at specified distance to loop anchor. Defaults with 2kb
    #     distance window."""
    #     return loops.window(features, w=distance, stream=True)
