#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Parse edges from interaction-type omics data"""


import contextlib
import csv
import os
from pathlib import Path
import pickle
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import pybedtools  # type: ignore

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from constants import REGULATORY_ELEMENTS
from rbp_network_filter import RBPNetworkFilter
from utils import _get_chromatin_loop_file
from utils import genes_from_gencode
from utils import time_decorator


class EdgeParser:
    """Object to construct tensor based graphs from parsed bedfiles

    Attributes:
        experiment_config: Dataclass containing the experiment configuration.
        tissue_config: Dataclass containing the tissue configuration.

    Methods
    --------
    _initialize_directories_and_vars:
        Initialize directory paths.
    _initialize_references:
        Initialize reference dictionaries.
    _create_reference_dict:
        Reads a file and stores its lines in a dictionary.
    _mirna_targets:
        Filters all miRNA -> target interactions from miRTarBase and only keeps
        the miRNAs that are active in the given tissue from mirDIP.
    _tf_markers:
        Filters TF markers based on specific conditions.
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

    Examples:
    --------
    >>> edgeparserObject = EdgeParser(
            experiment_config=experiment_config,
            tissue_config=tissue_config,
        )

    >>> edgeparserObject.parse_edges()

    DEPRECATED:
        _iid_ppi
        _marbach_regulatory_circuits:
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tissue_config: TissueConfig,
        # loop_file: str,
    ):
        """Initialize the class"""
        self.experiment_config = experiment_config
        self.tissue_config = tissue_config

        self.experiment_name = experiment_config.experiment_name
        self.interaction_types = experiment_config.interaction_types
        self.gene_gene = experiment_config.gene_gene
        self.root_dir = experiment_config.root_dir
        self.attribute_references = experiment_config.attribute_references
        self.regulatory_schema = experiment_config.regulatory_schema

        self.loop_file = _get_chromatin_loop_file(
            experiment_config=experiment_config, tissue_config=tissue_config
        )

        self.tf_extension = ""
        if experiment_config.differentiate_tf == True:
            self.tf_extension += "_tf"

        self._initialize_directories_and_vars()
        self._initialize_references()

    def _initialize_directories_and_vars(self) -> None:
        """Initialize directory paths"""
        self.gencode = self.tissue_config.local["gencode"]
        self.interaction_files = self.tissue_config.interaction
        self.blacklist_file = self.tissue_config.resources["blacklist"]
        self.tissue = self.tissue_config.resources["tissue"]
        self.chromfile = self.tissue_config.resources["chromfile"]
        self.marker_name = self.tissue_config.resources["marker_name"]
        self.ppi_tissue = self.tissue_config.resources["ppi_tissue"]

        self.shared_dir = self.root_dir / "shared_data"
        self.tissue_dir = self.root_dir / self.experiment_name / self.tissue
        self.local_dir = self.tissue_dir / "local"
        self.parse_dir = self.tissue_dir / "parsing"
        self.interaction_dir = self.tissue_dir / "interaction"
        self.shared_interaction_dir = self.shared_dir / "interaction"
        self.tss = self.local_dir / self.tissue_config.local["tss"]

    def _initialize_references(self) -> None:
        """Initialize reference dictionaries"""
        self.blacklist = pybedtools.BedTool(f"{self.blacklist_file}").sort().saveas()
        self.gencode_ref = pybedtools.BedTool(
            self.local_dir / self.tissue_config.local["gencode"]
        )
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_ref = self.gencode_ref.cut([0, 1, 2, 3]).saveas()
        self.gencode_attr_ref = self._create_reference_dict(
            self.attribute_references["gencode"]
        )
        self.regulatory_attr_ref = self._create_reference_dict(
            self.attribute_references["regulatory_elements"]
        )
        self.se_ref = self._create_reference_dict(
            self.attribute_references["super_enhancers"]
        )
        self.mirna_ref = self._create_reference_dict(
            self.attribute_references["mirna"], mirna=True
        )
        self.footprint_ref = self._create_reference_dict(
            self.attribute_references["tf_motifs"]
        )

    def _create_reference_dict(
        self, file: str, mirna: bool = False
    ) -> Dict[str, List[str]]:
        """Reads a file and stores its lines in a dictionary"""
        try:
            key_idx = 4 if mirna else 3
            return {
                line[key_idx]: line[:4]
                for line in csv.reader(open(file, newline=""), delimiter="\t")
            }
        except FileNotFoundError:
            return {}

    def _read_csv_wrapper(self, file_path: str) -> Iterator[List[str]]:
        """Wrapper function to read a CSV file."""
        with open(file_path, newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            return reader

    def _remove_blacklist(self, bed: pybedtools.BedTool) -> pybedtools.BedTool:
        """Remove blacklist regions from a BedTool object."""
        return bed.intersect(self.blacklist, v=True, sorted=True)

    def _create_bedtool(self, path: Path) -> Union[pybedtools.BedTool, None]:
        """Create a BedTool object if the file exists."""
        return (
            self._remove_blacklist(pybedtools.BedTool(path))
            if os.path.exists(path)
            else None
        )

    def _prepare_regulatory_elements(
        self,
    ) -> Tuple[pybedtools.BedTool, pybedtools.BedTool, Optional[pybedtools.BedTool]]:
        """Simple wrapper to load regulatory elements and return BedTools"""
        reg_elements = REGULATORY_ELEMENTS[self.regulatory_schema]
        bedtools_objects = {
            key: self._create_bedtool(self.local_dir / f"{key}_{self.tissue}.bed")
            for key in reg_elements
            if key in ["enhancers", "promoters", "dyadic"]
        }

        if self.regulatory_schema == "encode":
            bedtools_objects["dyadic"] = None

        return (
            bedtools_objects["enhancers"],
            bedtools_objects["promoters"],
            bedtools_objects["dyadic"],
        )

        # Code for filtering distal enhancers. Ignoring for now.
        # if self.regulatory_schema in ["encode", "intersect"]:
        #     bedtools_objects["distal_enhancers"] = (
        #         bedtools_objects["enhancers"].filter(lambda x: x[3] == "dELS").saveas()
        #     )
        # else:
        #     bedtools_objects["distal_enhancers"] = None

        # return (
        #     bedtools_objects["enhancers"],
        #     bedtools_objects["distal_enhancers"],
        #     bedtools_objects["promoters"],
        #     bedtools_objects["dyadic"],
        # )

    @time_decorator(print_args=True)
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
        basenodes = self._parse_chromloop_basegraph(gene_gene=self.gene_gene)
        print("Chrom loop edges complete!")

        print("Writing node references...")
        for node in basenodes:
            self._write_noderef_combination(node)
        print("Node references complete!")

    def _rbp_network(
        self,
        tpm_filter: int = 10,
        # ) -> Generator[Tuple[str, str, float, str], None, None]:
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Filters RBP interactions based on tpm filter, derived from POSTAR3"""
        rbp_network_obj = RBPNetworkFilter(
            rbp_proteins=self.experiment_config.rbp_proteins,
            gencode=self.gencode,
            network_file=self.experiment_config.rbp_network,
            tpm_filter=tpm_filter,
            rna_seq_file=self.tissue_config.resources["rna"],
        )
        rbp_network_obj.filter_rbp_network()

        # save gene list to file
        with open(self.interaction_dir / "active_rbps.pkl", "wb") as file:
            pickle.dump(rbp_network_obj.active_rbps, file)

        for tup in rbp_network_obj.filtered_network:
            yield (
                tup[0],
                tup[1],
                # -1,
                "rbp",
            )

    def _mirna_targets(
        self,
        target_list: str,
        tissue_active_mirnas: str,
        # ) -> Generator[Tuple[str, str, float, str], None, None]:
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Filters all miRNA -> target interactions from miRTarBase and based on
        the active miRNAs in the given tissue.
        """
        with open(tissue_active_mirnas, newline="") as file:
            active_mirna = set(csv.reader(file, delimiter="\t"))

        with open(target_list, newline="") as file:
            target_reader = csv.reader(file, delimiter="\t")
            for line in target_reader:
                if line[0] in active_mirna and line[1] in self.genesymbol_to_gencode:
                    yield (
                        line[0],
                        self.genesymbol_to_gencode[line[1]],
                        # -1,
                        "mirna",
                    )

    def _tf_markers(
        self,
        interaction_file: str,
        # ) -> Generator[Tuple[str, str, float, str], None, None]:
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Filters tf markers based on specified conditions.

        Args:
            interaction_file (str): The path to the interaction file.

        Returns:
            List[Tuple[str, str]]: A list of filtered tf marker interactions.
        """
        tf_keep = ["TF", "I Marker", "TFMarker"]
        with open(interaction_file, newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)  # Skip header

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
                    f"{self.genesymbol_to_gencode[tup[0]]}{self.tf_extension}",
                    self.genesymbol_to_gencode[tup[1]],
                    # -1,
                    "tf_marker",
                )

    # def _marbach_regulatory_circuits(
    #     self,
    #     interaction_file: str,
    #     score_filter: int,
    # ) -> Generator[Tuple[str, str, float, str], None, None]:
    #     """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
    #     network is in the following format:
    #         col_1   TF
    #         col_2   Target gene
    #         col_3   Edge weight
    #     """
    #     reader = self._read_csv_wrapper(interaction_file)

    #     scores = [
    #         float(line[2])
    #         for line in reader
    #         if (
    #             line[0] in self.genesymbol_to_gencode
    #             and line[1] in self.genesymbol_to_gencode
    #         )
    #     ]
    #     cutoff = np.percentile(scores, score_filter)

    #     reader = self._read_csv_wrapper(interaction_file)  # Re-read the file
    #     for line in reader:
    #         tf, target_gene, weight = line[0], line[1], float(line[2])
    #         if (
    #             tf in self.genesymbol_to_gencode
    #             and target_gene in self.genesymbol_to_gencode
    #             and weight >= cutoff
    #         ):
    #             yield (
    #                 f"{self.genesymbol_to_gencode[tf]}_tf",
    #                 self.genesymbol_to_gencode[target_gene],
    #                 # weight,
    #                 "circuits",
    #             )

    # def _iid_ppi(
    #     self,
    #     interaction_file: str,
    #     tissue: str,
    #     # ) -> Generator[Tuple[str, str, float, str], None, None]:
    # ) -> Generator[Tuple[str, str, str], None, None]:
    #     """Protein-protein interactions from the Integrated Interactions
    #     Database v 2021-05"""
    #     reader = self._read_csv_wrapper(interaction_file)
    #     header = next(reader)
    #     idxs = {
    #         key: header.index(key)
    #         for key in ["symbol1", "symbol2", "evidence_type", "n_methods", tissue]
    #     }

    #     for line in reader:
    #         if (
    #             float(line[idxs[tissue]]) > 0
    #             and int(line[idxs["n_methods"]]) >= 3
    #             and "exp" in line[idxs["evidence_type"]]
    #             and line[idxs["symbol1"]] in self.genesymbol_to_gencode
    #             and line[idxs["symbol2"]] in self.genesymbol_to_gencode
    #         ):
    #             yield (
    #                 self.genesymbol_to_gencode[line[idxs["symbol1"]]],
    #                 self.genesymbol_to_gencode[line[idxs["symbol2"]]],
    #                 # -1,
    #                 "ppi",
    #             )

    def _tfbinding_footprints(
        self,
        tfbinding_file: str,
        footprint_file: str,
        # ) -> Generator[Tuple[str, str, float, str], None, None]:
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Create edges based on whether or not known TF binding from Meuleman
        et al. overlap footprints from Vierstra et al.

        Args:
            tfbinding_file (str): shared tf binding locations from Meuleman
            footprint_file (str): tissue-specific footprints from Vierstra

        Returns:
            List[Tuple[str, str, float, str]]
        """
        tf_binding = self._remove_blacklist(pybedtools.BedTool(tfbinding_file))
        tf_edges = tf_binding.intersect(footprint_file, wb=True)
        for line in tf_edges:
            yield (
                f"{self.genesymbol_to_gencode[line[3]]}_tf",
                f"{line[5]}_{line[6]}_{line[3]}",
                # -1,
                "tf_binding_footprint",
            )

    # def _check_tss_gene_in_gencode(self, tss: str) -> bool:
    #     """Check if gene associated with TSS is in gencode v26"""
    #     gene = tss.split("_")[5]
    #     return self.genesymbol_to_gencode.get(gene, False)

    def _write_noderef_combination(self, node: str) -> None:
        """Writes chr, start, stop, node to a file. Gets coords from ref
        dict."""
        if "ENSG" in node:
            self._write_node_list(
                self._add_node_coordinates(node, self.gencode_attr_ref)
            )
        elif "superenhancer" in node:
            self._write_node_list(self._add_node_coordinates(node, self.se_ref))
        else:
            self._write_node_list(
                self._add_node_coordinates(node, self.regulatory_attr_ref)
            )

    def _write_node_list(self, node: List[str]) -> None:
        """Write gencode nodes to file"""
        with open(self.local_dir / "basenodes_hg38.txt", "a") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(node)

    def _write_edges(self, edge: Tuple[Union[str, int]]) -> None:
        """Write edge to file"""
        with open(self.interaction_dir / "interaction_edges.txt", "a") as output:
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(edge)

    def _prepare_interaction_generators(self):
        """Initiate interaction type generators"""
        ppi_generator = mirna_generator = tf_generator = circuit_generator = (
            tfbinding_generator
        ) = iter([])

        if "mirna" in self.interaction_types:
            mirna_generator = self._mirna_targets(
                target_list=self.interaction_dir / "active_mirna_{self.tissue}.txt",
                tissue_active_mirnas=self.interaction_dir
                / self.interaction_files["mirdip"],
            )
        if "tf_marker" in self.interaction_types:
            tf_generator = self._tf_markers(
                interaction_file=self.interaction_dir
                / self.interaction_files["tf_marker"],
            )
        if "tfbinding" in self.interaction_types:
            tfbinding_generator = self._tfbinding_footprints(
                tfbinding_file=self.shared_interaction_dir
                / self.interaction_files["tfbinding"],
                footprint_file=self.local_dir / self.shared["footprints"],
            )
        # if "ppis" in self.interaction_types:
        #     ppi_generator = self._iid_ppi(
        #         interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
        #         tissue=self.ppi_tissue,
        #     )
        # if "circuits" in self.interaction_types:
        #     circuit_generator = self._marbach_regulatory_circuits(
        #         f"{self.interaction_dir}/{self.interaction_files['circuits']}",
        #         score_filter=30,
        #     )

        return (
            # ppi_generator,
            # mirna_generator,
            tf_generator,
            circuit_generator,
            tfbinding_generator,
        )

    def _add_node_coordinates(
        self,
        node: str,
        node_ref: Dict[str, List[str]],
    ) -> List[str]:
        """Add coordinates to nodes based on the given node reference."""
        return node_ref[node]

    def _run_generator_common(
        self,
        generator: Generator,
        attr_refs: List[Dict[str, List[str]]],
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

    def _check_if_interactions_exists(self) -> bool:
        """Check if interaction edges file exists"""
        return bool(self.interaction_types)

    @time_decorator(print_args=True)
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

        if self._check_if_interactions_exists():
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
        overlap_func: Callable,
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
        file_path: Path,
        tss=False,
    ) -> Set[str]:
        """Write the edges to a file in bulk."""
        if tss:
            for col in ["edge_0", "edge_1"]:
                mask = edges_df[col].str.contains("ENSG")
                edges_df.loc[mask, col] = edges_df.loc[mask, col].str.split("_").str[-1]
        edges_df.to_csv(file_path, sep="\t", mode="a", header=False, index=False)
        return set(pd.concat([edges_df["edge_0"], edges_df["edge_1"]]).unique())

    @time_decorator(print_args=True)
    def _process_loop_edges(
        self,
        first_feature: pybedtools.BedTool,
        second_feature: pybedtools.BedTool,
        edge_type: str,
    ) -> Set[str]:
        """Connects nodes if they are linked by chromatin loops. Can specify if
        the loops should only be done for direct overlaps or if they should
        be within 2mb of a loop anchor for TSS. If using TSS, make sure to
        specify the TSS as the second feature!

        The overlap function arg only affects the first anchor. The second
        anchor will always use direct overlap.
        """
        # only perform overlaps if file exists
        if not first_feature or not second_feature:
            return set()

        first_anchor_overlaps = self._overlap_groupby(
            self.first_anchor, first_feature, self._loop_direct_overlap
        )
        second_anchor_overlaps = self._overlap_groupby(
            self.second_anchor, second_feature, self._loop_direct_overlap
        )
        second_anchor_overlaps = self._reverse_anchors(second_anchor_overlaps)

        # Set TSS bool if either first_feature or second_feature matches
        tss = first_feature in [self.tss, self.gencode_ref] or second_feature in [
            self.tss,
            self.gencode_ref,
        ]

        # get edges and write to file
        return self._write_loop_edges(
            edges_df=self._generate_edge_combinations(
                df1=first_anchor_overlaps.to_dataframe(),
                df2=second_anchor_overlaps.to_dataframe(),
                edge_type=edge_type,
            ),
            file_path=self.interaction_dir / "interaction_edges.txt",
            tss=tss,
        )

    def _add_slop_window(
        self, bed: pybedtools.BedTool, window: int
    ) -> pybedtools.BedTool:
        """Return a slopped BedTool object"""
        return bed.slop(g=self.chromfile, b=window)

    def _load_tss(self) -> pybedtools.BedTool:
        """Load TSS file and ignore any TSS that do not have a gene target.
        Additionally, adds a slop of 2kb to the TSS for downstream overlap.
        Ignores non-gene TSS by checking the length of the tss name:
        gene-associated tss are annotated with one extra field.

        Returns:
            pybedtools.BedTool - TSS w/ target genes
        """

        def _gene_only_tss(feature):
            if len(feature[3].split("_")) > 3:
                feature[3] = feature[3].split("_")[3]
                return feature
            return None

        tss = pybedtools.BedTool(f"{self.tss}")
        return self._add_slop_window(tss.filter(_gene_only_tss), 2000)

    def _process_overlaps(
        self, overlaps: List[Tuple[pybedtools.BedTool, pybedtools.BedTool, str]]
    ) -> Set[str]:
        """Process overlaps per type of regulatory connection and generate loop edges."""
        basenodes = set()
        for first_feature, second_feature, edge_type in overlaps:
            basenodes |= self._process_loop_edges(
                first_feature, second_feature, edge_type
            ) | self._process_loop_edges(second_feature, first_feature, edge_type)
            print(f"Processed chrom_loop {edge_type} edges")
        return basenodes

    @time_decorator(print_args=True)
    def _parse_chromloop_basegraph(self, gene_gene: bool = False) -> Set[str]:
        """Performs overlaps and write edges for regulatory features connected
        by chromatin loops. For gene overlaps, anchors are connected if anchors
        are within 2kb of the TSS. For regulatory features, anchors are
        connected if they directly overlap.

        Optional boolian 'gene_gene' specifies if gene_gene interactions will be
        parsed. If not, only gene_regulatory interactions will be parsed.
        Defaults to False.

        tss vs de, p , d
        p vs e, d, p
        e vs e, d
        """
        # Load elements
        all_enhancers, promoters, dyadic = self._prepare_regulatory_elements()
        self.tss = self._load_tss()
        self.first_anchor, self.second_anchor = self._split_chromatin_loops(
            self.loop_file
        )

        overlaps = [
            (self.tss, all_enhancers, "g_e"),
            (self.tss, promoters, "g_p"),
            (self.tss, dyadic, "g_d"),
            (promoters, all_enhancers, "p_e"),
            (promoters, dyadic, "p_d"),
            (promoters, promoters, "p_p"),
            (all_enhancers, all_enhancers, "e_e"),
            (all_enhancers, dyadic, "e_d"),
        ]

        with contextlib.suppress(TypeError):
            if "superenhancers" in self.interaction_types:
                super_enhancers = pybedtools.BedTool(
                    self.local_dir / "superenhancers_{self.tissue}.bed"
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
            gencode_2kb = self._add_slop_window(self.gencode_ref, 2000)
            overlaps.append((gencode_2kb, gencode_2kb, "g_g"))

        # perform two sets of overlaps
        return self._process_overlaps(overlaps)

    @staticmethod
    def _generate_edge_combinations(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        edge_type: str = "",
    ) -> pd.DataFrame:
        """Merging two DataFrames, generating edge combinations, and writing to
        a file."""
        # merge df
        df = (
            pd.merge(
                df1,
                df2,
                on=["chrom", "start", "end", "name", "score", "strand"],
                how="outer",
            )
            .rename(columns={"thickStart_x": "edge_0", "thickStart_y": "edge_1"})
            .dropna()
        )

        # get anchored regulatory features
        df["edge_0"] = df["edge_0"].str.split(",")
        df["edge_1"] = df["edge_1"].str.split(",")

        # explode and remove self loops
        df = df.explode("edge_0").explode("edge_1").query("edge_0 != edge_1")
        return df[["edge_0", "edge_1"]].drop_duplicates().assign(type=edge_type)

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
        feature[6] = f"{feature[6]}_{feature[7]}_{feature[9]}"  # type: ignore
        return feature

    @staticmethod
    def _loop_direct_overlap(
        loops: pybedtools.BedTool, features: pybedtools.BedTool
    ) -> pybedtools.BedTool:
        """Get features that directly overlap with loop anchor"""
        return loops.intersect(features, wo=True, stream=True)
