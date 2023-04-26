#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# / TODO /
# - [ ] properly instatiate the tss, and remove the hardcode name for tss or enhancers and use feats
# - [ ] get active enhancers from FENRIR
# - [ ] only keep TSS with gene targets
# - [ ] adjust the edges output to resembl that from edge_parser.py
# - [ ] figure out why product returns key error. need to check for empty keys


"""Create a base graph from chromatin loop data.

We filter for distal ELSs and link them to other enhancers and promoters (loop
overlap) or genes (within 2kb of a loop anchor)."""

import itertools
from typing import Dict, List, Tuple

import pybedtools


def _load_tss(tss_path: str) -> pybedtools.BedTool:
    """Load TSS file and ignore any TSS that do not have a gene target. 
    Returns:
        pybedtools.BedTool - TSS w/ target genes
    """
    tss = pybedtools.BedTool(tss_path)
    return tss.filter(lambda x: x[3].split('_')[3] != '').saveas()


def _split_chromatin_loops(
    loop_path: str,
) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
    """_summary_

    Args:
        loop_path (str): _description_

    Returns:
        Tuple[pybedtools.BedTool, pybedtools.BedTool]: _description_
    """
    first_anchor = pybedtools.BedTool(loop_path)
    second_anchor = pybedtools.BedTool(
        "\n".join(
            ["\t".join([x[3], x[4], x[5], x[1], x[2], x[3]]) for x in first_anchor]
        ),
        from_string=True,
    )
    return first_anchor, second_anchor


def _loop_direct_overlap(
    loops: pybedtools.BedTool, enhancers: pybedtools.BedTool
) -> pybedtools.BedTool:
    """_summary_ of function"""
    return loops.intersect(enhancers, wo=True)


def _loop_within_distance(
    loops: pybedtools.BedTool,
    tss: pybedtools.BedTool,
    distance: int,
) -> pybedtools.BedTool:
    """_summary_

    Args:
        loops (pybedtools.BedTool): _description_
        tss (pybedtools.BedTool): _description_
        distance (int): _description_

    Returns:
        pybedtools.BedTool: _description_
    """
    return loops.window(tss, w=distance)


def _flatten_anchors(*beds: pybedtools.BedTool) -> Dict[str, List[str]]:
    """Creates a dict to store each anchor and its overlaps. Adds the feature by
    ignoring the first 7 columns of the bed file and adding whatever is left."""
    anchor = {}
    for bed in beds:
        for feature in bed:
            anchor.setdefault("_".join(feature[0:3]), []).append("_".join(feature[7:]))
    return anchor


def _loop_edges(
    loops: pybedtools.BedTool,
    first_anchor_edges: Dict[str, List[str]],
    second_anchor_edges: Dict[str, List[str]],
) -> None:
    """Return a list of edges that are connected by their overlap over chromatin
    loop anchors by matching the anchor names across dicts"""
    edges = []
    for loop in loops:
        first_anchor = "_".join(loop[0:3])
        second_anchor = "_".join(loop[3:6])
        try:
            uniq_edges = list(
                itertools.product(
                    first_anchor_edges[first_anchor], second_anchor_edges[second_anchor]
                )
            )
            edges.extend(uniq_edges)
        except KeyError:
            continue
    return edges


def main() -> None:
    """Main function"""
    enhancer_path = (
        "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/GRCh38-ELS.bed"
    )
    loop_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/liver/Leung_2015.Liver.hg38.peakachu-merged.loops"
    tss_path = "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss/reftss_annotated.bed"

    enhancers = pybedtools.BedTool(enhancer_path)
    tss = _load_tss(tss_path)
    first_anchor, second_anchor = _split_chromatin_loops(loop_path)

    anchor_one_overlaps = _loop_direct_overlap(first_anchor, enhancers)
    anchor_two_overlaps = _loop_direct_overlap(second_anchor, enhancers)
    anchor_one_tss = _loop_within_distance(first_anchor, tss, 2000)
    anchor_two_tss = _loop_within_distance(second_anchor, tss, 2000)

    first_anchor_edges = _flatten_anchors(anchor_one_overlaps, anchor_one_tss)
    second_anchor_edges = _flatten_anchors(anchor_two_overlaps, anchor_two_tss)

    edges = _loop_edges(
        first_anchor, first_anchor_edges, second_anchor_edges
    )

if __name__ == "__main__":
    main()


# chr17	55520000	55530000	chr17	55710000	55720000	0.8337972452258688	chr17	55521213	55521552	EH38D5024294	EH38E3231805	dELS	339
# chr17	55710000	55720000	55520000	55530000	chr17	chr17	55710038	55710344	EH38D5024414	EH38E3231894dELS	306
# chr17	64010000	64020000	chr17	64130000	64140000	0.8159749505187259	chr17	64020220	64020241	tss_hg_162245.1_
# chr17	64130000	64140000	64010000	64020000	chr17	chr17	64130030	64130055	tss_hg_167554.1_ERN1


# [6:10]
