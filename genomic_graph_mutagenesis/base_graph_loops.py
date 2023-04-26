#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Create a base graph from chromatin loop data.

We filter for distal ELSs and link them to other enhancers and promoters (loop
overlap) or genes (within 2kb of a loop anchor)."""

from typing import Dict, List, Tuple

import pybedtools


def _split_chromatin_loops(
    loop_path: str,
) -> Tuple[pybedtools.BedTool, pybedtools.BedTool]:
    """_summary_ of function"""
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
    """_summary_ of function"""
    return loops.window(tss, w=distance)


def _flatten_anchors(*beds) -> Dict[str, List[str]]:
    """Creates"""
    anchor = {}
    for bed in beds:
        for feature in bed:
            anchor.setdefault("_".join(feature[0:3]), []).append(
                "_".join(feature[7:13])
            )
    return anchor


def _loop_connections_overlap(bed_one, bed_two) -> None:
    """Return a list of edges that are connected by their overlap over chromatin
    loop anchors

    First we flatten the anchors to keep all of their overlaps in a list. Then,
    we create tuples from one feature to another based on the corresponding
    anchor point list."""


def main() -> None:
    """Main function"""
    enhancer_path = (
        "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/GRCh38-ELS.bed"
    )
    loop_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/liver/Leung_2015.Liver.hg38.peakachu-merged.loops"
    tss_path = "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/reftss/reftss_annotated.bed"

    enhancers = pybedtools.BedTool(enhancer_path)
    first_anchor, second_anchor = _split_chromatin_loops(loop_path)

    anchor_one_overlaps = _loop_direct_overlap(first_anchor, enhancers)
    anchor_two_overlaps = _loop_direct_overlap(second_anchor, enhancers)
    anchor_one_tss = _loop_within_distance(first_anchor, tss_path, 2000)
    anchor_two_tss = _loop_within_distance(second_anchor, tss_path, 2000)


if __name__ == "__main__":
    main()


# chr17	55520000	55530000	chr17	55710000	55720000	0.8337972452258688	chr17	55521213	55521552	EH38D5024294	EH38E3231805	dELS	339
# chr17	55710000	55720000	55520000	55530000	chr17	chr17	55710038	55710344	EH38D5024414	EH38E3231894dELS	306
# chr17	64010000	64020000	chr17	64130000	64140000	0.8159749505187259	chr17	64020220	64020241	tss_hg_162245.1_
# chr17	64130000	64140000	64010000	64020000	chr17	chr17	64130030	64130055	tss_hg_167554.1_ERN1


# [6:10]
