#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Create a base graph from chromatin loop data.

We filter for distal ELSs and link them to other enhancers and promoters (loop
overlap) or genes (within 2kb of a loop anchor)."""

from typing import List, Tuple

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


def _direct_overlap_els(
    loops: pybedtools.BedTool, enhancers: pybedtools.BedTool
) -> pybedtools.BedTool:
    """_summary_ of function"""
    return loops.intersect(enhancers, wo=True)


def _tss_within_2kb_anchor(
    loops: pybedtools.BedTool, tss: pybedtools.BedTool
) -> pybedtools.BedTool:
    """_summary_ of function"""
    return loops.window(tss, w=2000)


def main() -> None:
    """Main function"""
    enhancer_path = (
        "/ocean/projects/bio210019p/stevesho/data/bedfile_preparse/GRCh38-ELS.bed"
    )
    loop_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/liver/Leung_2015.Liver.hg38.peakachu-merged.loops"

    enhancers = pybedtools.BedTool(enhancer_path)
    first_anchor, second_anchor = _split_chromatin_loops(loop_path)

    anchor_one_overlaps = _direct_overlap_els(first_anchor, enhancers)
    anchor_two_overlaps = _direct_overlap_els(second_anchor, enhancers)
    anchor_one_tss = _tss_within_2kb_anchor(first_anchor, enhancers)
    anchor_two_tss = _tss_within_2kb_anchor(second_anchor, enhancers)


if __name__ == "__main__":
    main()


# chr17	55520000	55530000	chr17	55710000	55720000	0.8337972452258688	chr17	55521213	55521552	EH38D5024294	EH38E3231805	dELS	339
# chr17	55710000	55720000	55520000	55530000	chr17	chr17	55710038	55710344	EH38D5024414	EH38E3231894dELS	306
# chr17	55520000	55530000	chr17	55710000	55720000	0.8337972452258688	chr17	55521213	55521552	EH38D5024294	EH38E3231805	dELS
# chr17	55710000	55720000	55520000	55530000	chr17	chr17	55709535	55709778	EH38D5024413	EH38E3231893dELS