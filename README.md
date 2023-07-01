# Genomic Graph Mutagenesis
Tools to construct graphs heterogenous multi-omics data and train a GNN to regress values of gene expression and protein abundance. Graphs are mutagenized to query the impact of individual features on biological function.
&nbsp;

<div align="center">
    <img src='docs/_static/placeholder.png'>
</div>

## Description
    The following features have node representations:
        Tissue-specific
            ENCODE cCRE Enhancers (fenrir)
            Genes (GENCODE, PPI interactions from IID)
            TFs (Marbach, TFMarker)
            MicroRNAs (mirDIP for tissue-specific, miRTarBase for interactions)

            Chromatinloops
            Histone binding clusters (collapsed)
            Transcription factor binding sites 
            TADs
            Super-enhancers (sedb)

        Genome-static
            Cpgislands
            Gencode (genes)
            Promoters (encode cCRE)
            CTCF-cCRE (encode cCRE)
            Transcription start sites


    The following are represented as attributes:
        Tissue-specific
            CpG methylation

            ChIP-seq peaks
                CTCF ChIP-seq peaks
                DNase ChIP-seq peaks
                H3K27ac ChIP-seq peaks
                H3K27me3 ChIP-seq peaks
                H3K36me3 ChIP-seq peaks
                H3K4me1 ChIP-seq peaks
                H3K4me3 ChIP-seq peaks
                H3K9me3 ChIP-seq peaks
                POLR2a ChIP-seq peaks

        Genome-static
            GC content
            Microsatellites
            Conservation (phastcons)
            Poly(a) binding sites (overlap)
            LINEs (overlap)
            Long terminal repeats (overlap)
            Simple repeats (overlap)
            SINEs (overlap)
            Hotspots
                snps
                indels
                cnvs 
            miRNA target sites
            RNA binding protein binding sites
            Replication phase
                g1b
                g2
                s1
                s3
                s4
                s4
            Recombination rate (averaged)


Working tissues:
    HeLa
    Hippocampus
    K562
    Left ventricle
    Liver
    Lung
    Mammary
    Neural progenitor cell
    Pancreas
    Skeletal muscle
    Skin
    Small intestine

&nbsp;

## Installation

```sh
$ git clone https://github.com/sciencesteveho/genomic_graph_mutagenesis.git
```

## Dependencies

```sh
$ lorem ipsum
```
&nbsp;

## Usage


Note: not all arguments are compatible with one another, so see examples below for the program's capabilities.
```sh
# Convert epimap bigwig files to broad and narrow peaks
for tissue in hela hippocampus k562 left_ventricle liver lung mammary npc pancreas skeletal_muscle skin small_intestine;
do
    merge_epimap.sh $tissue
done

# Add chromatin loops together
$ sh chrom_loops_basefiles.sh

# Preparse bedfiles
$ sh preparse.sh

# Run python scripts
$ python -u genomic_graph_mutagenesis/prepare_bedfiles.py --config ${yaml}
$ python -u genomic_graph_mutagenesis/edge_parser.py --config ${yaml}
$ python -u genomic_graph_mutagenesis/local_context_parser.py --config ${yaml}
$ python -u genomic_graph_mutagenesis/graph_constructor.py --config ${yaml}
```
