# Omics Graph Mutagenesis
Tools to construct graphs heterogenous multi-omics data and train a GNN to regress values of gene expression and protein abundance. Graphs are mutagenized to query the impact of individual features on biological function.
&nbsp;

<div align="center">
    <img src='docs/_static/placeholder.png'>
</div>

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
# First 3 steps process graphs
$ python -u genomic_graph_mutagenesis/prepare_bedfiles.py --config ${yaml}

$ python -u genomic_graph_mutagenesis/graph_constructor.py --config ${yaml}

$ python -u genomic_graph_mutagenesis/local_context_parser.py --config ${yaml}
```
