def plot_single_enrichment_after_ablation(
    node_ablations: Dict[int, List[Tuple[str, float]]],
    gencode_to_symbol: Dict[str, str],
    idx: int,
    sample: str,
    top_n: int = 100,
) -> None:
    """Get the top n genes after node feature ablation."""
    set_matplotlib_publication_parameters()
    outpath = "/Users/steveho/gnn_plots/interpretation/individual_enrichment"

    gene_sets = [
        "Reactome_Pathways_2024",
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
    ]

    # set up colors
    colors = ["#E6F3FF", "#2171B5"]  # Light to dark
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # get top_n_genes
    top = node_ablations[idx][:top_n]

    # convert to gene symbols
    top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

    # run enrichr
    for gene_set in gene_sets:
        gmt = f"/Users/steveho/gnn_plots/interpretation/gmt/{gene_set}.gmt"
        print(f"Running enrichr for {gmt}...")

        try:
            enrichr_result = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gmt,
                organism="Human",
                outdir=None,
            )
        except AttributeError:
            print(f"Error with {gene_set} for {sample}")
            continue

        # get significant terms
        significant = enrichr_result.results[
            enrichr_result.results["Adjusted P-value"] < 0.05
        ]
        df = significant.copy()
        df["neg_log_p"] = -np.log10(df["Adjusted P-value"])
        # plt.figure(figsize=(5, 1.25))

        # adjust name if GO
        # split the df "Term" column by "(GO:" and take the first part
        if "GO" in gene_set:
            df["Term"] = df["Term"].apply(lambda x: x.split("(GO:")[0])

        top5 = df.nlargest(5, "neg_log_p").copy()
        if len(top5) == 0:
            print(f"No significant terms for {gene_set} in {sample}")
            continue

        top5 = top5.sort_values("neg_log_p", ascending=False)

        color_vals = np.linspace(1, 0, len(top5))
        bar_colors = cmap(color_vals)

        ax = plt.gca()
        plt.barh(
            top5["Term"],
            top5["neg_log_p"],
            color=bar_colors,
            height=0.55,
        )

        # add significance line
        plt.axvline(
            -np.log10(0.05),
            color="black",
            linestyle="--",
            alpha=0.80,
            label="Adjusted p=0.05",
            linewidth=0.5,
            ymin=0,
            ymax=0.875,
        )

        plt.xlabel(r"-log$_{10}$ adjusted $\it{p}$")
        plt.gca().invert_yaxis()
        plt.gca().margins(y=0.15)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)

        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", width=0.5)

        plt.tight_layout()
        plt.savefig(f"{outpath}/{sample}_{FEATURES[idx]}_{gene_set}.png", dpi=450)
        plt.clf()


def collate_enrichment_results(
    gencode_to_symbol: Dict[str, str],
    idx: int,
    top_n: int = 100,
) -> None:
    """Get the top-n genes after node feature ablation and run Enrichr.
    Returns three DataFrames (Reactome, GO_BP, GO_MF) with up to top-5
    significant terms per sample (labeled by the 'Sample' column).
    """
    outpath = "/Users/steveho/gnn_plots/interpretation/collate_enrichment"

    # empty dfs to store results
    reactome_df = pd.DataFrame()
    go_bp_df = pd.DataFrame()
    go_mf_df = pd.DataFrame()

    gene_sets = [
        "Reactome_Pathways_2024",
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
    ]

    # run enrichment analysis for each gene set
    for gene_set in gene_sets:
        gmt = f"/Users/steveho/gnn_plots/interpretation/gmt/{gene_set}.gmt"
        print(f"Running enrichr for {gmt}...")

        for tissue, _ in TISSUES.items():
            node_ablations = pickle.load(
                open(f"{tissue}_release/node_feature_top_genes.pkl", "rb")
            )
            top = node_ablations[idx][:top_n]
            top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

            try:
                enrichr_result = gp.enrichr(
                    gene_list=top_genes,
                    gene_sets=gmt,
                    organism="Human",
                    outdir=None,
                )
            except AttributeError:
                print(f"Error with {gene_set} for {tissue}")
                continue

            significant = enrichr_result.results[
                enrichr_result.results["Adjusted P-value"] < 0.05
            ]

            if significant.empty:
                print(f"No significant terms for {gene_set} in {tissue}")
                continue

            df = significant.copy()
            df["neg_log_p"] = -np.log10(df["Adjusted P-value"])

            # remove GO term IDs
            if "GO" in gene_set:
                df["Term"] = df["Term"].apply(lambda x: x.split("(GO:")[0])

            # get top 5 terms
            top5 = df.nlargest(5, "neg_log_p").copy()
            top5.sort_values("neg_log_p", ascending=False, inplace=True)

            # add sample name
            top5["Sample"] = tissue

            # append to final dfs
            if gene_set == "Reactome_Pathways_2024":
                reactome_df = pd.concat([reactome_df, top5], ignore_index=True)
            elif gene_set == "GO_Biological_Process_2023":
                go_bp_df = pd.concat([go_bp_df, top5], ignore_index=True)
            elif gene_set == "GO_Molecular_Function_2023":
                go_mf_df = pd.concat([go_mf_df, top5], ignore_index=True)

    reactome_df.to_csv(f"{outpath}/reactome_{FEATURES[idx]}.csv", index=False)
    go_bp_df.to_csv(f"{outpath}/go_bp_{FEATURES[idx]}.csv", index=False)
    go_mf_df.to_csv(f"{outpath}/go_mf_{FEATURES[idx]}.csv", index=False)


def main() -> None:
    """Main function to generate ablation figures."""
    # make node_ablation heatmap figures
    df = load_node_feature_ablations()
    plot_node_feature_ablations(df, savename="node_feature_ablations")

    double_df = load_double_node_feature_ablations()
    plot_node_feature_ablations(double_df, savename="double_node_feature_ablations")

    # make individual GO enrichment figures
    working_dir = "/Users/steveho/gnn_plots"
    gencode_file = (
        f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
    )

    # load gencode to symbol mapping
    symbol_to_gencode = load_gencode_lookup(gencode_file)
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # for tissue, tissue_name in TISSUES.items():
    #     node_ablations = pickle.load(
    #         open(f"{tissue}_release/node_feature_top_genes.pkl", "rb")
    #     )
    #     for idx in FEATURES:
    #         plot_single_enrichment_after_ablation(
    #             node_ablations=node_ablations,
    #             gencode_to_symbol=gencode_to_symbol,
    #             idx=idx,
    #             sample=tissue,
    #         )
    for idx in FEATURES:
        collate_enrichment_results(
            gencode_to_symbol=gencode_to_symbol,
            idx=idx,
        )

    # # get top 100 genes for
    # top = genes[12]

    # # convert to gene symbols
    # top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, fc in top]
    # print(top_genes)

    # get common genes..(?)

    # get most differentially affected genes..(?)

    # load GO collated results
    set_matplotlib_publication_parameters()
    df = pd.read_csv("reactome_H3K4me1.csv")
    df["sample_display"] = df["Sample"].map(TISSUES)

    # sort terms by neg_log_p
    df = df.sort_values("neg_log_p", ascending=False).copy()

    # count number of genes
    df["gene_count"] = df["Genes"].str.split(";").str.len()


if __name__ == "__main__":
    main()


# deprecated GSEA code
# # get top_n_genes
# top = node_ablations[idx]

# # convert to absolute fc
# top = [(gene, abs(fc)) for gene, fc in top]

# # remove genes with 0 fold change
# top = [x for x in top if x[1] != 0]

# # convert to gene symbols
# top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

# # pre-rank df for gsea
# ranked_df = pd.DataFrame(top, columns=["Gene", "FC"])
# ranked_df["Gene"] = ranked_df["Gene"].apply(
#     lambda x: gencode_to_symbol.get(x.split("_")[0])
# )

# # run gsea
# pre_res = gp.prerank(
#     rnk=ranked_df,
#     gene_sets="GO_Molecular_Function_2023",
#     threads=4,
#     permutation_num=1000,
#     outdir=None,
#     seed=42,
#     verbose=True,
# )

# ax = dotplot(
#     pre_res.res2d,
#     column="FDR q-val",
#     title="Reactome Pathways",
#     figsize=(40, 8),
#     cutoff=0.25,
# )
# ax.figure.savefig("test_gsea_reactome.png", dpi=300)
