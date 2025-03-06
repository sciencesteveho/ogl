# Load libraries
library("argparse")
library(HiCDCPlus)

# Define command-line arguments
parser <- ArgumentParser()
parser$add_argument("--tissue", help="Name of the tissue/cell line", type="character")
parser$add_argument("--working_dir", help="Basedir location of the .hic file", type="character")
parser$add_argument("--gen_ver", help="Genome version", type="character")
parser$add_argument("--binsize", help="Bin size", type="numeric")
parser$add_argument("--ncore", help="Number of cores to use", type="numeric")
args <- parser$parse_args()

#' Load reference genome.
#'
#' @param gen_ver (string) Genome version ["hg19", "hg38"]
#' @return (BSgenome object) Reference genome
#'

load_ref <- function(gen_ver = NULL) {
    if (gen_ver == "hg38") {
        library(BSgenome.Hsapiens.UCSC.hg38)
    } else if (gen_ver == "hg19") {
        library(BSgenome.Hsapiens.UCSC.hg19)
    } else {
        stop("Genome version not supported")
    }
}

#' Estimate statistical significance of Hi-C interactions usinf HiCDCPlus
#'
#' @param tissue (string) Name of tissue or cell line
#' @param working_dir (string) Path to the working directory
#' @param gen_ver (string) Genome version ["hg19", "hg38"]
#' @param binsize (numeric) Bin size
#' @param ncore (numeric) Number of cores to use
#' @return (file) HiCDCPlus output file
#'

hic_processing <- function(tissue = NULL,
                           working_dir = NULL,
                           gen_ver = NULL,
                           binsize = 5000,
                           ncore = 8) {
  # Set up vars
  hicfile <- paste0(working_dir, '/', tissue, ".hic")
  feats <- paste0(working_dir, '/tmp/', tissue)
  bintolen <- paste0(working_dir, '/tmp/', tissue, "_bintolen.txt.gz")
  outfile <- paste0(working_dir, '/fdr_filtered/', tissue, "_result.txt.gz")

  # generate features
  hicfile_path <- system.file(hicfile, package = "HiCDCPlus")
  construct_features(output_path = feats,
                     gen = "Hsapiens", gen_ver = gen_ver,
                     sig = "GATC",
                     bin_type = "Bins-uniform",
                     binsize = binsize,
                     chrs = c("chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                              "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                              "chr20", "chr21", "chr22"),)
                    #  chrs = c("chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                    #           "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                    #           "chr20", "chr21", "chr22", "chrX", "chrY"),)

  gi_list <- generate_bintolen_gi_list(bintolen_path = bintolen, gen = "Hsapiens", gen_ver = gen_ver)
  gi_list <- add_hic_counts(gi_list, hic_path = hicfile)
  gi_list <- expand_1D_features(gi_list)
  set.seed(1010) # HiC-DC downsamples rows for modeling
  gi_list <- HiCDCPlus_parallel(gi_list, ncore = ncore)
  gc()  # force garbage collection to see if it helps with segmentation fault error
  gi_list_write(gi_list, fname = outfile)
  message(paste0("Processing ", tissue, " complete!"))
}

# Set up reference genome
load_ref(gen_ver=args$gen_ver)
message("Packages loaded")

# Run the HiCDCPlus pipeline
hic_processing(tissue=args$tissue, working_dir=args$working_dir, gen_ver=args$gen_ver, binsize=args$binsize, ncore=args$ncore)
