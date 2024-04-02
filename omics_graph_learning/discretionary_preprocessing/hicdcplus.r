# Load libraries
library("argparse")
library(HiCDCPlus)

# Define command-line arguments
parser <- ArgumentParser()
parser$add_argument("--tissue", help="Name of the tissue/cell line", type="character")
parser$add_argument("--working_dir", help="Basedir location of the .hic file", type="character")
parser$add_argument("--outdir", help="Path to the output directory", type="character")
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
#' @param outdir (string) Path to the output directory
#' @param gen_ver (string) Genome version ["hg19", "hg38"]
#' @param binsize (numeric) Bin size
#' @param ncore (numeric) Number of cores to use
#' @return (file) HiCDCPlus output file
#' 

hic_processing <- function(tissue = NULL,
                           working_dir = NULL,
                           outdir = '/tmp',
                           gen_ver = NULL,
                           binsize = 5000,
                           ncore = 8) {
  # Set up vars
  hicfile <- paste0(working_dr, '/', tissue, ".hic")
  outfile <- paste0(outdir, '/', tissue, "_result.txt.gz")

  # generate features
  hicfile_path <- system.file(hicfile, package = "HiCDCPlus")
  construct_features(output_path = outdir,
                     gen = "Hsapiens", gen_ver = gen_ver,
                     sig = "GATC",
                     bin_type = "Bins-uniform",
                     binsize = binsize,
                     chrs = c("chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                              "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                              "chr20", "chr21", "chr22"))
  
  gi_list <- generate_bintolen_gi_list(bintolen_path = outfile, gen = "Hsapiens", gen_ver = gen_ver)
  gi_list <- add_hic_counts(gi_list, hic_path = hic_file)
  gi_list <- expand_1D_features(gi_list)
  set.seed(1010) # HiC-DC downsamples rows for modeling
  gi_list <- HiCDCPlus_parallel(gi_list, ncore = ncore)
  gi_list_write(gi_list, fname = outfile)
  message(paste0("Processing ", tissue, " complete!"))
}

# Set up reference genome
load_ref(gen_ver=args$gen_ver)
message("Packages loaded")

# Run the HiCDCPlus pipeline
hic_processing(tissue=args$tissue, working_dir=args$working_dir, outdir=args$outdir, gen_ver=args$gen_ver, binsize=args$binsize, ncore=args$ncore)