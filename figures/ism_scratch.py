import numpy as np
from pyfaidx import Fasta

reference_fasta_path = "/ocean/projects/bio210019p/stevesho/resources/hg38.fa'

"""chr17	7587875	7588218	chr17_7587875_enhancer"""

SEQUENCE_LENGTH = 196_608  # Full input length for Enformer
BIN_SIZE = 128             # Each bin covers 128 bp
NUM_BINS_REGION = 128      # Region width in output bins
REGION_SIZE = BIN_SIZE * NUM_BINS_REGION  # 16,384
FLANK_SIZE = (SEQUENCE_LENGTH - REGION_SIZE) // 2  # 90,112

def extract_enformer_input_sequence(
    chrom: str, 
    region_start: int,
    region_end: int,
    reference_fasta_path: str
) -> str:
    """
    Extracts the 196,608 bp window for Enformer such that
    the [region_start, region_end) is centered in the output.
    
    Args:
      chrom (str): chromosome name, e.g. "chr1"
      region_start (int): 0-based start coordinate of region-of-interest
      region_end (int): 0-based end coordinate of region-of-interest (exclusive)
      reference_fasta_path (str): path to reference genome FASTA
      
    Returns:
      str: A 196,608-length sequence (ACGT) suitable as input to Enformer.
    """
    # Basic checks
    if (region_end - region_start) != REGION_SIZE:
        raise ValueError(
            f"Requested region must be {REGION_SIZE} bp (128 bins)."
        )
    
    # Middle of region-of-interest
    center = (region_start + region_end) // 2
    
    # Start and end for the entire 196,608-bp window
    window_start = center - (SEQUENCE_LENGTH // 2)
    window_end   = center + (SEQUENCE_LENGTH // 2)
    
    # Fetch sequence from reference
    ref = Fasta(reference_fasta_path)
    
    # Because we might be near the edges of the chromosome, clamp to valid range:
    chrom_length = len(ref[chrom])
    window_start_clamped = max(window_start, 0)
    window_end_clamped   = min(window_end, chrom_length)
    
    extracted = ref[chrom][window_start_clamped : window_end_clamped].seq
    extracted = extracted.upper().replace('N', 'A')  # or some policy for N
    
    # If needed, pad if window is out-of-bounds on either side
    left_pad  = window_start_clamped - window_start
    right_pad = window_end - window_end_clamped
    if left_pad < 0: left_pad = 0
    if right_pad < 0: right_pad = 0
    
    padded_seq = ('A' * left_pad) + extracted + ('A' * right_pad)
    if len(padded_seq) != SEQUENCE_LENGTH:
        raise ValueError("Window extraction failed to produce 196,608 bp.")
        
    return padded_seq
