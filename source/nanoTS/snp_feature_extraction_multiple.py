import argparse
import pickle
import pysam
import numpy as np
from multiprocessing import Pool, Manager

import pandas as pd
import math
import progressbar
import random

import math

import sys
from collections import defaultdict
from bisect import bisect_left
    

def get_all_nearest_neighbors(candidate_arr, query_positions, query_refs, query_alts, n=10):
    """
    Vectorized search for the nearest neighbors for multiple query variants.
    
    Parameters:
      candidate_arr: NumPy array of shape (M, 3) with columns: pos, ref, alt.
                     The array must be sorted by pos.
      query_positions: NumPy array of query positions.
      query_refs: NumPy array (or list) of query reference alleles.
      query_alts: NumPy array (or list) of query alternate alleles.
      n: Number of neighbors to search on each side.
      
    Returns:
      A list (one per query) of dictionaries mapping neighbor position -> (neighbor_ref, neighbor_alt).
    """
    # Create an array of candidate positions (assumed sorted)
    positions_only = candidate_arr[:, 0].astype(np.int64)
    # Find insertion indices for all query positions at once
    idx_array = np.searchsorted(positions_only, query_positions)
    
    neighbors_list = []
    for idx, qpos, qref, qalt in zip(idx_array, query_positions, query_refs, query_alts):
        # Define window: up to n neighbors on each side.
        start_idx = max(0, idx - n)
        end_idx = min(len(candidate_arr), idx + n)
        # Slice the candidate array in that window.
        window = candidate_arr[start_idx:end_idx]
        # Filter out an exact match, if present.
        # (Assumes qpos is numeric and qref, qalt are comparable to the candidate strings.)
        mask = ~((window[:, 0].astype(np.int64) == qpos) & 
                 (window[:, 1] == qref) & 
                 (window[:, 2] == qalt))
        filtered = window[mask]
        # Build dictionary: key = candidate position (as int), value = (ref, alt)
        neighbor_dict = {int(row[0]): (row[1], row[2]) for row in filtered}
        neighbors_list.append(neighbor_dict)
    
    return neighbors_list

# Example integration within your function (assuming all variants are on the same chromosome):
def extract_tasks_with_nearest_snvs_preserve_order(vcf_file, bam_file, reference_file, read_depth, all_var):
    """
    Extract tasks from vcf_file preserving order, and for each variant find up to 10 nearest SNVs
    from the comprehensive variant file (all_var). This version uses NumPy to vectorize the binary search.
    Returns a list of tasks.
    """
    # 1) Parse the main VCF to preserve order.
    all_records = []  # list of (global_index, chrom, pos, ref, alt)
    with open(vcf_file, 'r') as f_vcf:
        # Assume first non-comment line is the header.
        header = f_vcf.readline()
        line_index = 0
        for line in f_vcf:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[2]
            alt = fields[3]
            all_records.append((line_index, chrom, pos, ref, alt))
            line_index += 1

    # 2) Parse all_var and build a candidate array (assumes same chromosome)
    if all_var is None:
        raise ValueError("You must provide the path to all_var.")
    
    candidate_list = []
    with open(all_var, 'r') as f_all:
        header = f_all.readline()
        for line in f_all:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            cchrom = fields[0]
            cpos = int(fields[1])
            cref = fields[2]
            calt = fields[3]
            candidate_list.append((cpos, cref, calt))
    # Sort candidate_list by position (if not already sorted)
    candidate_list.sort(key=lambda x: x[0])
    # Convert to a numpy array of dtype object so that string columns are preserved.
    candidate_arr = np.array(candidate_list, dtype=object)
    
    # 3) Vectorize the nearest neighbor search:
    # Build arrays for query positions, refs, alts from all_records.
    query_positions = np.array([rec[2] for rec in all_records], dtype=np.int64)
    query_refs = np.array([rec[3] for rec in all_records])
    query_alts = np.array([rec[4] for rec in all_records])
    
    # Get nearest neighbors for all queries in one go.
    if len(candidate_arr)>0:
    	neighbors_list = get_all_nearest_neighbors(candidate_arr, query_positions, query_refs, query_alts, n=10)
    else:
        neighbors_list=[{}] * len(all_records)
    
    # 4) Build tasks by combining the records and the corresponding neighbors.
    tasks = []
    for (g_idx, chrom, pos, ref, alt), nearest_snvs in zip(all_records, neighbors_list):
        task = (
            g_idx,            # global index
            bam_file,
            reference_file,
            chrom,
            pos,
            ref,
            alt,
            read_depth,
            nearest_snvs      # dictionary of neighbor variants
        )
        tasks.append(task)
    
    return tasks


def calculate_local_mapping_quality_phred(read, position, ref_sequence, window_size=250):
    """
    Calculate the local mapping quality as a Phred score around a given 1-based position
    on the reference genome, using a provided reference sequence.
    Properly handles indels, skipped regions ('N'), and aligns read and reference sequences
    using the CIGAR string and segment handling.

    Parameters:
    - read: pysam.AlignedSegment object representing the read.
    - position: int, the 1-based position on the reference genome.
    - ref_sequence: str, the reference sequence of length (2 * window_size + 1) centered
                    around the position.
    - window_size: int, one-sided length of the window around the position (default is 250 bp).

    Returns:
    - phred_quality: int, the local mapping quality transformed into a Phred score.
      Returns 0 for unmapped or uncovered regions, or 60 for perfect alignment.
    """
    # Convert 1-based position to 0-based for internal processing
    position_0based = position - 1

    # Ensure the read aligns to the reference
    if read.is_unmapped:
        return 0  # Unmapped reads get a score of 0

    # Check if the position is within the read's alignment
    if not (read.reference_start <= position_0based < read.reference_end):
        return 0  # Position not covered gets a score of 0

    # Initialize variables for segment parsing
    segments = []
    current_ref_pos = read.reference_start
    current_read_pos = 0
    segment = None  # Holds the current segment

    # Parse the CIGAR string to identify segments
    for op, length in read.cigartuples:
        if op in [0, 7, 8]:  # Match or mismatch
            if segment is None:
                # Start a new segment
                segment = {
                    'read_start': current_read_pos,
                    'read_end': current_read_pos + length,
                    'ref_start': current_ref_pos,
                    'ref_end': current_ref_pos + length
                }
            else:
                # Extend the current segment
                segment['read_end'] += length
                segment['ref_end'] += length
            current_read_pos += length
            current_ref_pos += length

        elif op == 1:  # Insertion to the reference
            current_read_pos += length

        elif op == 2:  # Deletion from the reference
            current_ref_pos += length

        elif op == 3:  # Skipped region ('N')
            # End the current segment and start a new one after 'N'
            if segment is not None:
                segments.append(segment)
                segment = None  # Reset segment
            current_ref_pos += length
            # No change to current_read_pos because 'N' doesn't consume read bases

        elif op in [4, 5]:  # Soft clip or hard clip
            current_read_pos += length

    # Append the last segment after the loop if it exists
    if segment is not None:
        segments.append(segment)

    # Find the segment containing the reference position
    target_segment = None
    for segment in segments:
        if segment['ref_start'] <= position_0based < segment['ref_end']:
            target_segment = segment
            break

    if target_segment is None:
        return 0  # Position not aligned within any segment

    # Calculate window boundaries on the reference within the segment
    window_start_ref = max(target_segment['ref_start'], position_0based - window_size)
    window_end_ref = min(target_segment['ref_end'], position_0based + window_size + 1)  # +1 to include the position itself

    # Ensure window boundaries are within the provided reference sequence
    ref_seq_offset = window_start_ref - (position_0based - window_size)
    ref_seq_length = len(ref_sequence)
    if ref_seq_offset < 0 or ref_seq_offset + (window_end_ref - window_start_ref) > ref_seq_length:
        return 0  # Reference sequence does not cover the window

    # Build a mapping from reference positions to read positions
    ref_to_read_pos = {}
    aligned_pairs = read.get_aligned_pairs(matches_only=False)
    for read_pos, ref_pos in aligned_pairs:
        if ref_pos is not None and window_start_ref <= ref_pos < window_end_ref:
            ref_to_read_pos[ref_pos] = read_pos

    # Initialize counters
    total_bases = 0
    mismatches = 0
    indels = 0

    # Iterate over reference positions within the window
    for ref_pos in range(window_start_ref, window_end_ref):
        ref_idx = ref_pos - (position_0based - window_size)
        if ref_idx < 0 or ref_idx >= ref_seq_length:
            continue  # Outside the provided reference sequence

        ref_base = ref_sequence[ref_idx]
        read_pos = ref_to_read_pos.get(ref_pos)

        if read_pos is None:
            # Deletion in read (base in reference, gap in read)
            indels += 1
        else:
            read_base = read.query_sequence[read_pos]
            if read_base != ref_base:
                mismatches += 1
            # Include matches and mismatches in total_bases
            total_bases += 1

    # Handle insertions (positions where ref_pos is None)
    for read_pos, ref_pos in aligned_pairs:
        if ref_pos is None and window_start_ref <= current_ref_pos < window_end_ref:
            # Insertion in read (gap in reference)
            indels += 1

    # Total bases is the number of reference positions within the window
    total_bases += indels  # Include indels in total_bases

    if total_bases == 0:
        return 0  # No bases in the specified window

    # Calculate the error rate
    total_errors = mismatches + indels
    error_rate = total_errors / total_bases

    # Return Phred quality score
    if error_rate == 0:
        return 60  # Perfect alignment gets a maximum score
    else:
        phred_quality = int(-10 * math.log10(error_rate))
        return phred_quality
        
        
def recalculate_quality_from_nm(read, max_quality=60):
    """
    Recalculate the mapping quality for a single read using the NM tag,
    considering only the match length (M) for the aligned length.

    Args:
        read (pysam.AlignedSegment): A single read object from pysam.
        max_quality (int): Maximum allowed quality score (default: 60).

    Returns:
        int: Recalculated Phred quality score.
    """
    if read.is_unmapped:
        return 0  # Mapping quality for unmapped reads is 0

    # Check if NM tag exists
    if not read.has_tag("NM"):
        raise ValueError(f"Read {read.query_name} does not have an NM tag.")

    # Get NM value (edit distance)
    nm_value = read.get_tag("NM")

    # Calculate match length (M) from CIGAR string
    match_length = sum(length for op, length in read.cigartuples if op == 0)  # 0 = Match (M)

    # Avoid division by zero
    if match_length == 0:
        return 0  # Default to minimum quality for zero match length

    # Calculate error rate
    error_rate = nm_value / match_length

    # Convert error rate to Phred quality score
    if error_rate == 0:
        phred_quality = max_quality  # Maximum quality
    else:
        phred_quality = -10 * math.log10(error_rate)
        phred_quality = min(phred_quality, max_quality)
        phred_quality = max(phred_quality, 0)  # Ensure non-negative

    return int(phred_quality)




def calculate_entropy_column(fractions):
    if np.sum(fractions) == 0:  # Check if all values are zero
        return 2.5
    entropy = 0
    for p in fractions:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calculate_entropy_matrix(fraction_matrix):
    entropies = []
    for col in fraction_matrix.T:  # Iterate over columns
        entropies.append(calculate_entropy_column(col))
    return np.array(entropies)

def filter_vcf(vcf_file, filtered_vcf_file, ratio_threshold, llr_threshold, llr_diff_threshold,alt_min,total_min):
    # Load VCF data (example assumes a pandas DataFrame for simplicity)

    
    vcf_data = pd.read_csv(vcf_file, sep='\t')
     #ref_reads       alt_reads
    # Apply the filters
    filtered_vcf = vcf_data[
        (vcf_data['ratio_1'] >= ratio_threshold) &
        (vcf_data['alt_reads']>=alt_min) &
        ((vcf_data['ref_reads']+vcf_data['alt_reads'])>=(total_min)) &
        (vcf_data['ref'].str.len() == 1) &
        (vcf_data['alt'].str.len() == 1) &
        (vcf_data['LLR_1'] >= llr_threshold) &
        ((vcf_data['LLR_1'] - vcf_data['LLR_2']) >= llr_diff_threshold)
    ]
    # 1. Filter to only include chromosomes 1 to 22
    df_filtered = filtered_vcf[filtered_vcf['chrom'].str.match(r'^chr([1-9]|1[0-9]|2[0-2]|X|Y|M)$')].copy()

    # 2. Create a mapping for chromosome ordering
    chrom_order = {f'chr{i}': i for i  in list(map(str, range(1, 23))) + ['X', 'Y', 'M']}


    # 3. Map the chromosome column to a numeric order column
    df_filtered['chrom_order'] = df_filtered['chrom'].map(chrom_order)

    # 4. Sort the DataFrame by the numeric chromosome order, then by position
    df_sorted = df_filtered.sort_values(by=['chrom_order', 'pos']).drop(columns='chrom_order')

    # Optionally, reset the index
    df_sorted = df_sorted.reset_index(drop=True)

    # Save filtered VCF data
    #df_sorted.to_csv(filtered_vcf_file, sep='\t', index=False)
    #print(f"Filtered SNV saved to {filtered_vcf_file}")
    #print(f'{filtered_vcf.shape[0]} SNV pass initial filter')
    return df_sorted

def get_reference_sequence(reference_file, chrom, pos,window=30):
    reference = pysam.FastaFile(reference_file)
    start = max(0, pos -1- window)
    end = pos + window
    sequence = reference.fetch(chrom, start, end).upper()
    reference.close()
    return sequence

def one_hot_encode_sequence(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_matrix = np.array([mapping.get(base, [0, 0, 0, 0]) for base in sequence])
    return one_hot_matrix.T

def calculate_base_fractions(reads, pos,nearest_snvs):
    #print(pos)
    #print(nearest_snvs)
    base_counts = np.zeros((4, 61))
    nearest_snvs_count=np.zeros((3,20))
    bases = ["A", "C", "G", "T"]
    coverage = np.zeros(61)
    pos=pos-1
    nearest_snvs_20_pos=list(nearest_snvs.keys())
    for read in reads:
        read_positions = read.get_reference_positions(full_length=True)
        read_bases = read.query_sequence # if aligned sense, same to ref, otherwise, reverse complemented

        for idx, ref_pos in enumerate(read_positions):
            #print(idx)
            #print(ref_pos)
            #print(read.is_reverse)
            #print(idx)
            if ref_pos is None:
                continue
            window_idx = ref_pos - (pos - 30)
            if 0 <= window_idx < 61:
                base = read_bases[idx]
                coverage[window_idx] += 1
                if base in bases:
                    base_idx = bases.index(base)
                    base_counts[base_idx, window_idx] += 1
                    
            ref_pos_1_based=ref_pos+1
            if ref_pos_1_based in nearest_snvs:
                
                read_base = read_bases[idx]
                #if read.is_reverse:
                 #   ref_base=reverse_complement(seq_base)
                #else:
                 #   ref_base=seq_base
                #if ref_pos_1_based == 2207898:
                 #   print(ref_pos_1_based)
                  #  print(read.is_reverse)
                   # print(ref_base)
                    
                    #print(idx)
                    #print(read.query_name)
                    #print(read_bases)
                if read_base == nearest_snvs[ref_pos_1_based][0]:
                    nearest_snvs_count[0,nearest_snvs_20_pos.index(ref_pos_1_based)]+=1
                elif read_base == nearest_snvs[ref_pos_1_based][1]:
                    nearest_snvs_count[1,nearest_snvs_20_pos.index(ref_pos_1_based)]+=1
                else:
                    nearest_snvs_count[2,nearest_snvs_20_pos.index(ref_pos_1_based)]+=1
        
    base_fractions = base_counts / (base_counts.sum(axis=0) + 1e-10)
    return base_fractions, base_counts , coverage,nearest_snvs_count

def counts2fractions(ref_nearest_snvs_counts_forward,ref_nearest_snvs_counts_reverse,alt_nearest_snvs_counts_forward,alt_nearest_snvs_counts_reverse):
    """
    each numpy is format of np.zeros((2,20))
    calculate the fraction of each position 
    """
    ref_nearest_snvs_counts_forward = ref_nearest_snvs_counts_forward / (ref_nearest_snvs_counts_forward.sum(axis=0) + 1e-10)
    ref_nearest_snvs_counts_reverse = ref_nearest_snvs_counts_reverse / (ref_nearest_snvs_counts_reverse.sum(axis=0) + 1e-10)
    alt_nearest_snvs_counts_forward = alt_nearest_snvs_counts_forward / (alt_nearest_snvs_counts_forward.sum(axis=0) + 1e-10)
    alt_nearest_snvs_counts_reverse = alt_nearest_snvs_counts_reverse / (alt_nearest_snvs_counts_reverse.sum(axis=0) + 1e-10)
    return [ref_nearest_snvs_counts_forward,ref_nearest_snvs_counts_reverse,alt_nearest_snvs_counts_forward,alt_nearest_snvs_counts_reverse]
    


    

def calculate_coverage_array(reads, pos): #also include indel
    coverage = np.zeros(61)

    for read in reads:
        read_positions = read.get_reference_positions(full_length=True)

        for ref_pos in read_positions:
            if ref_pos is None:
                continue
            window_idx = ref_pos - (pos - 30)
            if 0 <= window_idx < 61:
                coverage[window_idx] += 1

    return coverage
def transform_value(value):
    """
    Transform the input value into:
    - 0 if value <= 30
    - 1 if 30 < value <= 100
    - 2 if value > 100

    Parameters:
        value (float or int): Input value to transform.

    Returns:
        int: Transformed value (0, 1, or 2).
    """
    if value <= 30:
        return 0
    elif 30 < value <= 100:
        return 1
    else:
        return 2
def compute_average_distances(reads, pos):
    pos0 = pos - 1  # Convert 1-based position to 0-based
    start_distances = []
    end_distances = []
    #print(f'pos0:{pos0}')
    for read in reads:
        read_start = read.reference_start  # 0-based
        read_end = read.reference_end - 1  # Last aligned position (inclusive)
        if pos0 < read_start or pos0 > read_end:
            # SNP position not in read alignment (should not happen)
            continue
        #print(read_start)
        start_distance = pos0 - read_start
        end_distance = read_end - pos0
        start_distances.append(transform_value(start_distance))
        end_distances.append(transform_value(end_distance))
    avg_start_distance = np.mean(start_distances) if start_distances else -1
    avg_end_distance = np.mean(end_distances) if end_distances else -1
    #print(avg_start_distance)
    return avg_start_distance, avg_end_distance

def extract_features_for_snp_wrapper(args):
    (index, bam_file, reference_file, chrom, pos, ref, alt,read_depth,nearest_SNVs) =args
    features = extract_features_for_snp(bam_file, reference_file, chrom, pos, ref, alt,read_depth,nearest_SNVs)
    return (index, features)

def sample_reads_randomly(bam_file, chrom, start, end, sample_size=1000, seed=42):
    """
    Randomly sample up to `sample_size` reads from a given region using reservoir sampling.
    """
    random.seed(seed)  # For reproducibility (optional)
    reservoir = []
    n = 0  # Counts the total number of reads seen so far
    
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            n += 1
            # If the reservoir isn't full yet, just append
            if len(reservoir) < sample_size:
                reservoir.append(read)
            else:
                # Once the reservoir is full, replace an existing item with 
                # decreasing probability as n grows
                j = random.randrange(n)  # random int in [0, n-1]
                if j < sample_size:
                    reservoir[j] = read

    return reservoir
def calculate_base_fractions_total(reads, pos):
    base_counts = np.zeros((4, 61))

    bases = ["A", "C", "G", "T"]
    coverage = np.zeros(61)
    pos=pos-1
    for read in reads:
        read_positions = read.get_reference_positions(full_length=True)
        read_bases = read.query_sequence

        for idx, ref_pos in enumerate(read_positions):
            if ref_pos is None:
                continue
            window_idx = ref_pos - (pos - 30)
            if 0 <= window_idx < 61:
                base = read_bases[idx]
                coverage[window_idx] += 1
                if base in bases:
                    base_idx = bases.index(base)
                    base_counts[base_idx, window_idx] += 1

    base_fractions = base_counts / (base_counts.sum(axis=0) + 1e-10)
    return base_fractions, base_counts , coverage

        
def extract_features_for_snp(bam_file, reference_file, chrom, pos, ref, alt,read_depth,nearest_snvs):
    mapq_threshold=20
    if read_depth>0:
    	all_reads = sample_reads_randomly(bam_file, chrom,  pos - 1, pos, sample_size=read_depth, seed=2025) # only select 1000 reads to speed up the calculation
    else:
    	bam = pysam.AlignmentFile(bam_file, "rb")
    	all_reads = bam.fetch(chrom, pos - 1, pos)
    ref_reads_forward = []
    ref_reads_reverse = []
    alt_reads_forward = []
    alt_reads_reverse = []
    other_reads = []
    total_reads = []
    reference_sequence = get_reference_sequence(reference_file, chrom, pos)
    reference_sequence_500 = get_reference_sequence(reference_file, chrom, pos,250)
    for read in all_reads:
        if read.is_unmapped or read.is_duplicate or read.mapping_quality < mapq_threshold or read.is_supplementary:
            continue
        
        read.mapping_quality=recalculate_quality_from_nm(read) # new mapping_quality
        if read.mapping_quality<10:
            continue
        #read_lq=calculate_local_mapping_quality_phred(read, pos,reference_sequence_500,250)

        #if read_lq<10:
           # continue
        total_reads.append(read)
        if pos - 1 in read.get_reference_positions(full_length=True):
            base_at_pos = read.query_sequence[read.get_reference_positions(full_length=True).index(pos - 1)]
            if base_at_pos == ref:
                if read.is_reverse:
                    ref_reads_reverse.append(read)
                else:
                    ref_reads_forward.append(read)
            elif base_at_pos == alt:
                if read.is_reverse:
                    alt_reads_reverse.append(read)
                else:
                    alt_reads_forward.append(read)
            else:
                other_reads.append(read)
        else:
            other_reads.append(read)

    # Existing computations
    total_base_fraction, total_base_counts , total_coverage = calculate_base_fractions_total(total_reads, pos)
    
    ref_base_fractions_forward, ref_base_counts_forward , ref_coverage_forward, ref_nearest_snvs_counts_forward = calculate_base_fractions(ref_reads_forward, pos,nearest_snvs)
    #ref_coverage_forward = calculate_coverage_array(ref_reads_forward, pos)
    ref_base_fractions_reverse, ref_base_counts_reverse , ref_coverage_reverse, ref_nearest_snvs_counts_reverse = calculate_base_fractions(ref_reads_reverse, pos,nearest_snvs)
    #ref_coverage_reverse = calculate_coverage_array(ref_reads_reverse, pos)
    alt_base_fractions_forward, alt_base_counts_forward , alt_coverage_forward, alt_nearest_snvs_counts_forward = calculate_base_fractions(alt_reads_forward, pos,nearest_snvs)
    #alt_coverage_forward = calculate_coverage_array(alt_reads_forward, pos)
    alt_base_fractions_reverse, alt_base_counts_reverse , alt_coverage_reverse, alt_nearest_snvs_counts_reverse = calculate_base_fractions(alt_reads_reverse, pos,nearest_snvs)
    #alt_coverage_reverse = calculate_coverage_array(alt_reads_reverse, pos)

    one_hot_encoded_sequence = one_hot_encode_sequence(reference_sequence)

    #other_base_fractions, other_base_counts = calculate_base_fractions(other_reads, pos)
    #total_base_counts = ref_base_counts_forward + ref_base_counts_reverse + alt_base_counts_forward + alt_base_counts_reverse + other_base_counts
    #total_base_fraction = total_base_counts / (total_base_counts.sum(axis=0) + 1e-10)

    base_entropy = calculate_entropy_matrix(total_base_fraction)

    # Compute average distances
    ref_forward_start_dist, ref_forward_end_dist = compute_average_distances(ref_reads_forward, pos)
    ref_reverse_start_dist, ref_reverse_end_dist = compute_average_distances(ref_reads_reverse, pos)
    #print('alt_reads_forward')
    alt_forward_start_dist, alt_forward_end_dist = compute_average_distances(alt_reads_forward, pos)
    alt_reverse_start_dist, alt_reverse_end_dist = compute_average_distances(alt_reads_reverse, pos)

    nearest_snvs_fraction=counts2fractions(ref_nearest_snvs_counts_forward,ref_nearest_snvs_counts_reverse,alt_nearest_snvs_counts_forward,alt_nearest_snvs_counts_reverse)
    
    return (ref_base_fractions_forward, ref_coverage_forward, ref_base_fractions_reverse, ref_coverage_reverse,
            alt_base_fractions_forward, alt_coverage_forward, alt_base_fractions_reverse, alt_coverage_reverse,
            one_hot_encoded_sequence, total_base_fraction, base_entropy,
            ref_forward_start_dist, ref_forward_end_dist, ref_reverse_start_dist, ref_reverse_end_dist,
            alt_forward_start_dist, alt_forward_end_dist, alt_reverse_start_dist, alt_reverse_end_dist,
            nearest_snvs_fraction[0],nearest_snvs_fraction[1],nearest_snvs_fraction[2],nearest_snvs_fraction[3])

def extract_features_with_progress(task, progress):
    result = extract_features_for_snp_wrapper(task)  # Your task processing function
    with progress.get_lock():  # Safely update shared progress counter
        progress.value += 1
    return result
def extract_VCF_feature(vcf_file, bam_file, reference_file, num_processes,read_depth=1000,all_var=None):
    tasks = extract_tasks_with_nearest_snvs_preserve_order(
        vcf_file,
        bam_file,
        reference_file,
        read_depth,
        all_var
    )

    widgets = ['Extract SNV features: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='=', left='[', right=']'), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(tasks)).start()

    # Batch processing
    batch_size = 20 * num_processes  # Tune this based on the workload and threads
    

    with Pool(processes=num_processes) as pool:
        all_results = []
        count = 0

        # 2) Use the top-level function in imap_unordered
        for result in pool.imap_unordered(extract_features_for_snp_wrapper, tasks):
            all_results.append(result)
            count += 1
            bar.update(count)
        
        # Sort results by the original index if needed
    all_results.sort(key=lambda x: x[0])
    bar.finish()
    # Build variant features dictionary
    variant_features = {}
    for (_, bam_file, reference_file, chrom, pos, ref, alt,read_depth,nearest_snvs), (_, features) in zip(tasks, all_results):
        scale_coverage = max(
            list(features[1]) + list(features[3]) + list(features[5]) + list(features[7])
        )
        if scale_coverage > 0:
            features = list(features)
            features[1] /= scale_coverage
            features[3] /= scale_coverage
            features[5] /= scale_coverage
            features[7] /= scale_coverage
            features = tuple(features)

        variant_features[(chrom, pos)] = {
            "ref_base_fractions_forward": features[0],
            "ref_coverage_forward": features[1],
            "ref_base_fractions_reverse": features[2],
            "ref_coverage_reverse": features[3],
            "alt_base_fractions_forward": features[4],
            "alt_coverage_forward": features[5],
            "alt_base_fractions_reverse": features[6],
            "alt_coverage_reverse": features[7],
            "one_hot_encoded_sequence": features[8],
            "total_base_fraction": features[9],
            "base_entropy": features[10],
            "ref_forward_start_dist": features[11],
            "ref_forward_end_dist": features[12],
            "ref_reverse_start_dist": features[13],
            "ref_reverse_end_dist": features[14],
            "alt_forward_start_dist": features[15],
            "alt_forward_end_dist": features[16],
            "alt_reverse_start_dist": features[17],
            "alt_reverse_end_dist": features[18],
            "max_coverage":scale_coverage,
            'ref_snvs_fractions_forward':features[19],
            'ref_snvs_fractions_reverse':features[20],
            'alt_snvs_fractions_forward':features[21],
            'alt_snvs_fractions_reverse':features[22]
        }
    return variant_features
def run_snp_feature_extraction_multiple(args):
    filtered_variant = filter_vcf(
        args.variant,
        args.filtered_variant,
        args.ratio_threshold,
        args.llr_threshold,
        args.llr_diff_threshold,
        args.ALT,
        args.total
    )
    if args.all_v:
        all_filtered_variant = filter_vcf(
            args.all_v,
            args.all_v+'.filter',
            args.ratio_threshold,
            args.llr_threshold,
            args.llr_diff_threshold,
            5,
            5
            
        )
    else:
        all_filtered_variant = filter_vcf(
            args.variant,
            args.filtered_variant+'.alt_5',
            args.ratio_threshold,
            args.llr_threshold,
            args.llr_diff_threshold,
            5,
            5   
        )
    if args.all_v:
        feature_for_train = extract_VCF_feature(args.filtered_variant, args.bam, args.ref, args.threads,args.depth,args.all_v+'.filter')
    else:
        feature_for_train = extract_VCF_feature(args.filtered_variant, args.bam, args.ref, args.threads,args.depth,args.filtered_variant+'.alt_5')
    with open(args.output, "wb") as file:
        pickle.dump(feature_for_train, file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for SNPs from VCF and BAM files.")
    parser.add_argument("-v", '--variant', required=True, help="Path to input SNV file")
    parser.add_argument("--bam", required=True, help="Path to input BAM file")
    parser.add_argument("--ref", required=True, help="Path to reference genome fasta file")
    parser.add_argument("--output", required=True, help="Path to output pickle file for storing SNP features")
    parser.add_argument("--all_v",  help="Only for training. Input all variants")
    parser.add_argument("--filtered_variant", required=True, help="Path to output filtered SNV file")
    parser.add_argument('-t', "--threads", type=int, default=4, help="Number of parallel processes to use")
    parser.add_argument("--ratio_threshold", type=float, default=0.05, help="Minimum ratio threshold for filtering (default: 0.05)")
    parser.add_argument("--llr_threshold", type=float, default=0, help="Minimum LLR threshold for filtering (default: 0)")
    parser.add_argument("--llr_diff_threshold", type=float, default=0, help="Minimum difference between LLR_1 and LLR_2 for filtering (default: 0)")
    parser.add_argument("-d","--depth", type=int, default=1000, help="Maximum reads to load for each variant. Set 0 to remove the limit (default: 1000)")
    parser.add_argument("--ALT", type=int, default=5, help="Minimum ALT SNV coverage (default: 5)")
    parser.add_argument("--total", type=int, default=5, help="Minimum total base coverage (default: 5)")
    args = parser.parse_args()
    
    run_snp_feature_extraction_multiple(args)




