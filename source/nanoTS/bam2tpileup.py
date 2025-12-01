import sys
import argparse
import subprocess
from collections import Counter
import re
import pysam
from scipy.optimize import minimize
from multiprocessing import Pool
import numpy as np
import progressbar
def log_likelihood(lambda_, i, k, epsilon=0.05):
    """Log-likelihood function for RNA editing level estimation."""
    p = lambda_ * (1 - epsilon) + (1 - lambda_) * epsilon
    return -(i * np.log(p) + (k - i) * np.log(1 - p))


def estimate_lambda_0(k, i, epsilon=0.05):
    """Estimate RNA editing level (lambda_0) for the alternate allele."""
    #result = minimize(log_likelihood, x0=0.5, args=(i, k, epsilon), bounds=[(0, 1)])
    #return result.x[0] if result.success else "NA"
    #print(k)
    return i/k


def calculate_LLR(lambda_0, i, k, epsilon=0.005):
    """Calculate the log-likelihood ratio (LLR) comparing lambda_0 to the error-only model."""
    ll_with_lambda = -log_likelihood(lambda_0, i, k, epsilon)
    ll_with_epsilon = -log_likelihood(0, i, k, epsilon)
    return ll_with_lambda - ll_with_epsilon

def parse_pileup(input_stream, mismath_min=5, total_min=10, ratio_min=0.05, epsilon_value=0.01):
    """
    Parse a SAMtools pileup input stream to calculate base composition,
    treating upper and lower case bases equally, and write results to stdout.
    """
    deletion_tracker = {}  # Tracks deletions for subsequent positions
    results = []
    
    for line in input_stream:
        cols = line.strip().split()
        if len(cols) < 5:
            continue  # Skip incomplete lines

        chrom, pos, ref_base, coverage, bases = cols[:5]
        pos = int(pos)  # Convert position to integer for tracking deletions

        # Adjust for deletions tracked from earlier positions
        deletions = deletion_tracker.pop(pos, 0)

        processed_bases = []
        insertions = 0
        i = 0
        while i < len(bases):
            base = bases[i]
            if base == '+':
                # Match '+' followed by digits (insertion length)
                match = re.match(r'\+(\d+)', bases[i:])
                if match:
                    insertion_length = int(match.group(1))
                    num_digits = len(match.group(1))
                    # Skip over the '+' sign, number, and inserted bases
                    i += 1 + num_digits + insertion_length
                    insertions += 1
                    continue
            elif base == '-':
                # Match '-' followed by digits (deletion length)
                match = re.match(r'\-(\d+)', bases[i:])
                if match:
                    deletion_length = int(match.group(1))
                    num_digits = len(match.group(1))
                    # Record deletions for subsequent positions
                    for j in range(1, deletion_length + 1):
                        deletion_tracker[pos + j] = deletion_tracker.get(pos + j, 0) + 1
                    # Skip over the '-' sign, number, and deleted bases
                    i += 1 + num_digits + deletion_length
                    continue
            elif base == '^':
                i += 2  # Skip the start of a read segment and mapping quality
                continue
            elif base == '$':
                i += 1  # Skip the end of a read segment
                continue
            else:
                processed_bases.append(base)
                i += 1

        processed_bases = ''.join(processed_bases).upper()
        base_counts = Counter(processed_bases)
        # Get individual base counts

        match_count = base_counts.get('.', 0) + base_counts.get(',', 0)
        ref_count = match_count
        four_base_reads = [
            base_counts.get('A', 0),
            base_counts.get('T', 0),
            base_counts.get('C', 0),
            base_counts.get('G', 0)
        ]
        alt_count = max(four_base_reads)
        total_coverage = ref_count + alt_count  # Include deletions if needed

        # Identify the alt base
        if alt_count > 0:
            max_alt_index = four_base_reads.index(alt_count)
            alt = ['A', 'T', 'C', 'G'][max_alt_index]
            # new code 11/11/2025; get all max ALT
            max_alt_indices = [i for i, x in enumerate(four_base_reads) if x == alt_count]
            if deletions == alt_count:
                max_alt_indices.append(4)
            if insertions == alt_count:
                max_alt_indices.append(5)
            alts = [base for i, base in enumerate(['A', 'T', 'C', 'G','DEL','INS']) if i in max_alt_indices]

        else:
            alt = 'N'  # No alternate base

        if alt_count > 0:
            ratio1 = alt_count / total_coverage if total_coverage > 0 else 0
            # Apply filters
            if (total_coverage >= total_min and
                alt_count >= mismath_min and
                ratio1 >= ratio_min):
                lambda_1 = estimate_lambda_0(total_coverage, alt_count, epsilon_value)
                LLR1 = calculate_LLR(lambda_1, alt_count, total_coverage, epsilon_value) if lambda_1 != "NA" else "NA"

                # Secondary analysis
                other_counts = (
                    four_base_reads[:max_alt_index] +
                    four_base_reads[(max_alt_index + 1):] +
                    [deletions, insertions]
                )
                max_other_count = max(other_counts)
                other_total_counts=ref_count + max_other_count
                if other_total_counts>0:
                    lambda_2 = estimate_lambda_0(other_total_counts, max_other_count, epsilon_value)
                    LLR2 = calculate_LLR(lambda_2, max_other_count, other_total_counts, epsilon_value) if lambda_2 != "NA" else "NA"
                    ratio2 = max_other_count / (ref_count + max_other_count) 
                else:
                    (lambda_2,LLR2,ratio2)=(0,0,0)
                    

                # Write results
                results.append('\t'.join([
                    str(field) for field in [
                        chrom, pos, ref_base.upper(), alt,
                        ref_count, alt_count, ratio1,
                        lambda_1, LLR1, ratio2, lambda_2, LLR2 , ','.join(alts)
                    ]
                ]))
    #for each in results:
     #   sys.stdout.writelines(each + '\n')
    #sys.stdout.flush()
    return results

def run_mpileup_wrapper(args):
    (region, bam_path, ref_path, mismatch, total, ratio,epsilon)=args
    return run_mpileup(region, bam_path, ref_path, mismatch, total, ratio,epsilon)

def run_mpileup(region, bam_path, ref_path, mismatch, total, ratio,epsilon):
    """
    Run SAMtools mpileup for a specific region and parse the output.
    """
    samtools_cmd = [
        "samtools", "mpileup", "-B",
        "-f", ref_path,
        "--excl-flags", "2316",
        "--min-BQ", "0",
        "--max-depth", "5000",
        "--min-MQ", "20",
        bam_path,
        "-r", region
    ]
    process = subprocess.Popen(samtools_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with process.stdout as output_stream:
        return parse_pileup(output_stream, mismath_min=mismatch, total_min=total, ratio_min=ratio,epsilon_value=epsilon)

def split_regions(chrom, length, block_size=100000):
    """
    Split a chromosome into smaller blocks of specified size.
    """
    regions = []

    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        regions.append(f"{chrom}:{start+1}-{end}")
    return regions

def split_regions_start(chrom, start_pos,end_pos, block_size=100000):
    """
    Split a chromosome into smaller blocks of specified size.
    """
    regions = []
    target_chrom=[f'chr{i}' for i in range(1,23)] + ['chrX','chrY']
    if chrom in target_chrom:
        for start in range(start_pos, end_pos+1, block_size):
            end = min(start + block_size, end_pos)
            regions.append(f"{chrom}:{start}-{end}")
        return regions
        
def process_bam(bam_path, ref_path, region=None, mismatch=5, total=10, ratio=0.05, threads=4,epsilon=0.01):
    """
    Process the BAM file using multiprocessing for the specified region.
    """
    # Read chromosome information from BAM file
    bam = pysam.AlignmentFile(bam_path, "rb")
    chrom_lengths = {ref["SN"]: ref["LN"] for ref in bam.header["SQ"]}
    bam.close()
    headline='\t'.join([str(i) for i in ["chrom", "pos", "ref", "alt", "ref_reads", "alt_reads", "ratio_1", "lambda_1", "LLR_1",
                         "ratio_2", "lambda_2", "LLR_2",'alt_tie']])
    # Combine and write results to stdout
    sys.stdout.write(
       headline+'\n'
    )
    # Determine regions to process
    regions = []
    if region:
        match = re.match(r"(\w+)(?::(\d+)-(\d+))?", region)
        if not match:
            raise ValueError("Invalid region format. Use 'chrom:start-end' or 'chrom'.")
        chrom = match.group(1)
        start = int(match.group(2)) if match.group(2) else 1
        end = int(match.group(3)) if match.group(3) else chrom_lengths[chrom]
        if start > chrom_lengths[chrom] or start<1 or start>end:
            raise ValueError("Invalid region 'start' scale.")
        if end >chrom_lengths[chrom]:
            end=chrom_lengths[chrom]
        print(f"Detect SNV from {chrom}:{start}-{end}", file=sys.stderr)

        regions = split_regions_start(chrom,start, end)
    else:
        print(f"Detect SNV from", file=sys.stderr)
        target_chrom=[f'chr{i}' for i in range(1,23)] + ['chrX','chrY']
        print(target_chrom,file=sys.stderr)
        
        for chrom, length in chrom_lengths.items():
            if chrom in target_chrom:
                regions.extend(split_regions(chrom, length))

    # Run samtools mpileup for each region in parallel
    #with Pool(threads) as pool:
     #   results = pool.starmap(
      #      run_mpileup,
       #     [(region, bam_path, ref_path, mismatch, total, ratio,epsilon) for region in regions]
        #)

    widgets = ['Detect SNV: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='=', left='[', right=']'), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(regions)).start()

    # Batch processing
    batch_size = 20 * threads  # Tune this based on the workload and threads

    tasks=[(region, bam_path, ref_path, mismatch, total, ratio, epsilon) for region in regions]
    with Pool(processes=threads) as pool:
        all_results = []
        count = 0

        # 2) Use the top-level function in imap_unordered
        for result in pool.imap_unordered(run_mpileup_wrapper, tasks):
            #all_results.append(result)
            for each in result:
     	        sys.stdout.writelines(each + '\n')
            sys.stdout.flush()
            count += 1
            bar.update(count)
        



    bar.finish()
    #for result in results:
     #   for each in result:
      #      sys.stdout.writelines(each+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract read depth coverage from mpileup with filtering and ALT SNV level estimation")
    parser.add_argument("--bam", required=True, help="Path to input BAM file")
    parser.add_argument("--ref", required=True, help="Path to reference genome fasta file")
    parser.add_argument("--region", required=False, help="Region to process (e.g., chr1 or chr1:1000-2000)")
    parser.add_argument("--ALT", type=int, default=5, help="Minimum ALT SNV coverage (default: 5)")
    parser.add_argument("--total", type=int, default=10, help="Minimum total base coverage (default: 10)")
    parser.add_argument("--ratio", type=float, default=0.05, help="Minimum ratio of ALT SNV (default: 0.05)")
    parser.add_argument("--epsilon", default=0.05, type=float, help="Sequencing error rate")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use (default: 4)")
    args = parser.parse_args()
    args.epsilon=args.epsilon/5
    process_bam(args.bam, args.ref, args.region, args.ALT, args.total, args.ratio, args.threads,args.epsilon)

