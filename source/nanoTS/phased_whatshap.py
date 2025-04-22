import os
import subprocess
import sys
import argparse

def run_command(cmd):
    """Run a shell command and exit if it fails."""
    print("Running command:", cmd)
    subprocess.check_call(cmd, shell=True)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phasing pipeline for ONT long-read data using Whatshap and SAMtools."
    )
    parser.add_argument("--vcf", required=True,
                        help="Path to the input VCF file (output_predict_VCF).")
    parser.add_argument("--ref", required=True,
                        help="Path to the reference genome (hg38.fa).")
    parser.add_argument("--bam", required=True,
                        help="Path to the input BAM file.")
    parser.add_argument("--phase", required=True,
                        help="Directory for phasing output.")
    parser.add_argument("--phase", required=True,
                        help="Directory for phasing output.")
    return parser.parse_args()
def filter_vcf_contigs(input_vcf, output_vcf):
    """
    Reads a VCF file and removes header contig lines that are not from chr1 to chr22.

    :param input_vcf: Path to the input VCF file.
    :param output_vcf: Path to the output VCF file with filtered header.
    """
    import re

    valid_contigs = {f"chr{i}" for i in range(1, 23)}  # Set of valid chromosome IDs (chr1 to chr22)
    
    with open(input_vcf, 'r') as infile, open(output_vcf, 'w') as outfile:
        for line in infile:
            if line.startswith("##contig="):
                # Extract contig ID using regex
                match = re.search(r'##contig=<ID=([^,>]+)', line)
                if match:
                    contig_id = match.group(1)
                    if contig_id not in valid_contigs:
                        continue  # Skip writing this line if it's not in chr1-chr22

            # Write all other lines
            outfile.write(line)

# Example usage:
# filter_vcf_contigs("input.vcf", "filtered_output.vcf")

def run_whatshap(vcf, ref, bam, phase,outdir):
    """Run the phasing pipeline using provided parameters."""
    # Create output directory if it doesn't exist
    os.makedirs(phase, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # Step 0: Filter VCF by quality (keep header lines and variants with QUAL >= 10)
    filter_cmd = (
        f"awk 'BEGIN {{OFS=\"\\t\"}} /^#/ {{print; next}} $6 >= 10 {{print}}' "
        f"\"{vcf}\" > \"{vcf}_filtered.vcf\""
    )
    run_command(filter_cmd)
    filter_vcf_contigs(f'{vcf}_filtered.vcf', f'{vcf}_filtered2.vcf')
    # Step 1: Perform Whatshap Phasing using high-confidence variants
    phase_cmd = (
        f"whatshap phase --ignore-read-groups --reference \"{ref}\" "
        f"-o \"{outdir}/whatshap.vcf\" "
        f"\"{vcf}_filtered2.vcf\" \"{bam}\""
    )
    run_command(phase_cmd)

    # Step 2: Compress and index the phased VCF
    compress_cmd = f"bgzip -c \"{outdir}/whatshap.vcf\" > \"{outdir}/whatshap.vcf.gz\""
    run_command(compress_cmd)
    index_vcf_cmd = f"tabix -p vcf \"{outdir}/whatshap.vcf.gz\""
    run_command(index_vcf_cmd)

    # Step 3: Haplotag BAM file using Whatshap
    haplotag_cmd = (
        f"whatshap haplotag --skip-missing-contigs --output-haplotag-list \"{outdir}/haplotagged.list\" "
        f"--ignore-read-groups --reference \"{ref}\" "
        f"-o \"{outdir}/haplotagged.bam\" "
        f"\"{outdir}/whatshap.vcf.gz\" \"{bam}\""
    )
    run_command(haplotag_cmd)

    # Step 4: Split BAM into two haplotypes using Whatshap
    split_cmd = (
        f"whatshap split --output-h1 \"{outdir}/h1.bam\" "
        f"--output-h2 \"{outdir}/h2.bam\" "
        f"\"{outdir}/haplotagged.bam\" \"{outdir}/haplotagged.list\""
    )
    run_command(split_cmd)

    # Step 5: Index the haplotype-specific BAM files using SAMtools
    index_h1_cmd = f"samtools index \"{outdir}/h1.bam\""
    run_command(index_h1_cmd)
    index_h2_cmd = f"samtools index \"{outdir}/h2.bam\""
    run_command(index_h2_cmd)

def main():
    args = parse_arguments()
    run_whatshap(vcf=args.vcf, ref=args.ref, bam=args.bam, phase=args.phase,outdir=args.outdir)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit status {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

