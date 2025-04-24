import pysam
from collections import defaultdict
import argparse
import sys
import subprocess
import pickle
import argparse,sys
from contextlib import redirect_stdout

from nanoTS.bam2tpileup import process_bam
from nanoTS.snp_feature_extraction_multiple import extract_VCF_feature,filter_vcf
from nanoTS.predict_DNN import run_predict_dnn
from nanoTS.predict_DNN_phase import run_predict_dnn_phase
from nanoTS.whatshap import run_whatshap,run_command
from nanoTS.version import __version__  

import os 

import glob
import logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print logs to console
        # logging.FileHandler("nanoTS.log")  # Uncomment to enable file logging
    ]
)

def run_suffix_qname_by_alignment(bam_in_path, bam_out_path):
    logging.info(f"Process BAM file {bam_in_path}")
    if not os.path.exists(bam_in_path):
        logging.error(f"Error: Required file {bam_in_path} does not exist.")
        sys.exit(1)
    bam_in = pysam.AlignmentFile(bam_in_path, "rb")
    bam_out = pysam.AlignmentFile(bam_out_path, "wb", template=bam_in)

    qname_counts = defaultdict(int)

    for read in bam_in:
        qname = read.query_name
        qname_counts[qname] += 1
        read.query_name = f"{qname}_{qname_counts[qname]}"
        bam_out.write(read)

    bam_in.close()
    bam_out.close()
    logging.info(f"Index BAM file {bam_out_path}")
    # Index the new BAM
    pysam.index(bam_out_path)
    logging.info("Done")

def delete_related_files(outdir):
    """
    Deletes all related files in the specified output directory.
    
    Parameters:
        outdir (str): Path to the output directory.
    """
    # Define file patterns to match
    file_patterns = [
        "H1_features.pkl*", "H2_features.pkl*", "unphased_alt.5.txt*",
        "unphased_alt.txt*", "unphased_features.pkl*", "haplotagged.*",
        "h1.bam*", "h2.bam*", "whatshap*", "unphased_predict.pass.vcf_filtered.vcf",
        "unphased_candidates.txt","unphased_predict.pass.vcf_filtered2.vcf","phased_predict.pass.vcf.tmp"
    ]

    # Loop through each pattern and delete matching files
    for pattern in file_patterns:
        files_to_delete = glob.glob(os.path.join(outdir, pattern))
        for file in files_to_delete:
            try:
                os.remove(file)
                logging.info(f"Deleted: {file}")
            except Exception as e:
                logging.error(f"Error deleting {file}: {e}")
    
def run_phased_call(bam, ref, threads, depth, model, outdir):
    ensure_directory_exists(outdir)
    unphased_dir=hap_dir=outdir
    unphased_step2_feature_alt_file=os.path.join(unphased_dir,'unphased_alt.txt')
    unphased_step2_feature_alt_file_5 = os.path.join(unphased_dir, 'unphased_alt.5.txt')
    unphased_step2_feature_pkl_file = os.path.join(unphased_dir, 'unphased_features.pkl') 
    H1_feature_pkl_file = os.path.join(outdir, 'H1_features.pkl')
    H2_feature_pkl_file = os.path.join(outdir, 'H2_features.pkl') 
    H1_bam=os.path.join(hap_dir, 'h1.bam')
    H2_bam=os.path.join(hap_dir, 'h2.bam')
    
    # List of required files to check
    required_files = [
        H1_bam,
        H2_bam,
        model,
        bam
    ]
    
    # Check if each required file exists
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Error: Required file {file_path} does not exist.")
            sys.exit(1)
    required_chr_files=[
        unphased_step2_feature_alt_file,
        unphased_step2_feature_alt_file_5,
        unphased_step2_feature_pkl_file
    ]
    target_chr=[]
    for i in list(map(str, range(1, 23))) + ['X', 'Y']:
        chrom=f'.chr{i}'
        if os.path.exists(unphased_step2_feature_alt_file+chrom) and os.path.exists(unphased_step2_feature_alt_file_5+chrom) and os.path.exists(unphased_step2_feature_pkl_file+chrom):
            target_chr.append(i)
    logging.info(f"Processing chromosomes: {target_chr}")
    
    phased_predict_detail_file = os.path.join(outdir, 'phased_predict.txt')
    
    for i in target_chr:
        chrom=f'.chr{i}'
        logging.info(f'Start processing chrom {chrom} H1...')
        H1_feature_for_train = extract_VCF_feature( 
            unphased_step2_feature_alt_file+chrom, H1_bam, ref, threads, depth, unphased_step2_feature_alt_file_5+chrom
        )
        with open(H1_feature_pkl_file+chrom, "wb") as file:
            pickle.dump(H1_feature_for_train, file)
        logging.info(f'Start processing chrom {chrom} H2...')    
        H2_feature_for_train = extract_VCF_feature(
            unphased_step2_feature_alt_file+chrom, H2_bam, ref, threads, depth, unphased_step2_feature_alt_file_5+chrom
        )
        with open(H2_feature_pkl_file+chrom, "wb") as file:
            pickle.dump(H2_feature_for_train, file)
    
    logging.info("Run DNN model")
    run_predict_dnn_phase(unphased_step2_feature_pkl_file,H1_feature_pkl_file,H2_feature_pkl_file,
    unphased_step2_feature_alt_file,model,phased_predict_detail_file,bam,target_chr)
    
    logging.info("Phasing variants")
    tmp_pass_vcf=os.path.join(outdir,'phased_predict.pass.vcf.tmp')
    final_pass_vcf=os.path.join(outdir,'phased_predict.pass.vcf')
    phase_cmd = (
        f"whatshap phase --ignore-read-groups --reference \"{ref}\" "
        f"-o \"{final_pass_vcf}\" "
        f"\"{tmp_pass_vcf}\" \"{bam}\""
    )
    run_command(phase_cmd)
    
    logging.info("Done")
    
def run_haplotype(vcf, ref, bam,outdir,QUAL):
    ensure_directory_exists(outdir)
    # List of required files to check
    required_files = [
        vcf,
        ref,
        bam
    ]
    
    # Check if each required file exists
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Error: Required file {file_path} does not exist.")
            sys.exit(1)
    
    # Proceed with further processing if all files exist
    logging.info("All required files exist. Proceeding with haplotype processing...")
    
    run_whatshap(vcf, ref, bam, outdir,QUAL)
    logging.info("Done")

def loop_unphased_feature(step2_feature_alt_file,step2_feature_alt_file_5,step2_feature_pkl_file,filtered_variant,filtered_variant_5,bam, ref, threads,depth):
    target_chr=[]
    for i in list(map(str, range(1, 23))) + ['X', 'Y']:
        each_f_var_file=step2_feature_alt_file+f'.chr{i}'
        each_alt5_var=step2_feature_alt_file_5+f'.chr{i}'
        each_filtered_variant=filtered_variant.loc[filtered_variant.loc[:,'chrom']==f'chr{i}',:]
        if each_filtered_variant.shape[0]==0:
            continue
        logging.info(f"Start processing chr{i}")
        target_chr.append(i)
        each_filtered_variant.to_csv(each_f_var_file, sep='\t', index=False)       
        filtered_variant_5.loc[filtered_variant_5.loc[:,'chrom']==f'chr{i}',:].to_csv(each_alt5_var, sep='\t', index=False)       
         
        feature_for_train = extract_VCF_feature(each_f_var_file, bam, ref, threads,depth,each_alt5_var)
        with open(step2_feature_pkl_file+'.chr'+str(i), "wb") as file:
            pickle.dump(feature_for_train, file)
    return(target_chr)

def run_unphased_call(bam, ref, region, ALT, total, ratio, threads, depth, model, outdir):
    ensure_directory_exists(outdir)
    required_files = [
        bam,
        ref,
        model
    ]
    
    # Check if each required file exists
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Error: Required file {file_path} does not exist.")
            sys.exit(1)
    
    # Proceed with further processing if all files exist
    logging.info("All required files exist. Proceeding with unphased call processing...")
    
    
    """
    Run the unphased_call subcommand, which processes the BAM file,
    extracts features, and runs the deep learning prediction.
    """
    step1_candidate_out_file = os.path.join(outdir, 'unphased_candidates.txt') # step1
    step2_feature_pkl_file = os.path.join(outdir, 'unphased_features.pkl') # step2
    step2_feature_alt_file = os.path.join(outdir, 'unphased_alt.txt') # step2
    step2_feature_alt_file_5 = os.path.join(outdir, 'unphased_alt.5.txt') # step2
    step3_predict_detail_file = os.path.join(outdir, 'unphased_predict.txt')

    epsilon = 0.05 / 5

    # Step 1: Process BAM to generate candidate variants
    
    with open(step1_candidate_out_file, "w") as f:
        with redirect_stdout(f):
            process_bam(
                bam,
                ref,
                region,
                min(ALT, 5),
                min(total, 5),
                ratio,
                threads,
                epsilon
            )
    
    # Step 2: Filter VCF candidates
    filtered_variant = filter_vcf(
        step1_candidate_out_file,
        step2_feature_alt_file,
        ratio,
        0,
        0,
        ALT,
        total
    )
    filtered_variant_5 = filter_vcf(
        step1_candidate_out_file,
        step2_feature_alt_file_5,
        ratio,
        0,
        0,
        5,
        5
    )
    target_chr=loop_unphased_feature(step2_feature_alt_file,step2_feature_alt_file_5,step2_feature_pkl_file,filtered_variant,filtered_variant_5,bam, ref, threads,depth)
    
    matching_files = glob.glob(step2_feature_pkl_file+'*')

    if not matching_files:
        logging.info("No SNP candidates found")
    else:
        # Run DNN prediction
        logging.info("Run DNN model")
        run_predict_dnn(step2_feature_pkl_file, step2_feature_alt_file, model, step3_predict_detail_file, bam,target_chr)
        logging.info("Done")

def ensure_directory_exists(directory):
    """
    Ensure that the specified directory and its parent directories exist.
    If the directory doesn't exist, it will be created, including all child directories.
    
    Args:
        directory (str): Path to the directory to check or create.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        logging.info(f"Directory '{directory}' does not exist. Creating it...")
        os.makedirs(directory, exist_ok=True)
    else:
        logging.info(f"Directory '{directory}' already exists. Continuing...")
def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="nanoTS: Nanopore SNV calling using deep-learning method.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")
    
    # Subcommand: bam
    parser_bam = subparsers.add_parser("bam", help="Suffix each BAM alignment QNAME by count per read (e.g., read_1, read_2 for split reads).")
    parser_bam.add_argument("-i", "--input", required=True, help="Input sorted BAM file")
    parser_bam.add_argument("-o", "--output", required=True, help="Output BAM file with suffixed QNAMEs")
    
    # Subcommand: unphased_call
    parser_unphased = subparsers.add_parser("unphased_call", help="Run unphased variant calling")
    parser_unphased.add_argument("--bam", required=True, help="Path to input sorted and indexed BAM file.")
    parser_unphased.add_argument("--ref", required=True, help="Path to reference genome FASTA file.")
    parser_unphased.add_argument("--threads", type=int, default=24, help="Number of threads to use (default: 24).")
    parser_unphased.add_argument("--outdir", required=True, help="Path to output directory.")
    parser_unphased.add_argument("--region", help="Region to process (e.g., chr1 or chr1:1000-2000).")
    parser_unphased.add_argument("--ALT", type=int, default=2, help="Minimum ALT SNP coverage (default: 2).")
    parser_unphased.add_argument("--total", type=int, default=2, help="Minimum total base coverage (default: 2).")
    parser_unphased.add_argument("--ratio", type=float, default=0.05, help="Minimum ratio of ALT SNV (default: 0.05).")
    parser_unphased.add_argument("--depth", type=int, default=1000, help="Maximum reads to load for each variant. Set 0 to remove the limit (default: 1000).")
    parser_unphased.add_argument("--model", required=True, help="Path to the saved model .pth file.")

    # Subcommand: haplotype
    parser_haplotype = subparsers.add_parser("haplotype", help="Run haplotype phasing")
     
    parser_haplotype.add_argument("--vcf", required=True,
                        help="Path to the input VCF file.")
    parser_haplotype.add_argument("--ref", required=True,
                        help="Path to the reference genome.")
    parser_haplotype.add_argument("--bam", required=True,
                        help="Path to the input BAM file.")
    parser_haplotype.add_argument("--outdir", required=True,
                        help="Directory for haplotype output.")
    parser_haplotype.add_argument(
        "-q", "--QUAL",
        type=int,
        default=10,
        help="Minimum QUAL score to filter SNPs (default: 10)"
    )
                        

    # Subcommand: phased_call
    parser_phased = subparsers.add_parser("phased_call", help="Run phased variant calling")
    #parser_phased.add_argument("--unphased_dir", required=True, help="Path to unphased_call outdir.")
    #parser_phased.add_argument("--hap_dir", required=True, help="Path to haplotype outdir.")
    parser_phased.add_argument("--bam", required=True, help="Path to input sorted and indexed BAM file.")
    parser_phased.add_argument("--ref", required=True, help="Path to reference genome FASTA file.")
    parser_phased.add_argument("--threads", type=int, default=24, help="Number of threads to use (default: 24).")
    parser_phased.add_argument("--outdir", required=True, help="Path to output directory.")
    parser_phased.add_argument("--depth", type=int, default=1000, help="Maximum reads to load for each variant. Set 0 to remove the limit (default: 1000).")
    parser_phased.add_argument("--model", required=True, help="Path to the saved model .pth file.")

    # Subcommand: clean
    parser_clean = subparsers.add_parser("clean", help="Delete all temporary files")
    parser_clean.add_argument("--outdir", required=True, help="Path to result directory.")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.command == "bam":
        run_suffix_qname_by_alignment(args.input,args.output)  
    elif args.command == "unphased_call":
        run_unphased_call(
            bam=args.bam,
            ref=args.ref,
            region=args.region,
            ALT=args.ALT,
            total=args.total,
            ratio=args.ratio,
            threads=args.threads,
            depth=args.depth,
            model=args.model,
            outdir=args.outdir
        )
    elif args.command == "haplotype":
        run_haplotype(
            args.vcf, args.ref,args.bam,args.outdir,args.QUAL
        )
    elif args.command == "phased_call":
        run_phased_call(
        args.bam, args.ref, args.threads, args.depth, args.model, args.outdir
        )
    elif args.command == "clean":
        delete_related_files(args.outdir)
    else:
        print("Invalid command")
        sys.exit(1)
        

 



if __name__ == "__main__":
    main()

