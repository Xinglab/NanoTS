import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import math
import pysam

from scipy.stats import fisher_exact

def convert_to_vcf(df, output_file,contigs):
    """
    Convert a DataFrame to a VCF file with GT:GQ:DP:AD:AF format and scaled QUAL.

    Args:
        df (pd.DataFrame): Input DataFrame with variant information.
        output_file (str): Path to save the VCF file.
    """
    with open(output_file, 'w') as vcf:
        # Write VCF header
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write("##source=nanoTS\n")

        for chrom, length in contigs.items():
            vcf.write(f"##contig=<ID={chrom},length={length}>\n")
        vcf.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Read depth (total reads)\">\n")
        vcf.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">\n")
        vcf.write("##INFO=<ID=G0,Number=1,Type=Float,Description=\"Genotype probability for 0/0\">\n")
        vcf.write("##INFO=<ID=G1,Number=1,Type=Float,Description=\"Genotype probability for 0/1\">\n")
        vcf.write("##INFO=<ID=G2,Number=1,Type=Float,Description=\"Genotype probability for 1/1\">\n")
        vcf.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        vcf.write("##FORMAT=<ID=GQ,Number=1,Type=Float,Description=\"Genotype Quality\">\n")
        vcf.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n")
        vcf.write("##FORMAT=<ID=AD,Number=2,Type=Integer,Description=\"Allelic depths for the ref and alt alleles\">\n")
        vcf.write("##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Extract required fields
            chrom = row['chrom']
            pos = int(row['pos'])
            ref = row['ref']
            alt = row['alt']
            
            # Calculate QUAL (scaled max probability)
            genotype_probs = [row['genotype_0'], row['genotype_1'], row['genotype_2']]
            max_prob = max(genotype_probs)
            error_prob = 1 - max_prob
            if error_prob > 0:
                qual = -10 * math.log10(error_prob)
            else:
                qual = 99  # Set a high value for perfect confidence
            
            # Set GQ to be the same as QUAL
            gq = qual
            
            # Determine the FILTER status
            if max_prob == row['genotype_1'] or max_prob == row['genotype_2']:
                filter_status = "PASS"
            else:
                filter_status = "HomRef"
            
            # INFO field
            dp = row['ref_reads'] + row['alt_reads']
            af = row['ratio_1']
            info = (
                f"DP={dp};AF={af:.3f};"
                f"G0={row['genotype_0']:.3f};G1={row['genotype_1']:.3f};G2={row['genotype_2']:.3f}"
            )
            
            # FORMAT and SAMPLE fields
            fmt = "GT:GQ:DP:AD:AF"
            gt = (
                "0/0" if row['Result_genotype'] == 0 else
                "0/1" if row['Result_genotype'] == 1 else
                "1/1" if row['Result_genotype'] == 2 else "./."
            )
            ad = f"{row['ref_reads']},{row['alt_reads']}"  # Allelic Depths
            af_sample = f"{af:.3f}"  # Allele Frequency
            sample = f"{gt}:{gq:.2f}:{dp}:{ad}:{af_sample}"
            
            # Write the VCF row
            vcf.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual:.2f}\t{filter_status}\t{info}\t{fmt}\t{sample}\n")






class DNASeqCoverageModel_phase(nn.Module):
    def __init__(self):
        super(DNASeqCoverageModel_phase, self).__init__()

        # Combined Conv2D for all inputs (5 channels)
        self.combined_conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(4, 3), padding=(0, 1))
        self.combined_pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.combined_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.combined_pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Dropout for convolutional layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Combined Conv2D for SNV inputs (4 channels)
        self.snv_conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 1), padding=(0, 0))
        self.snv_pool1 = nn.MaxPool2d(kernel_size=(1, 20))
        self.snv_fc1=nn.Linear(8,20)
        # Dropout for convolutional layers
        #self.snv_dropout1 = nn.Dropout(p=0.3)
        
        # Dense layers for reference allele coverage
        self.ref_cov_fc1_fwd = nn.Linear(61, 32)
        self.ref_cov_fc1_rev = nn.Linear(61, 32)
        self.ref_cov_fc2 = nn.Linear(32 * 2, 16)

        # Dense layers for alternative allele coverage
        self.alt_cov_fc1_fwd = nn.Linear(61, 32)
        self.alt_cov_fc1_rev = nn.Linear(61, 32)
        self.alt_cov_fc2 = nn.Linear(32 * 2, 16)

        # Dense layers for base entropy
        self.base_entropy1 = nn.Linear(61, 32)
        self.base_entropy2 = nn.Linear(32, 16)

        # Dense layers for distance of read end to SNV
        self.dist_end1 = nn.Linear(8, 8)

        # Fully connected layers for combining features
        combined_dim = (64 * 15 + 16 + 16 + 16 + 8 +1 + 20)*3
        combined_dim_1=64 * 15 + 16 + 16 + 16 + 8 +1 + 20
        self.fc1 = nn.Linear(combined_dim_1, 128)
        self.fc2 = nn.Linear(128 *3, 128)
        self.fc3 = nn.Linear(128, 64)
        # Single output layer for softmax (3 classes for 0, 1, 2)
        self.fc_output = nn.Linear(64, 3)

        # Dropout for fully connected layers
        self.dropout3 = nn.Dropout(p=1)
        self.dropout4 = nn.Dropout(p=0.3)
    def prepare_combine(self, one_hot_seq, ref_frac_fwd, ref_frac_rev, alt_frac_fwd, alt_frac_rev,
                ref_cov_fwd, ref_cov_rev, alt_cov_fwd, alt_cov_rev, base_entropy, dist_value,coverage,
               ref_nearest_snvs_counts_forward,ref_nearest_snvs_counts_reverse,alt_nearest_snvs_counts_forward,alt_nearest_snvs_counts_reverse):

        combined_input = torch.stack([one_hot_seq, ref_frac_fwd, ref_frac_rev, alt_frac_fwd, alt_frac_rev], dim=1)
        
        # Process combined inputs with Conv2D and pooling
        x = F.relu(self.combined_conv1(combined_input))
        x = self.combined_pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.combined_conv2(x))
        x = self.combined_pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        
        # Process reference allele coverage
        ref_cov_fwd = F.relu(self.ref_cov_fc1_fwd(ref_cov_fwd))
        ref_cov_rev = F.relu(self.ref_cov_fc1_rev(ref_cov_rev))
        ref_cov = torch.cat((ref_cov_fwd, ref_cov_rev), dim=1)
        ref_cov = F.relu(self.ref_cov_fc2(ref_cov))
        
        # Process alternative allele coverage
        alt_cov_fwd = F.relu(self.alt_cov_fc1_fwd(alt_cov_fwd))
        alt_cov_rev = F.relu(self.alt_cov_fc1_rev(alt_cov_rev))
        alt_cov = torch.cat((alt_cov_fwd, alt_cov_rev), dim=1)
        alt_cov = F.relu(self.alt_cov_fc2(alt_cov))
        
        # Process base entropy
        base_entropy = F.relu(self.base_entropy1(base_entropy))
        base_entropy = F.relu(self.base_entropy2(base_entropy))
        
        # Process dist end
        dist_value = F.relu(self.dist_end1(dist_value))
        
        # Process nearest_snvs
        combined_snv=torch.stack([ref_nearest_snvs_counts_forward,ref_nearest_snvs_counts_reverse,alt_nearest_snvs_counts_forward,alt_nearest_snvs_counts_reverse],dim=1)
        combined_snv = F.relu(self.snv_conv1(combined_snv))
        combined_snv = self.snv_pool1(combined_snv)
        
        combined_snv = combined_snv.view(combined_snv.size(0),-1)
        combined_snv = F.relu(self.snv_fc1(combined_snv))
        
        #combined_initial=torch.cat((x, ref_cov, alt_cov, base_entropy, dist_value,coverage), dim=1)
        #combined_initial=self.dropout3(combined_initial)
        
        # Concatenate all processed features
        #combined_features = torch.cat((combined_initial,combined_snv), dim=1)
        combined_features = torch.cat((x, ref_cov, alt_cov, base_entropy, dist_value,coverage,combined_snv), dim=1) 
        combined_features=F.relu(self.fc1(combined_features))
        return combined_features 
    def forward(self, R_one_hot_seq, R_ref_frac_fwd, R_ref_frac_rev, R_alt_frac_fwd, R_alt_frac_rev,
                R_ref_cov_fwd, R_ref_cov_rev, R_alt_cov_fwd, R_alt_cov_rev, R_base_entropy, R_dist_value,R_coverage,
               R_ref_nearest_snvs_counts_forward,R_ref_nearest_snvs_counts_reverse,R_alt_nearest_snvs_counts_forward,R_alt_nearest_snvs_counts_reverse,
               H1_one_hot_seq, H1_ref_frac_fwd, H1_ref_frac_rev, H1_alt_frac_fwd, H1_alt_frac_rev,
                H1_ref_cov_fwd, H1_ref_cov_rev, H1_alt_cov_fwd, H1_alt_cov_rev, H1_base_entropy, H1_dist_value,H1_coverage,
               H1_ref_nearest_snvs_counts_forward,H1_ref_nearest_snvs_counts_reverse,H1_alt_nearest_snvs_counts_forward,H1_alt_nearest_snvs_counts_reverse,
               H2_one_hot_seq, H2_ref_frac_fwd, H2_ref_frac_rev, H2_alt_frac_fwd, H2_alt_frac_rev,
                H2_ref_cov_fwd, H2_ref_cov_rev, H2_alt_cov_fwd, H2_alt_cov_rev, H2_base_entropy, H2_dist_value,H2_coverage,
               H2_ref_nearest_snvs_counts_forward,H2_ref_nearest_snvs_counts_reverse,H2_alt_nearest_snvs_counts_forward,H2_alt_nearest_snvs_counts_reverse):

        R_combine=self.prepare_combine(R_one_hot_seq, R_ref_frac_fwd, R_ref_frac_rev, R_alt_frac_fwd, R_alt_frac_rev,
                R_ref_cov_fwd, R_ref_cov_rev, R_alt_cov_fwd, R_alt_cov_rev, R_base_entropy, R_dist_value,R_coverage,
               R_ref_nearest_snvs_counts_forward,R_ref_nearest_snvs_counts_reverse,R_alt_nearest_snvs_counts_forward,R_alt_nearest_snvs_counts_reverse)

        H1_combine=self.prepare_combine(H1_one_hot_seq, H1_ref_frac_fwd, H1_ref_frac_rev, H1_alt_frac_fwd, H1_alt_frac_rev,
                H1_ref_cov_fwd, H1_ref_cov_rev, H1_alt_cov_fwd, H1_alt_cov_rev, H1_base_entropy, H1_dist_value,H1_coverage,
               H1_ref_nearest_snvs_counts_forward,H1_ref_nearest_snvs_counts_reverse,H1_alt_nearest_snvs_counts_forward,H1_alt_nearest_snvs_counts_reverse)

        H2_combine=self.prepare_combine(H2_one_hot_seq, H2_ref_frac_fwd, H2_ref_frac_rev, H2_alt_frac_fwd, H2_alt_frac_rev,
                H2_ref_cov_fwd, H2_ref_cov_rev, H2_alt_cov_fwd, H2_alt_cov_rev, H2_base_entropy, H2_dist_value,H2_coverage,
               H2_ref_nearest_snvs_counts_forward,H2_ref_nearest_snvs_counts_reverse,H2_alt_nearest_snvs_counts_forward,H2_alt_nearest_snvs_counts_reverse)

        combined_features=torch.cat((R_combine, H1_combine,H2_combine), dim=1)  
        # Fully connected layers
        x = F.relu(self.fc2(combined_features))
        #x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout4(x)

        # Softmax output for 3 classes
        output = self.fc_output(x)

        return output







def recognize_exon_body(coverage, threshold=0.2, min_exon_length=3):
    """
    Recognize exon bodies based on significant changes in read coverage.

    Args:
        coverage (list or np.ndarray): List of read coverage values scaled from 0 to 1.
        threshold (float): Minimum coverage value to consider as part of an exon body.
        min_exon_length (int): Minimum length for an exon body (number of continuous points above the threshold).

    Returns:
        list of tuple: A list of (start_idx, end_idx) pairs indicating the indices of exon bodies.
    """
    if not isinstance(coverage, (list, np.ndarray)):
        raise ValueError("Input coverage must be a list or numpy array.")
    
    coverage = np.array(coverage)
    exon_bodies = []
    in_exon = False
    start_idx = None

    for idx, value in enumerate(coverage):
        if value >= threshold:
            if not in_exon:  # Start of a new exon
                in_exon = True
                start_idx = idx
        else:
            if in_exon:  # End of the current exon
                in_exon = False
                if start_idx is not None and idx - start_idx >= min_exon_length:
                    exon_bodies.append((start_idx, idx - 1))
                start_idx = None

    # Handle the case where the last point is part of an exon
    if in_exon and start_idx is not None and len(coverage) - start_idx >= min_exon_length:
        exon_bodies.append((start_idx, len(coverage) - 1))

    return exon_bodies


def normalize_match_score(pass_idx_pos,match_score):
    len_sum=0
    score_sum=0
    for i in pass_idx_pos:
        score_sum+=sum(match_score[i[0]:(i[1]+1)])
        len_sum+=i[1]-i[0]+1
    if len_sum==0:
        return 0
    return(score_sum/len_sum)

def cal_match_score(test_total_frac,test_one_hot_seq,total_coverage):
    test_total_frac=test_total_frac.tolist()
    test_one_hot_seq=test_one_hot_seq.tolist()
    #total_coverage=total_coverage.tolist()
    out_list=[]
    for i in range(len(test_total_frac)):
        ohe=np.array(test_one_hot_seq[i])
        tbf=np.array(test_total_frac[i])
        #print(ohe)
        #print(tbf)
        match_score=[ ohe[j] * tbf[j] for j in range(4)]
        match_score=np.sum(match_score,axis= 0)
        #pass_idx=np.sum(tbf,axis=0)>0
        pass_idx_pos=recognize_exon_body(total_coverage[i])

        #print(match_score)
        #out_list.append(sum(np.sum(match_score,axis= 0))/sum(pass_idx))
        out_list.append(normalize_match_score(pass_idx_pos,match_score))
    return out_list
def cal_total_coverage(test_ref_cov_fwd, test_ref_cov_rev, test_alt_cov_fwd, test_alt_cov_rev):
    test_ref_cov_fwd=test_ref_cov_fwd.tolist()
    test_ref_cov_rev=test_ref_cov_rev.tolist()
    test_alt_cov_fwd=test_alt_cov_fwd.tolist()
    test_alt_cov_rev=test_alt_cov_rev.tolist()
    out_list=[]
    for i in range(len(test_ref_cov_fwd)):
        total_cov=np.array(test_ref_cov_fwd[i])+np.array(test_ref_cov_rev[i])+np.array(test_alt_cov_fwd[i])+np.array(test_alt_cov_rev[i])
        out_list.append(total_cov)
    return(out_list)

def prepare_data(data):
    dist_tag = [
        'ref_forward_start_dist', 'ref_forward_end_dist', 'ref_reverse_start_dist', 'ref_reverse_end_dist',
        'alt_forward_start_dist', 'alt_forward_end_dist', 'alt_reverse_start_dist', 'alt_reverse_end_dist'
    ]

    # Convert to NumPy arrays first (efficient batching)
    dist_list = np.array([[d[tag] for tag in dist_tag] for d in data], dtype=np.float32)
    coverage_list = np.array([[1] if d['max_coverage'] >= 50 else [d['max_coverage'] / 50] for d in data], dtype=np.float32)
    
    one_hot_seq = np.array([d['one_hot_encoded_sequence'] for d in data], dtype=np.float32)
    ref_frac_fwd = np.array([d['ref_base_fractions_forward'] for d in data], dtype=np.float32)
    ref_frac_rev = np.array([d['ref_base_fractions_reverse'] for d in data], dtype=np.float32)
    alt_frac_fwd = np.array([d['alt_base_fractions_forward'] for d in data], dtype=np.float32)
    alt_frac_rev = np.array([d['alt_base_fractions_reverse'] for d in data], dtype=np.float32)
    
    ref_cov_fwd = np.array([d['ref_coverage_forward'] for d in data], dtype=np.float32)
    ref_cov_rev = np.array([d['ref_coverage_reverse'] for d in data], dtype=np.float32)
    alt_cov_fwd = np.array([d['alt_coverage_forward'] for d in data], dtype=np.float32)
    alt_cov_rev = np.array([d['alt_coverage_reverse'] for d in data], dtype=np.float32)
    
    total_frac = np.array([d['total_base_fraction'] for d in data], dtype=np.float32)
    base_entropy = np.array([d['base_entropy'] for d in data], dtype=np.float32)

    ref_snvs_fractions_forward = np.array([d['ref_snvs_fractions_forward'] for d in data], dtype=np.float32)
    ref_snvs_fractions_reverse = np.array([d['ref_snvs_fractions_reverse'] for d in data], dtype=np.float32)
    alt_snvs_fractions_forward = np.array([d['alt_snvs_fractions_forward'] for d in data], dtype=np.float32)
    alt_snvs_fractions_reverse = np.array([d['alt_snvs_fractions_reverse'] for d in data], dtype=np.float32)

    # Convert all NumPy arrays to PyTorch tensors (faster)
    return (
        torch.from_numpy(one_hot_seq),
        torch.from_numpy(ref_frac_fwd), torch.from_numpy(ref_frac_rev),
        torch.from_numpy(alt_frac_fwd), torch.from_numpy(alt_frac_rev),
        torch.from_numpy(ref_cov_fwd), torch.from_numpy(ref_cov_rev),
        torch.from_numpy(alt_cov_fwd), torch.from_numpy(alt_cov_rev),
        torch.from_numpy(total_frac), torch.from_numpy(base_entropy),
        torch.from_numpy(dist_list), torch.from_numpy(coverage_list),
        torch.from_numpy(ref_snvs_fractions_forward), torch.from_numpy(ref_snvs_fractions_reverse),
        torch.from_numpy(alt_snvs_fractions_forward), torch.from_numpy(alt_snvs_fractions_reverse)
    )
def extract_SNV_target_coverage(data, test_idx):
    selected = [data[i] for i in test_idx]
    ref = np.array([d['ref_coverage_forward'][30] * d['max_coverage'] + d['ref_coverage_reverse'][30] * d['max_coverage'] for d in selected], dtype=np.float32)
    alt = np.array([d['alt_coverage_forward'][30] * d['max_coverage'] + d['alt_coverage_reverse'][30] * d['max_coverage']for d in selected], dtype=np.float32)
    return ref, alt
   
def adjust_homo_alt(d1, d2, test_idx):
    ref_h1, alt_h1 = extract_SNV_target_coverage(d1, test_idx)
    ref_h2, alt_h2 = extract_SNV_target_coverage(d2, test_idx)

    p_values = []

    for i in range(len(test_idx)):
        contingency_table = [[ref_h1[i], alt_h1[i]],
                             [ref_h2[i], alt_h2[i]]]
        _, p_value = fisher_exact(contingency_table)
        p_values.append(p_value)

    return p_values

def cal_SNV_density(base_entropy):
    base_entropy=base_entropy.tolist()
    out_list=[]
    for i in base_entropy:
        nbe=(0.5 < np.array(i) ) & ( np.array(i) <=2)
        out_list.append(sum(nbe))
    return out_list

def extract_read_end(test_dist_value):
    # Convert to list if needed
    test_dist_value = test_dist_value.tolist()
    out_list = []

    for i in test_dist_value:
        arr_alt = i[4:]  
        arr_ref = i[:4] 
        if (0 <= arr_alt[0] <= 0.5 and 0 <= arr_alt[2] <= 0.5) or (0 <= arr_ref[0] <= 0.5 and 0 <= arr_ref[2] <= 0.5):
            out_list.append('prime5end')
        elif (0 <= arr_alt[1] <= 0.5 and 0 <= arr_alt[3] <= 0.5) or ( 0 <= arr_ref[1] <= 0.5 and 0 <= arr_ref[3] <= 0.5):
            out_list.append('prime3end')
        else:
            out_list.append('pass')

    return out_list

def organize_variant_phase(raw_list,h1_list,h2_list,variant_df,model):
    R_one_hot_seq, R_ref_frac_fwd, R_ref_frac_rev, R_alt_frac_fwd, R_alt_frac_rev, R_ref_cov_fwd, R_ref_cov_rev, R_alt_cov_fwd, R_alt_cov_rev,R_total_frac,R_base_entropy,R_dist_value,R_coverage,R_ref_nearest_snvs_fractions_forward,R_ref_nearest_snvs_fractions_reverse,R_alt_nearest_snvs_fractions_forward,R_alt_nearest_snvs_fractions_reverse  = prepare_data(raw_list)
    H1_one_hot_seq, H1_ref_frac_fwd, H1_ref_frac_rev, H1_alt_frac_fwd, H1_alt_frac_rev, H1_ref_cov_fwd, H1_ref_cov_rev, H1_alt_cov_fwd, H1_alt_cov_rev,H1_total_frac,H1_base_entropy,H1_dist_value,H1_coverage,H1_ref_nearest_snvs_fractions_forward,H1_ref_nearest_snvs_fractions_reverse,H1_alt_nearest_snvs_fractions_forward,H1_alt_nearest_snvs_fractions_reverse = prepare_data(h1_list)
    H2_one_hot_seq, H2_ref_frac_fwd, H2_ref_frac_rev, H2_alt_frac_fwd, H2_alt_frac_rev, H2_ref_cov_fwd, H2_ref_cov_rev, H2_alt_cov_fwd, H2_alt_cov_rev,H2_total_frac,H2_base_entropy,H2_dist_value,H2_coverage,H2_ref_nearest_snvs_fractions_forward,H2_ref_nearest_snvs_fractions_reverse,H2_alt_nearest_snvs_fractions_forward,H2_alt_nearest_snvs_fractions_reverse = prepare_data(h2_list)
    
    read_prime=extract_read_end(R_dist_value)
   

    # Run predictions
    with torch.no_grad():
        pred_genotype = model( R_one_hot_seq, R_ref_frac_fwd, R_ref_frac_rev, R_alt_frac_fwd, R_alt_frac_rev,
            R_ref_cov_fwd, R_ref_cov_rev, R_alt_cov_fwd, R_alt_cov_rev, R_base_entropy, R_dist_value,R_coverage,
           R_ref_nearest_snvs_fractions_forward,R_ref_nearest_snvs_fractions_reverse,R_alt_nearest_snvs_fractions_forward,R_alt_nearest_snvs_fractions_reverse,
           H1_one_hot_seq, H1_ref_frac_fwd, H1_ref_frac_rev, H1_alt_frac_fwd, H1_alt_frac_rev,
            H1_ref_cov_fwd, H1_ref_cov_rev, H1_alt_cov_fwd, H1_alt_cov_rev, H1_base_entropy, H1_dist_value,H1_coverage,
           H1_ref_nearest_snvs_fractions_forward,H1_ref_nearest_snvs_fractions_reverse,H1_alt_nearest_snvs_fractions_forward,H1_alt_nearest_snvs_fractions_reverse,
           H2_one_hot_seq, H2_ref_frac_fwd, H2_ref_frac_rev, H2_alt_frac_fwd, H2_alt_frac_rev,
            H2_ref_cov_fwd, H2_ref_cov_rev, H2_alt_cov_fwd, H2_alt_cov_rev, H2_base_entropy, H2_dist_value,H2_coverage,
           H2_ref_nearest_snvs_fractions_forward,H2_ref_nearest_snvs_fractions_reverse,H2_alt_nearest_snvs_fractions_forward,H2_alt_nearest_snvs_fractions_reverse    )
        pred_genotype=torch.softmax(pred_genotype, dim=1)
        pred_genotype_prob= pred_genotype.cpu()
        pred_genotype = torch.argmax(pred_genotype_prob, dim=1).numpy()
        pred_genotype_prob= pred_genotype_prob.numpy()


    homo_alt_idx = (pred_genotype == 2) & (pred_genotype_prob[:, 2] < 0.9) & (pred_genotype_prob[:, 1] >pred_genotype_prob[:, 0])

    homo_alt_p_value = adjust_homo_alt(h1_list, h2_list, np.where(homo_alt_idx)[0])

    # Update genotype predictions where p-value < 0.01
    update_indices = np.where(homo_alt_idx)[0][np.array(homo_alt_p_value) < 0.01]

    #print(update_indices)
    pred_genotype[update_indices] = 1
    
    orig_g2=pred_genotype_prob[update_indices,2].copy()
    pred_genotype_prob[update_indices,2]=pred_genotype_prob[update_indices,1]
    pred_genotype_prob[update_indices,1]=orig_g2
    #print(predictions)
    # Save predictions
    predicted_snv_labels = (pred_genotype != 0).astype(int)  # 1 if genotype is 1 or 2, else 0
    # Compute zygosity labels
    predicted_zyg_labels = (pred_genotype == 1).astype(int)  # 1 if genotype is 1, else 0
    #print(len(Pvalue))
    variant_df['End_dist']=read_prime
    variant_df.loc[:,'SNV_density']=cal_SNV_density(R_base_entropy)
    total_coverage=cal_total_coverage(R_ref_cov_fwd, R_ref_cov_rev, R_alt_cov_fwd, R_alt_cov_rev)
    variant_df.loc[:,'SNV_score']=cal_match_score(R_total_frac,R_one_hot_seq,total_coverage)


    variant_df['genotype_0']=[i[0] for i in pred_genotype_prob]
    variant_df['genotype_1']=[i[1] for i in pred_genotype_prob]
    variant_df['genotype_2']=[i[2] for i in pred_genotype_prob]
    
    variant_df['Result_snv'] = predicted_snv_labels.tolist()
    variant_df['Result_heterzygosity'] = predicted_zyg_labels.tolist()
    variant_df['Result_genotype'] = pred_genotype.tolist()
    
    variant_df['chrom'] = variant_df['chrom'].astype(str)
    variant_df['pos'] = variant_df['pos'].astype(int)
    variant_df = variant_df.sort_values(by=['chrom', 'pos'], ascending=[True, True])
    variant_df = variant_df.reset_index(drop=True)
    return(variant_df)
    
def run_predict_dnn_phase(raw_pkl,h1_pkl,h2_pkl,variant_file,model_path,output_file,bam_path,target_chr=[]):
    if bam_path:
        bam = pysam.AlignmentFile(bam_path, "rb")
        contigs = {ref["SN"]: ref["LN"] for ref in bam.header["SQ"]}
    else:
        # hg38
        contigs = {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
        "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415}
        
    # Load the model
    model = DNASeqCoverageModel_phase().to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded for predictions.")
    if len(target_chr)>0:
        variant_df_list = []
        for i in target_chr:
            print(f'Start processing chr{i}')
            chrom=f'.chr{i}'
            # Load input data
            with open(raw_pkl+chrom, "rb") as file:
                raw_list = list(pickle.load(file).values())
            with open(h1_pkl+chrom, "rb") as file:
                h1_list = list(pickle.load(file).values())
            with open(h2_pkl+chrom, "rb") as file:
                h2_list = list(pickle.load(file).values())
            # Load variant DataFrame
            variant_df_each = pd.read_csv(variant_file+chrom, sep='\t')

            # Prepare data
            print(f'Predicting {len(raw_list)} SNVs')
            if len(raw_list)>0:
                variant_df_each=organize_variant_phase(raw_list,h1_list,h2_list,variant_df_each,model)
                variant_df_list.append(variant_df_each)
        variant_df = pd.concat(variant_df_list, axis=0, ignore_index=True)
    else:
        print(f'Start processing')
        with open(raw_pkl, "rb") as file:
            raw_list = list(pickle.load(file).values())
        with open(h1_pkl, "rb") as file:
            h1_list = list(pickle.load(file).values())
        with open(h2_pkl, "rb") as file:
            h2_list = list(pickle.load(file).values())
        variant_df = pd.read_csv(variant_file, sep='\t')
        # Prepare data
        print(f'Predicting {len(raw_list)} SNVs')
        if len(raw_list)>0:
            variant_df=organize_variant_phase(raw_list,h1_list,h2_list,variant_df,model)
            
    variant_df.to_csv(output_file, index=False, sep='\t')
    variant_df_pass=variant_df.loc[variant_df.Result_genotype>0,:]
    
    
    print("Predictions details saved to", output_file)
    filename = output_file
    if filename.endswith(".txt"):
        filename_no_extension = filename[:-4]  # Remove the last 4 characters
    else:
        filename_no_extension = filename
    variant_df_pass.to_csv(filename_no_extension+'.pass.txt', index=False, sep='\t')
    convert_to_vcf(variant_df, filename_no_extension+'.vcf',contigs)
    convert_to_vcf(variant_df_pass, filename_no_extension+'.pass.vcf.tmp',contigs)
    print("Predictions VCF saved to", filename_no_extension+'.pass.vcf.tmp')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DNA sequence coverage model predictions.")
    parser.add_argument("--raw_pkl", type=str, required=True, help="Path to the input original .pkl file")
    parser.add_argument("--h1_pkl", type=str, required=True, help="Path to the input H1 .pkl file")
    parser.add_argument("--h2_pkl", type=str, required=True, help="Path to the input H2 .pkl file")
    parser.add_argument("--variant_file", type=str, required=True, help="Path to the variant .txt file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the phased model .pth file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions in .txt format")
    parser.add_argument("--bam", type=str, required=False, help="Path to the alignment file")
    args = parser.parse_args()

    run_predict_dnn_phase(args.raw_pkl,args.h1_pkl,args.h2_pkl,args.variant_file,args.model_path,args.output_file,args.bam)


