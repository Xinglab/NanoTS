import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
class DNASeqCoverageModel(nn.Module):
    def __init__(self):
        super(DNASeqCoverageModel, self).__init__()

        # Combined Conv2D for all inputs (5 channels)
        self.combined_conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=(4, 3), padding=(0, 1))
        self.combined_pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.combined_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.combined_pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Dropout for convolutional layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

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
        combined_dim = 64 * 15 + 16 + 16 + 16 + 8
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Two separate output layers for SNV and zygosity predictions
        self.fc3_snv = nn.Linear(64, 1)  # For SNV prediction
        self.fc4_zyg = nn.Linear(64, 1)  # For zygosity prediction

        # Dropout for fully connected layers
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

    def forward(self, one_hot_seq, ref_frac_fwd, ref_frac_rev, alt_frac_fwd, alt_frac_rev,
                ref_cov_fwd, ref_cov_rev, alt_cov_fwd, alt_cov_rev, base_entropy, dist_value):
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

        # Concatenate all processed features
        combined_features = torch.cat((x, ref_cov, alt_cov, base_entropy, dist_value), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)

        # Two separate outputs
        snv_output = torch.sigmoid(self.fc3_snv(x))
        zygosity_output = torch.sigmoid(self.fc4_zyg(x))

        return snv_output, zygosity_output


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
    dist_tag=['ref_forward_start_dist', 'ref_forward_end_dist','ref_reverse_start_dist', 'ref_reverse_end_dist','alt_forward_start_dist', 'alt_forward_end_dist','alt_reverse_start_dist', 'alt_reverse_end_dist']
    dist_list=[]
    for d in data:
        dist_list.append([d[i] for i in dist_tag])
    #print(dist_list)
    one_hot_seq = torch.tensor([d['one_hot_encoded_sequence'] for d in data], dtype=torch.float32)
    ref_frac_fwd = torch.tensor([d['ref_base_fractions_forward'] for d in data], dtype=torch.float32)
    ref_frac_rev = torch.tensor([d['ref_base_fractions_reverse'] for d in data], dtype=torch.float32)
    alt_frac_fwd = torch.tensor([d['alt_base_fractions_forward'] for d in data], dtype=torch.float32)
    alt_frac_rev = torch.tensor([d['alt_base_fractions_reverse'] for d in data], dtype=torch.float32)
    ref_cov_fwd = torch.tensor([d['ref_coverage_forward'] for d in data], dtype=torch.float32)
    ref_cov_rev = torch.tensor([d['ref_coverage_reverse'] for d in data], dtype=torch.float32)
    alt_cov_fwd = torch.tensor([d['alt_coverage_forward'] for d in data], dtype=torch.float32)
    alt_cov_rev = torch.tensor([d['alt_coverage_reverse'] for d in data], dtype=torch.float32)
    total_frac= torch.tensor([d['total_base_fraction'] for d in data], dtype=torch.float32)
    base_entropy=torch.tensor([d['base_entropy'] for d in data], dtype=torch.float32)
    
    dist_value=torch.tensor(dist_list, dtype=torch.float32)
    return one_hot_seq, ref_frac_fwd, ref_frac_rev, alt_frac_fwd, alt_frac_rev, ref_cov_fwd, ref_cov_rev, alt_cov_fwd, alt_cov_rev,total_frac,base_entropy,dist_value

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
def run_predict_dnn(args):
    # Load input data
    with open(args.input_file, "rb") as file:
        predict_list = list(pickle.load(file).values())
    
    # Load variant DataFrame
    variant_df = pd.read_csv(args.variant_file, sep='\t')

    # Prepare data
    print(f'Predicting {len(predict_list)} SNVs')
    test_one_hot_seq, test_ref_frac_fwd, test_ref_frac_rev, test_alt_frac_fwd, test_alt_frac_rev, test_ref_cov_fwd, test_ref_cov_rev, test_alt_cov_fwd, test_alt_cov_rev,test_total_frac,test_base_entropy,test_dist_value = prepare_data(predict_list)

    read_prime=extract_read_end(test_dist_value)
    # Load the model
    model = DNASeqCoverageModel().to('cpu')
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    print("Model loaded for predictions.")

    # Run predictions
    with torch.no_grad():
        snv_output, zygosity_output = model(test_one_hot_seq, test_ref_frac_fwd, test_ref_frac_rev, test_alt_frac_fwd, test_alt_frac_rev, test_ref_cov_fwd, test_ref_cov_rev, test_alt_cov_fwd, test_alt_cov_rev,test_base_entropy,test_dist_value)
        predicted_snv_labels = (snv_output > 0.5).float()
        predicted_zyg_labels = (zygosity_output > 0.5).float()
    #print(predictions)
    # Save predictions
    Pvalue_snv=[i[0] for i in snv_output.tolist()]
    Pvalue_zyg=[i[0] for i in zygosity_output.tolist()]
    #print(len(Pvalue))
    variant_df['End_dist']=read_prime
    variant_df.loc[:,'SNV_density']=cal_SNV_density(test_base_entropy)
    total_coverage=cal_total_coverage(test_ref_cov_fwd, test_ref_cov_rev, test_alt_cov_fwd, test_alt_cov_rev)
    variant_df.loc[:,'SNV_score']=cal_match_score(test_total_frac,test_one_hot_seq,total_coverage)

    variant_df['PredP_snv'] = Pvalue_snv
    variant_df['PredP_heterzygosity'] = Pvalue_zyg
    variant_df['Result_snv'] = [i[0] for i in predicted_snv_labels.tolist()]
    variant_df['Result_heterzygosity'] = [i[0] for i in predicted_zyg_labels.tolist()]
    variant_df.to_csv(args.output_file, index=False, sep='\t')
    print("Predictions saved to", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DNA sequence coverage model predictions.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input .pkl file")
    parser.add_argument("--variant_file", type=str, required=True, help="Path to the variant .txt file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model .pth file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions in .txt format")
    args = parser.parse_args()

    run_predict_dnn(args)


