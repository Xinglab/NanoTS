# **NanoTS**

**NanoTS** is a deep-learning-based variant caller for SNP detection from Nanopore transcriptome sequencing data. It provides efficient and accurate SNP calling from **long-read Nanopore RNA sequencing data**, supporting both **cDNA and direct RNA** technologies, particularly with the latest **R10.4 chemistry**.

---

## **Installation**

### Option 1: Conda
Ensure you have **Conda** installed, then create and activate the `nanoTS` environment:

```bash
git clone git@github.com:Xinglab/NanoTS.git # clone nanoTS
cd nanoTS
conda env create -f environment.yml # create nanoTS conda environment
conda activate nanoTS
```

Install `nanoTS`:

```bash
pip install .
```

Alternatively, run the NanoTS runner directly:

```bash
python ./source/nanoTS-runner.py [OPTIONS]
```
### Option 2: Singularity
Ensure you have **Singularity** installed, then pull the `nanoTS` container:

```bash
singularity pull library://zelinliu/nanots/nanots:latest

```


---

## **Usage**

NanoTS is a command-line tool with the following subcommands:

| Subcommand       | Description                                              |
|------------------|----------------------------------------------------------|
| **bam**          | Suffix each BAM alignment QNAME by count per read. |
| **unphased_call** | Extracts candidate SNPs and features from a BAM file.   |
| **haplotype**    | Performs haplotype phasing using a VCF, reference, and BAM file. |
| **phased_call**  | Processes phased SNP calls with the deep-learning model. |
| **clean**        | Removes temporary files from the output directory.       |
| **full_pipeline**        | All-in-one command that includes all of the above steps. |

### **General Command Format**
```bash
nanoTS <subcommand> [OPTIONS]
```

---

## **Models**
NanoTS uses pre-trained models for SNP calling:

| Model Name                                | Description                          |
|-------------------------------------------|--------------------------------------|
| `unphased_cDNA_nanopore_R104_HG002.pth`   | Unphased SNP model for cDNA |
| `unphased_dRNA_nanopore_R104_HG002.pth`   | Unphased SNP model for direct RNA   |
| `phased_cDNA_nanopore_R104_HG002.pth`     | Phased SNP model for cDNA  |
| `phased_dRNA_nanopore_R104_HG002.pth`     | Phased SNP model for direct RNA     |

---

## **Subcommands & Arguments**
### **1️⃣ bam**
Append a numeric suffix to each BAM alignment QNAME based on its count per read, ensuring each alignment has a unique QNAME (e.g., read_1, read_2).
#### **Required Arguments:**
| Argument   | Description                                        |
|------------|----------------------------------------------------|
| `-i/--input`    | Input sorted BAM file.                      |
| `-o/--output`    | Output BAM file with suffixed QNAMEs.                      |
#### **Optional Arguments:**
| Argument    | Default | Description                                               |
|-------------|---------|-----------------------------------------------------------|
| `--region`  | None    | Target region (`chr1` or `chr1:1000-2000`).              |

### **2️⃣ unphased_call**
Extracts candidate variants and features for **deep learning-based SNP calling**.

#### **Required Arguments:**
| Argument   | Description                                        |
|------------|----------------------------------------------------|
| `--bam`    | Sorted and indexed BAM file.                      |
| `--ref`    | Reference genome FASTA file.                      |
| `--outdir` | Output directory.                                 |
| `--model`  | Pre-trained model (`.pth` file).                 |

#### **Optional Arguments:**
| Argument    | Default | Description                                               |
|-------------|---------|-----------------------------------------------------------|
| `--threads` | 24      | Number of CPU threads.                                    |
| `--region`  | None    | Target region (`chr1` or `chr1:1000-2000`).              |
| `--ALT`     | 2       | Minimum ALT SNP coverage.                                |
| `--total`   | 2       | Minimum total base coverage.                             |
| `--ratio`   | 0.05    | Minimum ALT allele frequency.                            |
| `--depth`   | 1000    | Maximum reads per variant (`0` to disable limit).       |

🔹 **Note:** Increasing `--ALT` and `--total` thresholds may affect haplotype inference efficiency.

---

### **3️⃣ haplotype**
Performs **haplotype phasing** using an input **VCF, reference genome, and BAM file**.

#### **Required Arguments:**
| Argument   | Description                      |
|------------|----------------------------------|
| `--vcf`    | Input VCF file.                 |
| `--ref`    | Reference genome FASTA file.    |
| `--bam`    | Input BAM file.                 |
| `--outdir` | Output directory.               |
#### **Optional Arguments:**
| Argument    | Default | Description                                           |
|-------------|---------|-------------------------------------------------------|
| `--hap_qual`  | 10      | QUAL threshold to filter SNPs during haplotype phasing.       |

---

### **4️⃣ phased_call**
Processes **phased variant calls** using the deep-learning model.

#### **Required Arguments:**
| Argument   | Description                        |
|------------|------------------------------------|
| `--bam`    | Sorted and indexed BAM file.      |
| `--ref`    | Reference genome FASTA file.      |
| `--outdir` | Output directory.                 |
| `--model`  | Pre-trained model (`.pth` file). |

#### **Optional Arguments:**
| Argument    | Default | Description                                           |
|-------------|---------|-------------------------------------------------------|
| `--threads` | 24      | Number of CPU threads.                                |
| `--depth`   | 1000    | Maximum reads per variant (`0` to disable limit).   |

---

### **5️⃣ clean**
Removes **temporary files** in the output folder.

#### **Required Arguments:**
| Argument   | Description                  |
|------------|------------------------------|
| `--outdir` | Output directory to clean. |

---

### **6️⃣ full_pipeline**
Runs the **entire NanoTS pipeline** from a BAM file through final phased VCF output, then cleans up intermediate files.  

#### **Required Arguments:**
| Argument             | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `--bam`              | Input sorted & indexed BAM file.                              |
| `--ref`              | Reference genome FASTA file.                                 |
| `--model_unphased`   | Path to unphased model (`.pth` file).                         |
| `--model_phased`     | Path to phased model (`.pth` file).                           |
| `--outdir`           | Output directory for all steps.                               |

#### **Optional Arguments:**
| Argument      | Default | Description                                                   |
|---------------|---------|---------------------------------------------------------------|
| `--region`    | None    | Target region (e.g., `chr1` or `chr1:1000-2000`).             |
| `--threads`   | 24      | Number of CPU threads.                                        |
| `--ALT`       | 2       | Minimum ALT SNP coverage.                                     |
| `--total`     | 2       | Minimum total base coverage.                                  |
| `--ratio`     | 0.05    | Minimum ALT allele frequency.                                 |
| `--depth`     | 1000    | Maximum reads per variant (`0` to disable limit).             |
| `--hap_qual`  | 10      | QUAL threshold to filter SNPs during haplotype phasing.       |

---
## **Example Workflow**
Here’s an end-to-end example using **NanoTS**:

```bash
cd example/

# Download and prepare the reference genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gzip -d hg38.fa.gz
samtools faidx hg38.fa

# Align Nanopore reads to the reference genome
minimap2 -ax splice -ub -t 24 -k 14 -w 4 --secondary=no hg38.fa tutorial.fastq.gz | \
  samtools sort -o tutorial.bam

# Define output directory
outdir=nanoTS_result/
##################################
#### All-in-one step ####
nanoTS full_pipeline \
  --bam tutorial.bam \
  --ref hg38.fa \
  --model_unphased ../model/unphased_cDNA_nanopore_R104_HG002.pth \
  --model_phased   ../model/phased_cDNA_nanopore_R104_HG002.pth \
  --outdir $outdir \
  --threads 24 \
  --ALT 2 \
  --total 2 \
  --ratio 0.05 \
  --depth 1000 \
  --hap_qual 10
##################################
#### Separate steps ####
# Step 1: Suffix each BAM alignment QNAME by count per read
nanoTS bam \
  -i tutorial.bam \
  -o tutorial.qname.bam

# Step 2: Unphased SNP calling
nanoTS unphased_call \
  --bam tutorial.qname.bam \
  --ref hg38.fa \
  --threads 24 \
  --ALT 2 \
  --total 2 \
  --ratio 0.05 \
  --depth 1000 \
  --outdir $outdir \
  --model ../model/unphased_cDNA_nanopore_R104_HG002.pth

# Step 3: Haplotype phasing
nanoTS haplotype \
  --vcf ${outdir}unphased_predict.pass.vcf \
  --ref hg38.fa \
  --bam tutorial.qname.bam \
  --outdir $outdir \
  --QUAL 10

# Step 4: Phased SNP calling
nanoTS phased_call \
  --bam tutorial.qname.bam \
  --ref hg38.fa \
  --threads 24 \
  --depth 1000 \
  --outdir $outdir \
  --model ../model/phased_cDNA_nanopore_R104_HG002.pth

# Step 5: Clean temporary files
nanoTS clean \
  --outdir $outdir
```

Run NanoTS with Singularity (always bind host folders you read/write)
```
cd example/

SIF=../../../nanots_latest.sif # use your SIF path
EXAMPLE_DIR="$(pwd)"
MODEL_DIR="$(cd ../model && pwd)"   # the model folder in this repository. Adjust if your model files live elsewhere

OUTDIR="$EXAMPLE_DIR/nanoTS_result"
mkdir -p "$OUTDIR"

# Bind BOTH the data dir and the model dir so the container can see them.
singularity exec -B "$EXAMPLE_DIR","$MODEL_DIR" "$SIF" \
  /opt/conda/envs/nanoTS/bin/nanoTS full_pipeline \
    --bam "$EXAMPLE_DIR/tutorial.bam" \
    --ref "$EXAMPLE_DIR/hg38.fa" \
    --model_unphased "$MODEL_DIR/unphased_cDNA_nanopore_R104_HG002.pth" \
    --model_phased  "$MODEL_DIR/phased_cDNA_nanopore_R104_HG002.pth" \
    --outdir "$OUTDIR" \
    --threads 24 \
    --ALT 2 --total 2 --ratio 0.05 --depth 1000 --hap_qual 10
```

---

## **Output Files**
- **`unphased_predict.pass.vcf`** → SNP calls generated from the `unphased_call` step.
- **`phased_predict.pass.vcf`** → SNP calls generated from the `phased_call` step after haplotype phasing.

All results are stored in the specified **`--outdir`** directory.

---

## **License**
This project is licensed under the **GPL-3.0 License**. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
For questions, bug reports, or feature requests, contact:

📧 **Zelin Liu** – liuz6@chop.edu

