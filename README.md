# LMNglyPred

**LMNglyPred: Prediction of Human N-Linked Glycosylation Sites using embeddings from a pre-trained protein language model**

LMNglyPred is a research code and data repository for predicting human N-linked glycosylation sites (N-glycosylation) using protein-language-model embeddings, including ProtT5-derived 1,024-dimensional residue features.

> **Repository status:** This repository contains the original research notebooks, trained Keras models, feature files, datasets, and a ProtT5 feature-extraction script. It is primarily intended to reproduce the reported experiments rather than to provide a packaged command-line application.

## Repository layout

```text
LMNglyPred/
├── data/                  # Training, test, and ProtT5 feature data
├── models/                # Trained Keras models (.h5)
├── notebooks/             # Experiment and feature-extraction notebooks
├── scripts/               # ProtT5 feature extraction scripts
├── LICENSE
└── README.md
```

The filenames of the original datasets and models are preserved so that they remain traceable to the published/reported experiments.

## Requirements

The original experiments were developed with the following environment:

- Python 3.8.3
- pandas 1.0.5
- NumPy 1.18.5
- SciPy 1.4.1
- scikit-learn 0.23.1
- Keras 2.4.3
- TensorFlow 2.3.1
- Linux x86_64
- Anaconda 2020.07

The notebooks also record a Python 3.7.4 / TensorFlow 2.3.1 Jupyter kernel. Because these are historical dependencies, newer Python/TensorFlow versions may require compatibility adjustments.

## Quick start: reproduce the reported experiments

Clone the repository and create an environment close to the original one:

```bash
git clone https://github.com/partrita/LMNglyPred.git
cd LMNglyPred
conda create -n lmnglypred python=3.8.3
conda activate lmnglypred
pip install pandas==1.0.5 numpy==1.18.5 scipy==1.4.1 scikit-learn==0.23.1 tensorflow==2.3.1 keras==2.4.3 jupyter
```

Then launch Jupyter from the repository root:

```bash
jupyter notebook
```

### 1. NGlyDE original experiment

Open:

```text
notebooks/GlycoBiology_NGlyDE_Original.ipynb
```

This notebook evaluates the trained NGlyDE model against the independent ProtT5 feature set and also inspects the separation between training and independent-test protein IDs.

The required files are:

```text
data/Independent_Test_Set_Prot_T5_feature_Aug_12.txt
data/Subash_August_8_2022_NGlyDE_Prot_T5_feature.txt
models/Undersampling_Glycobiology_NGLYDE_Final6947757.h5
```

### 2. 90% training / 10% independent testing experiment

Open:

```text
notebooks/GlycoBiology_NGlyDE_90__Training_10__Indepedent_Testing.ipynb
```

Required files:

```text
data/Glycobiology_NGlyDE_Independent_Positive_202_Negative_100.csv
data/Glycobiology_NGlyDE_Training_Positive_1821_Negative_901.csv
models/NGlyDE_Prot_T5_Final.h5
```

The notebook reports MCC, confusion matrix, accuracy, sensitivity, specificity, precision, and the classification report.

### 3. NGlycositeAtlas experiment

Open:

```text
notebooks/GlycoBiology_NGlycositeAtlas.ipynb
```

Required files:

```text
data/df_indepenent_test_again_done_that_has_unique_protein_and_unique_sequence.csv
data/df_train_data_without_indepenent_test_and_protein.csv
models/Final_GlycoBiology_ANN_Glycobiology_ER_RSA(GA_Extracell_cellmem)187.h5
```

This notebook evaluates the model on the NGlycositeAtlas-derived independent test set and checks for redundant proteins/sequences between training and independent test sets.

## ProtT5 feature extraction

The repository includes a script for extracting residue-level ProtT5 embeddings from FASTA files:

```text
scripts/analyze_Cell_Mem_ER_Extrac_Protein.py
```

It uses:

```text
Rostlab/prot_t5_xl_uniref50
```

and expects the input FASTA path in the `FILENAME` environment variable. For example:

```bash
export FILENAME=/path/to/proteins.fasta
python scripts/analyze_Cell_Mem_ER_Extrac_Protein.py
```

The script writes one CSV file per FASTA record, containing the amino acid and its 1,024-dimensional ProtT5 representation.

For the original HPC/SLURM workflow, see:

```text
scripts/analyze_Cell_Mem_ER_Extrac_Protein.sh
```

The SLURM script contains site-specific module, virtual-environment, and filesystem paths from the original research environment. **Do not run it unchanged on another cluster.** Update the TensorFlow module, virtual environment, input directory, and script location first.

### Feature-vector extraction notebook

The original post-processing notebook is:

```text
notebooks/Feature_Extraction_Program_from_the_generated_files.ipynb
```

It converts per-protein ProtT5 CSV files into the feature matrix used by the downstream models. The notebook contains historical absolute paths and expects the original training metadata file and generated per-protein CSV files; update those paths for your environment before execution.

## Important path note

The original notebooks were written for the authors' filesystem and contain hard-coded working-directory paths in places. The repository has been reorganized for clarity, but the scientific workflow and original filenames are intentionally preserved.

If a notebook raises `FileNotFoundError`, check the `basedir` / `os.chdir(...)` cells first and point them to the repository's `data/` and `models/` directories as appropriate.

For example, from a notebook running in `notebooks/`, repository data can be addressed as:

```python
from pathlib import Path

ROOT = Path.cwd().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
```

Then use paths such as `DATA / "Independent_Test_Set_Prot_T5_feature_Aug_12.txt"` and `MODELS / "NGlyDE_Prot_T5_Final.h5"`.

## Reproducibility notes

- The `.h5` files are trained model artifacts; no model training pipeline is provided as a standalone script.
- The large `.txt` files under `data/` are precomputed feature matrices and can require substantial disk space and RAM.
- Reported metrics shown in notebook outputs are historical results stored in the notebooks; re-running the notebooks may produce different results if the software stack or numerical libraries differ.
- The repository does not currently expose a single `predict` command. For a new protein sequence, first generate ProtT5 residue embeddings and then adapt the feature-preparation/inference code in the relevant notebook to the desired model.

## Contact

For questions about the original research, the previous README listed:

- Dr. Subash Chandra Pakhrin — `pakhrins@uhd.edu`
- Dr. Dukka B. KC — `dbkc@mtu.edu`

## License

See [LICENSE](LICENSE).
