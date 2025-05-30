# LMNglyPred: Prediction of Human N-Linked Glycosylation Sites Using Embeddings from Pre-trained Protein Language Models

LMNglyPred is a deep learning-based tool for predicting N-linked glycosylation sites in human proteins using embeddings from pre-trained protein language models. It focuses on the biologically relevant N-X-[S/T] sequon, leveraging advanced machine learning to improve prediction accuracy and reliability.


## Table of Contents

- [Authors](#authors)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Evaluation](#model-evaluation)
  - [Prediction on Your Own Sequences](#prediction-on-your-own-sequences)
  - [Training and Experiments](#training-and-experiments)
- [Citation](#citation)
- [Contact](#contact)

---

## Authors

- Subash C Pakhrin1
- Suresh Pokharel2
- Kiyoko F Aoki-Kinoshita3
- Moriah R Beck4
- Tarun K Dam5
- Doina Caragea6
- Dukka B KC2*

1 School of Computing, Wichita State University, Wichita, KS, USA  
Department of Computer Science and Engineering Technology, University of Houston-Downtown, Houston, TX, USA  
2 Department of Computer Science, Michigan Technological University, Houghton, MI, USA  
3 Glycan and Life Systems Integration Center (GaLSIC), Soka University, Tokyo, Japan  
4 Department of Chemistry and Biochemistry, Wichita State University, Wichita, KS, USA  
5 Laboratory of Mechanistic Glycobiology, Department of Chemistry, Michigan Technological University, Houghton, MI, USA  
6 Department of Computer Science, Kansas State University, Manhattan, KS, USA  

*Corresponding Author: dbkc@mtu.edu


## Overview

Protein N-linked glycosylation is a crucial post-translational modification in humans, occurring at the N-X-[S/T] motif (where X ≠ proline). However, not all such sequons are glycosylated, making computational prediction vital for biological research. LMNglyPred utilizes deep learning and protein language model embeddings to achieve robust performance, with benchmark results demonstrating sensitivity, specificity, Matthews Correlation Coefficient, precision, and accuracy of 76.50%, 75.36%, 0.49, 60.99%, and 75.74%, respectively[8][9][10].

## Installation

**Python version:** 3.9.7

### Clone the repository:

```bash
gh repo clone partrita/LMNglyPred
```
### Install required libraries:

Use uv to dependency control.

```bash
uv sync
```

## Usage

### Prediction on Your Own Sequences

To predict N-linked glycosylation sites on your own sequences:

1. Place your FASTA sequences in `input/sequence.fa`.
2. Run:
   ```bash
   uv run cli.py -i input/sequence.fa -o output/result.csv
   ```
3. Results will appear in the `output/result.csv`

### Training and Experiments

- Data and scripts for training and additional experiments can be found in the `training_and_evaluation` folder.
- Follow the `readme.md` inside that folder for detailed instructions.


## Citation

If you use LMNglyPred in your research, please cite:

> Subash C Pakhrin, PhD, et al., "LMNglyPred: prediction of human N-linked glycosylation sites using embeddings from a pre-trained protein language model," Glycobiology, Volume 33, Issue 5, May 2023, Pages 411–422.  
> [https://doi.org/10.1093/glycob/cwad033](https://doi.org/10.1093/glycob/cwad033)


## Contact

For questions or support, contact:  
- Dr. Subash Chandra Pakhrin: pakhrins@uhd.edu  
- Dr. Dukka B. KC: dbkc@mtu.edu
