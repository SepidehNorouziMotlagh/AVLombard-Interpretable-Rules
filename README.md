# AVLombard: Interpretable Acoustic Rule Discovery for Lombard Speech

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project applies **association rule mining (Apriori)** to discover interpretable acoustic patterns that distinguish Lombard speech (spoken in noise) from plain speech. Using the **AVLombard corpus** (54 speakers, 5390 utterances), we perform a **speaker‑wise cross‑validation** to ensure generalisability.

The goal is not state‑of‑the‑art accuracy, but **transparency** – providing simple, human‑readable rules that could be useful in clinical or assistive contexts.

## Key Findings

- **Best generalisable rule:**  
  `{mfcc1_high} → Lombard`  
  - Test confidence: **91.4%**  
  - Test lift: **1.815** (Lombard speech is 81.5% more likely when MFCC1 is high)  

- MFCC1 (spectral tilt / energy) is the single most informative feature – consistent with the known increase in loudness during Lombard speech.

- A total of **58 rules** were generated on the training set; the top rules consistently involve high‑energy or high‑frequency features.

## Repository Structure
.
├── src/
│ ├── extract_features.py # Converts .wav files to acoustic features
│ └── discretize_and_mine.py # Discretises features, runs Apriori, cross‑validation
├── results/
│ ├── cross_validated_rules.csv # Top rules with train/test metrics
│ ├── lombard_rules_training.csv # All rules from training set
│ └── train_test_confidence_comparison.png
├── requirements.txt
├── .gitignore
└── README.md

text

## Dataset

We use the **AVLombard corpus** (Alghamdi et al., JASA 2018):
- 54 speakers, 100 utterances each (50 Plain, 50 Lombard)
- Audio sampled at 16 kHz
- 30 acoustic features extracted (MFCCs, pitch, zero‑crossing rate, RMS energy, spectral centroid, rolloff, chroma)

## How to Reproduce

1. **Clone the repository**  
   `git clone https://github.com/yourusername/AVLombard-Interpretable-Rules.git`

2. **Install dependencies**  
   `pip install -r requirements.txt`

3. **Download the AVLombard dataset** from the [official site](https://spandh.dcs.shef.ac.uk/avlombard/) (Audio files for all 54 speakers).  
   Place the unzipped speaker folders (s2 to s55) into a folder named `data/raw/`.

4. **Run feature extraction**  
   `python src/extract_features.py`  
   → Creates `results/audio_features_raw.csv` (approx. 100 MB)

5. **Run cross‑validated rule mining**  
   `python src/discretize_and_mine.py`  
   → Generates rule CSV files and the comparison plot.

## Results Summary (Top 5 Rules by Test Lift)

| Antecedents (features) | Train Conf. | Test Conf. | Test Lift |
|------------------------|-------------|------------|-----------|
| `{mfcc1_high}` | 76.8% | **91.4%** | **1.815** |
| `{mfcc3_low, mfcc1_high}` | 81.6% | 90.6% | 1.799 |
| `{mfcc3_low, rms_high, mfcc1_high}` | 95.9% | 90.1% | 1.790 |
| `{rms_high}` | 76.8% | 89.2% | 1.771 |
| `{rolloff_high, zcr_high, mfcc2_low}` | 80.4% | 85.6% | 1.700 |

> *All metrics are computed on a held‑out set of 14 speakers (not seen during rule generation).*

## Why Apriori?

In clinical or safety‑critical applications, **interpretability** can be as important as raw accuracy. Association rules produce transparent, logical statements (`if X then Lombard`) that clinicians can understand and trust – unlike black‑box models.

## Limitations

- The dataset contains only English speakers.
- Features were discretised into three bins (low/medium/high), losing some nuance.
- Rules were evaluated on a single fixed train/test split (40/14 speakers); future work could use leave‑one‑speaker‑out cross‑validation.

## Citation

If you use this code or the AVLombard dataset, please cite:

Alghamdi, N., et al. (2018). *AVLombard: A multi‑speaker Lombard speech corpus.* The Journal of the Acoustical Society of America, 144(3), 1840‑1840.

## License

MIT

## Author

Sepideh Norouzi Motlagh  
github.com/SepidehNorouziMotlagh
