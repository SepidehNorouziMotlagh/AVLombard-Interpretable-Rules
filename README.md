# AVLombard: Interpretable Acoustic Rule Mining for Lombard Speech

## Overview

This project explores the use of Apriori association rule mining to identify interpretable acoustic feature patterns associated with Lombard speech (speech produced in noisy environments) in the AVLombard corpus.

The analysis uses acoustic features extracted from speech recordings and evaluates selected feature-bin associations on a speaker-disjoint held-out test set. The objective is to examine whether simple, human-readable associations can describe differences between Lombard and plain speech.

This is an exploratory association-rule analysis, not a trained or clinically validated speech classification system.

## Dataset

The project uses the [Audio-Visual Lombard Speech corpus](https://spandh.dcs.shef.ac.uk/avlombard/) by Alghamdi et al. (2018).

The original corpus contains:

- 54 speakers
- 5,400 utterances: 2,700 Lombard and 2,700 plain speech
- Audio recordings resampled to 16 kHz during feature extraction

The current feature-extraction run in this project reports 5,390 processed utterances. The difference from the corpus's published total has not been investigated.

The audio files are not included in this repository and must be downloaded separately.

## Acoustic Features

The feature extraction script computes 30 utterance-level acoustic features:

- 12 MFCC-derived features, calculated as the mean across frames
- Mean pitch estimate using Librosa's `piptrack`
- Root mean square (RMS) energy
- Zero-crossing rate (ZCR)
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- 12 chroma features

**MFCC naming note:** The feature currently named `mfcc1` corresponds to Librosa's MFCC coefficient at index 0 (`mfccs_mean[0]`), often referred to as MFCC0. The repository retains the original feature names for consistency with its existing results.

## Method

The current implementation follows a single speaker-disjoint train/test evaluation:

1. **Feature extraction:** Audio recordings are loaded and resampled to 16 kHz. The extracted feature values and speaker/condition metadata are stored in a CSV file.

2. **Speaker split:** Speaker IDs are sorted, with the first 40 speakers assigned to training and the remaining 14 assigned to the held-out test set. No speaker appears in both sets.

3. **Feature discretization:** Each numerical feature is divided into low, medium, and high bins using the 33rd and 67th percentiles calculated from the training set. The same thresholds are applied to the test set.

4. **Transaction construction:** Each utterance is represented as a transaction containing its discretized acoustic feature values and its speech-condition label.

5. **Association rule mining:** Apriori is applied to the training transactions using a minimum support of 0.15. Association rules are generated with a minimum confidence of 0.70, and rules whose consequent contains the `Lombard` label are selected.

6. **Held-out evaluation:** The 20 selected training rules with the highest training lift are taken forward. For each selected rule's antecedent, the script calculates its association with the `Lombard` label in the held-out test set.

**Important evaluation detail:** The current implementation may mine rules whose consequents contain both `Lombard` and additional acoustic feature items. However, the held-out evaluation calculates the association between the antecedent and `Lombard` alone. Consequently, the stored training confidence/lift and held-out confidence/lift do not necessarily describe the same complete rule. They should not be interpreted as a direct train/test comparison.

This is a single fixed speaker-disjoint evaluation, not cross-validation.

## Held-Out Test Results

The following table reports the five evaluated antecedents with the highest held-out test lift.

| Antecedent | Test confidence | Test lift | Test support | Antecedent count |
|---|---:|---:|---:|---:|
| `{mfcc1_high}` | 91.4% | 1.815 | 38.35% | 583 |
| `{mfcc3_low, mfcc1_high}` | 90.6% | 1.799 | 24.89% | 382 |
| `{mfcc3_low, rms_high, mfcc1_high}` | 90.1% | 1.790 | 20.36% | 314 |
| `{mfcc3_low, rms_high}` | 90.0% | 1.787 | 20.72% | 320 |
| `{rms_high}` | 89.2% | 1.771 | 33.17% | 517 |

For the first antecedent, 533 of the 583 held-out utterances containing `mfcc1_high` were labelled Lombard, giving a test confidence of 91.4%.

### Interpreting the metrics

- **Confidence:** The proportion of held-out utterances containing the antecedent that are labelled Lombard.
- **Lift:** The ratio between the conditional prevalence of Lombard speech given the antecedent and the overall prevalence of Lombard speech in the held-out set.
- **Support:** The proportion of all held-out utterances that contain both the antecedent and the Lombard label.
- **Antecedent count:** The number of held-out utterances containing the specified feature combination.

A lift of 1.815 means that the observed conditional prevalence of Lombard speech is 1.815 times its overall prevalence in the held-out set. These association metrics are not classification accuracy.

The prominence of `mfcc1_high` identifies an association in this experiment. It does not, by itself, establish that this feature measures loudness or spectral tilt.

## Repository Structure

```text
AVLombard-Interpretable-Rules/
├── src/
│   ├── extract_features.py
│   └── discretize_and_mine.py
├── results/
│   ├── audio_features_raw.csv
│   ├── heldout_speaker_rule_evaluation.csv
│   ├── lombard_rules_training.csv
│   └── train_test_confidence_comparison.png
├── data/
│   └── raw/                  # Downloaded corpus; not included
├── requirements.txt
├── LICENSE
└── README.md
```

The evaluation CSV is currently stored under the legacy filename `cross_validated_rules.csv`; rename it as described below. The comparison plot is also generated by the current implementation, but its training and test confidence values are not necessarily like-for-like because of the consequent mismatch described above.

## Reproducing the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/SepidehNorouziMotlagh/AVLombard-Interpretable-Rules.git
cd AVLombard-Interpretable-Rules
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the audio recordings for speakers 2–55 from the [official AVLombard website](https://spandh.dcs.shef.ac.uk/avlombard/).

Place the extracted speaker folders inside:

```text
data/raw/
```

### 4. Extract acoustic features

Run from the repository root:

```bash
python src/extract_features.py
```

This generates:

```text
results/audio_features_raw.csv
```

### 5. Mine and evaluate association rules

Run:

```bash
python src/discretize_and_mine.py
```

The script generates the training association-rule CSV, the held-out evaluation CSV, and a confidence comparison plot.

## Limitations

- The analysis uses one fixed speaker-disjoint train/test split of 40 training speakers and 14 held-out speakers.
- Speaker assignment is based on sorted speaker IDs rather than a randomized or repeated speaker-level evaluation.
- Only the 20 selected training rules with the highest lift are evaluated.
- The current implementation does not always evaluate the exact same complete rule on the training and test sets because mined consequents may contain additional acoustic items.
- Discretizing continuous features into three bins loses information and makes results dependent on the training-set quantile thresholds.
- The interpretation of the most prominent MFCC-derived feature requires further acoustic analysis.
- The analysis uses one English-language speech corpus.
- No clinical validation or deployment evaluation has been performed.

Future work includes correcting the rule-selection/evaluation consistency, using a reproducible speaker-level evaluation protocol, and investigating the acoustic characteristics associated with the most prominent feature patterns.

## Citation

If you use the AVLombard corpus, please cite:

Alghamdi, N., Maddock, S., Marxer, R., Barker, J., & Brown, G. J. (2018). A corpus of audio-visual Lombard speech with frontal and profile views. *The Journal of the Acoustical Society of America, 143*(6), EL523–EL529.

https://doi.org/10.1121/1.5042758

The AVLombard corpus is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please refer to the [official corpus website](https://spandh.dcs.shef.ac.uk/avlombard/) for its terms.

## License

The project code is intended to be released under the MIT License. See the `LICENSE` file.

## Author

**Sepideh Norouzi Motlagh**

[GitHub](https://github.com/SepidehNorouziMotlagh)
