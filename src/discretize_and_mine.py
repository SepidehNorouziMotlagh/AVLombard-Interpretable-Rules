#!/usr/bin/env python3
"""
discretize_and_mine.py - Full dataset association rule mining with speaker-wise cross-validation

Steps:
1. Load full audio_features_raw.csv (using project‑root path)
2. Split speakers into training and test sets (e.g., 40 train, 14 test)
3. Discretize numeric features into low/medium/high using training set quantiles
4. Apply same discretization to test set
5. Run Apriori on training set to generate rules
6. Evaluate the rules on test set (confidence, lift, support)
7. Save results and plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Path setup (work from project root, not script location)
# ----------------------------------------------------------------------
# Get the directory where THIS script is located
script_dir = Path(__file__).parent  # .../audio_mining_project/src
project_root = script_dir.parent  # .../audio_mining_project

# Input and output directories
data_csv = project_root / "results" / "audio_features_raw.csv"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_SPEAKERS = 40  # number of speakers for training
TEST_SPEAKERS = None  # automatically use the rest
MIN_SUPPORT = 0.15
MIN_CONFIDENCE = 0.7

# ----------------------------------------------------------------------
# 1. Load full dataset
# ----------------------------------------------------------------------
if not data_csv.exists():
    raise FileNotFoundError(f"Could not find {data_csv}. Run extract_features.py first.")

df = pd.read_csv(data_csv)
print(f"Loaded {len(df)} samples from {df['speaker'].nunique()} speakers")
df = df[df['condition'] != 'Unknown']
print(f"After removing 'Unknown': {len(df)} samples")
print(f"Speakers: {sorted(df['speaker'].unique())}")
print(f"Condition counts:\n{df['condition'].value_counts()}")

# ----------------------------------------------------------------------
# 2. Speaker-wise train/test split
# ----------------------------------------------------------------------
speakers = sorted(df['speaker'].unique())
if TEST_SPEAKERS is None:
    TEST_SPEAKERS = len(speakers) - TRAIN_SPEAKERS

# Split speaker list
train_speakers = speakers[:TRAIN_SPEAKERS]
test_speakers = speakers[TRAIN_SPEAKERS:]

print(f"\nTraining speakers ({len(train_speakers)}): {train_speakers}")
print(f"Test speakers ({len(test_speakers)}): {test_speakers}")

train_df = df[df['speaker'].isin(train_speakers)].copy()
test_df = df[df['speaker'].isin(test_speakers)].copy()

print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
print(f"Train condition counts:\n{train_df['condition'].value_counts()}")
print(f"Test condition counts:\n{test_df['condition'].value_counts()}")

# ----------------------------------------------------------------------
# 3. Define which columns are features (exclude metadata)
# ----------------------------------------------------------------------
metadata_cols = ['speaker', 'condition', 'filename']
feature_cols = [c for c in df.columns if c not in metadata_cols]

print(f"\nUsing {len(feature_cols)} feature columns: {feature_cols[:5]}...")


# ----------------------------------------------------------------------
# 4. Discretize using training set quantiles (low/medium/high)
# ----------------------------------------------------------------------
def discretize_by_quantiles(df_train, df_test, feature_cols):
    """
    Discretize features using quantiles from training set.
    Returns: df_train_binned, df_test_binned (both with categorical columns)
    """
    df_train_binned = df_train.copy()
    df_test_binned = df_test.copy()

    for col in feature_cols:
        # Get quantiles from training set only
        low = df_train[col].quantile(0.33)
        high = df_train[col].quantile(0.67)

        # Function to bin values
        def bin_value(x):
            if pd.isna(x):
                return f"{col}_unknown"
            if x <= low:
                return f"{col}_low"
            elif x <= high:
                return f"{col}_med"
            else:
                return f"{col}_high"

        # Apply to train and test
        df_train_binned[f"{col}_cat"] = df_train[col].apply(bin_value)
        df_test_binned[f"{col}_cat"] = df_test[col].apply(bin_value)

    return df_train_binned, df_test_binned


train_binned, test_binned = discretize_by_quantiles(train_df, test_df, feature_cols)

# Get categorical feature columns (those ending with _cat)
cat_cols = [c for c in train_binned.columns if c.endswith('_cat')]
print(f"Created {len(cat_cols)} categorical features.")


# ----------------------------------------------------------------------
# 5. Convert to transaction list format (for Apriori)
# ----------------------------------------------------------------------
def df_to_transactions(df, cat_cols, target_col='condition'):
    transactions = []
    for idx, row in df.iterrows():
        items = [row[col] for col in cat_cols]
        items.append(row[target_col])
        transactions.append(items)
    return transactions


train_transactions = df_to_transactions(train_binned, cat_cols)
test_transactions = df_to_transactions(test_binned, cat_cols)

print(f"\nTraining transactions: {len(train_transactions)}")
print(f"Test transactions: {len(test_transactions)}")
print(f"First training transaction (first few items): {train_transactions[0][:5]}...")

# ----------------------------------------------------------------------
# 6. Run Apriori on training set only
# ----------------------------------------------------------------------
te = TransactionEncoder()
te_ary = te.fit(train_transactions).transform(train_transactions)
df_train_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Training encoded shape: {df_train_encoded.shape}")

# Mine frequent itemsets
frequent_itemsets = apriori(df_train_encoded, min_support=MIN_SUPPORT, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets on training set.")

# Generate rules
train_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print(f"Generated {len(train_rules)} association rules on training set.")

# Filter rules where 'Lombard' is in the consequent
lombard_rules = train_rules[train_rules['consequents'].apply(lambda x: 'Lombard' in x)]
lombard_rules = lombard_rules.sort_values('lift', ascending=False)
print(f"Rules with Lombard in consequent: {len(lombard_rules)}")


# ----------------------------------------------------------------------
# 7. Evaluate top rules on test set
# ----------------------------------------------------------------------
def evaluate_rule_on_test(antecedent_items, consequent_item, test_transactions):
    """
    Calculate support, confidence, and lift for a single rule on test transactions.
    antecedent_items: frozenset of items (e.g., {'mfcc1_high', 'centroid_high'})
    consequent_item: string (e.g., 'Lombard')
    """
    total = len(test_transactions)
    if total == 0:
        return None

    # Count transactions containing antecedent
    ante_count = 0
    both_count = 0
    for trans in test_transactions:
        has_ante = all(item in trans for item in antecedent_items)
        has_conseq = consequent_item in trans
        if has_ante:
            ante_count += 1
            if has_conseq:
                both_count += 1

    support = both_count / total
    confidence = both_count / ante_count if ante_count > 0 else 0
    # Lift = P(conseq|ante) / P(conseq)
    p_conseq = sum(1 for t in test_transactions if consequent_item in t) / total
    lift = confidence / p_conseq if p_conseq > 0 else 0
    return {
        'support_test': support,
        'confidence_test': confidence,
        'lift_test': lift,
        'antecedent_count': ante_count,
        'both_count': both_count
    }


# Evaluate top 20 rules from training on test set
test_results = []
for idx, rule in lombard_rules.head(20).iterrows():
    ante = rule['antecedents']
    conseq = 'Lombard'  # we filtered for Lombard in consequent
    eval_metrics = evaluate_rule_on_test(ante, conseq, test_transactions)
    if eval_metrics:
        test_results.append({
            'antecedents': str(ante),
            'confidence_train': rule['confidence'],
            'lift_train': rule['lift'],
            'support_train': rule['support'],
            'support_test': eval_metrics['support_test'],
            'confidence_test': eval_metrics['confidence_test'],
            'lift_test': eval_metrics['lift_test'],
            'antecedent_count_test': eval_metrics['antecedent_count'],
            'both_count_test': eval_metrics['both_count']
        })

test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(results_dir / "cross_validated_rules.csv", index=False)
print(f"\nSaved cross-validated rules to {results_dir / 'cross_validated_rules.csv'}")
print("\nTop 5 rules with test confidence:")
print(test_results_df[['antecedents', 'confidence_train', 'confidence_test', 'lift_test']].head(5))

# ----------------------------------------------------------------------
# 8. Plot comparison: training confidence vs test confidence
# ----------------------------------------------------------------------
if len(test_results_df) > 0:
    plt.figure(figsize=(10, 6))
    x = range(len(test_results_df))
    plt.bar(x, test_results_df['confidence_train'], width=0.4, label='Training', color='skyblue', align='center')
    plt.bar(x, test_results_df['confidence_test'], width=0.4, label='Test', color='salmon', align='edge')
    plt.xlabel('Rule index')
    plt.ylabel('Confidence')
    plt.title('Training vs Test Confidence (Top 20 Rules)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'train_test_confidence_comparison.png', dpi=150)
    plt.close()
    print("Saved comparison plot: results/train_test_confidence_comparison.png")
else:
    print("No test results to plot.")

# ----------------------------------------------------------------------
# 9. Save top rules (training) for reference
# ----------------------------------------------------------------------
lombard_rules.to_csv(results_dir / "lombard_rules_training.csv", index=False)
print("\nFull training rules saved to results/lombard_rules_training.csv")