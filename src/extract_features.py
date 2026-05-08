#!/usr/bin/env python3
"""
extract_features.py - Extract audio features from all AVLombard .wav files

This script walks through data/raw/ folder, processes every .wav file,
and saves a master CSV with features and metadata.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# Suppress librosa warnings (optional)
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
def extract_audio_features(file_path):
    """
    Extract a comprehensive set of audio features from a single .wav file.
    Returns a dictionary of features.
    """
    try:
        # Load audio (resample to 16 kHz for consistency)
        y, sr = librosa.load(file_path, sr=16000)

        # 13 MFCCs (coefficients 1-13)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Pitch (fundamental frequency) using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)

        # Spectral bandwidth (spread)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)

        # Spectral rolloff (frequency below which 85% of energy)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)

        # Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Build feature dictionary
        features = {
            # MFCCs
            'mfcc1' : mfccs_mean[0], 'mfcc2' : mfccs_mean[1], 'mfcc3' : mfccs_mean[2],
            'mfcc4' : mfccs_mean[3], 'mfcc5' : mfccs_mean[4], 'mfcc6' : mfccs_mean[5],
            'mfcc7' : mfccs_mean[6], 'mfcc8' : mfccs_mean[7], 'mfcc9' : mfccs_mean[8],
            'mfcc10' : mfccs_mean[9], 'mfcc11' : mfccs_mean[10], 'mfcc12' : mfccs_mean[11],
            # Other prosodic/spectral features
            'pitch' : pitch_mean,
            'rms' : rms_mean,
            'zcr' : zcr_mean,
            'centroid' : centroid_mean,
            'bandwidth' : bandwidth_mean,
            'rolloff' : rolloff_mean,
        }
        # Add chroma features
        for i in range(12):
            features[f'chroma{i+1}'] = chroma_mean[i]

        return features
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return None

# ----------------------------------------------------------------------
def main():
    # Define data root (folder containing speaker subfolders)
    data_root = Path(__file__).parent.parent / "data" / "raw"
    if not data_root.exists():
        sys.exit(f'Data folder does not exist: {data_root.absolute()}')

    # Find all .wav files recursively
    wav_files = list(data_root.rglob('*.wav'))
    if not wav_files:
        sys.exit(f'No .wav files found in {data_root}')

    print(f'Found {len(wav_files)} .wav files')

    # List to hold all extracted records
    all_records = []

    # Process each file with progress bar
    for wav_path in tqdm(wav_files, desc='Extracting features'):
        # Extract speaker ID from folder name (e.g., data/raw/s2/... -> s2)
        speaker = wav_path.parent.name

        # Determine condition from filename
        # AVLombard naming: contains '_l_' for Lombard, '_p_' for Plain
        fname = wav_path.stem
        if '_l_' in fname:
            condition = 'Lombard'
        elif '_p_' in fname:
            condition = 'Plain'
        else:
            condition = 'Unknown'

        # Extract features
        feats = extract_audio_features(wav_path)
        if feats is not None:
            feats['speaker'] = speaker
            feats['condition'] = condition
            feats['filename'] = wav_path.name
            all_records.append(feats)

    # Convert to DataFrame and save
    if all_records:
        df = pd.DataFrame(all_records)
        # Reorder columns to put metadata first
        cols = ['speaker', 'condition', 'filename'] + [c for c in df.columns if c not in ['speaker', 'condition', 'filename']]
        df = df[cols]

        # Ensure results directory exists
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        output_csv = results_dir / 'audio_features_raw.csv'
        df.to_csv(output_csv, index=False)
        print(f'\nSaved {len(df)} records to {output_csv}')
        print(f'Columns: {list(df.columns)}')
    else:
        print('No features extracted. Check your .wav files')

if __name__ == '__main__':
    main()