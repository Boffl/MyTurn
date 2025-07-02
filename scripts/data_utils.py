import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import pandas as pd


def extract_and_store_mfccs(audio_folder: str, mfcc_folder: str, n_mfcc: int = 13, total:int=195):
    audio_folder = Path(audio_folder)
    mfcc_folder = Path(mfcc_folder)
    mfcc_folder.mkdir(parents=True, exist_ok=True)

    for wav_file in tqdm(audio_folder.glob("*.wav"), total=total):
        y, sr = librosa.load(wav_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Save MFCCs to .npy file with same base name
        output_file = mfcc_folder / (wav_file.stem + ".npy")
        np.save(output_file, mfcc)




def generate_stimuli_index(df: pd.DataFrame, pos_value: str) -> pd.DataFrame:
    # Filter rows by the given pos value
    filtered = df[df['pos'] == pos_value].reset_index(drop=True)
    
    # Generate all unique unordered pairs of row indices
    pairs = list(combinations(filtered.index, 2))
    
    # Construct rows for the new DataFrame
    data = []
    for i, j in pairs:
        row_i = filtered.loc[i]
        row_j = filtered.loc[j]
        condition = 'same' if row_i['speaker'] == row_j['speaker'] else 'different'
        data.append({
            'speaker1': row_i['speaker'],
            'turn1': row_i['turn'],
            'speaker2': row_j['speaker'],
            'turn2': row_j['turn'],
            'condition': condition
        })
        # Upsample the target class, by "restoring Order"
        if condition == "same":
                    data.append({
            'speaker1': row_j['speaker'],
            'turn1': row_j['turn'],
            'speaker2': row_i['speaker'],
            'turn2': row_i['turn'],
            'condition': condition
        })
    
    return pd.DataFrame(data)



def summarize_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """
    Summarize MFCC matrix with mean and std pooling across time (axis=1).
    Returns a 1D vector: [mean_1, ..., mean_n, std_1, ..., std_n]
    """
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std])  # shape: (2 * n_mfcc,)

def build_feature_dataset(stimuli_df: pd.DataFrame, pos_value: str, mfcc_folder: str) -> tuple[np.ndarray, np.ndarray]:
    mfcc_folder = Path(mfcc_folder)
    X = []
    y = []

    for _, row in tqdm(stimuli_df.iterrows(), total=len(stimuli_df)):
        # Build file paths
        file1 = mfcc_folder / f"{pos_value}_{row['speaker1']}_{row['turn1']}.npy"
        file2 = mfcc_folder / f"{pos_value}_{row['speaker2']}_{row['turn2']}.npy"

        # Load MFCCs
        mfcc1 = np.load(file1)
        mfcc2 = np.load(file2)

        # Pool to feature vectors
        vec1 = summarize_mfcc(mfcc1)
        vec2 = summarize_mfcc(mfcc2)

        # Combine both vectors
        combined = np.concatenate([vec1, vec2])  # shape: (4 * n_mfcc,)
        X.append(combined)
        y.append(1 if row['condition'] == 'same' else 0)

    return (np.stack(X), np.array(y))
