import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
import pandas as pd
import argparse
import os, pickle


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


def create_pos_df_and_summary(noise):
    """Helper function to create DFs of the results"""
    summary = {"pos":[]}
    pos_dfs = {}
    for pos in ["b", "m", "e"]:
        df = pd.read_csv(f"results/onlyTarget_folds-5_{noise}_pos-{pos}.csv")
        pos_dfs[pos] = df
        summary["pos"].append(pos)
        for column in df.columns:
            if column in ["Unnamed: 0", 'fold']:
                continue
            if column in summary:
                summary[column].append(df[column].mean())
            else:
                summary[column] = [df[column].mean()]
    summary_df = pd.DataFrame(summary)
    return pos_dfs, summary_df


def extract_and_store_mfccs_with_noise(
    audio_folder: str,
    mfcc_folder: str,
    n_mfcc: int = 13,
    total: int = 195,
    noise_std: float = 0.005  # Standard deviation of added Gaussian noise
):
    audio_folder = Path(audio_folder)
    mfcc_folder = Path(mfcc_folder)
    mfcc_folder.mkdir(parents=True, exist_ok=True)

    for wav_file in tqdm(audio_folder.glob("*.wav"), total=total):

        y, sr = librosa.load(wav_file, sr=None)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, y.shape)
        y_noisy = y + noise


        mfcc = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=n_mfcc)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audioDir", help="Path to the directory with the wav files")
    parser.add_argument("--noise", default=0, help="Standard deviation of added Gaussian noise, defaults to no noise")
    args = parser.parse_args()

     
     # make an index of all the data we have
    files = os.listdir(args.audioDir)
    index_df = pd.DataFrame(columns=["pos", "speaker", "turn"])
    for file in files:
        name = file.split(".")[0]
        pos, speaker, turn = name.split("_")
        index_df.loc[len(index_df)] = [pos, speaker, turn]



    # Downsample to have for each position the same amount of data:
    min_count = index_df['pos'].value_counts().min()
    index_df = (
        index_df.groupby('pos', group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .sort_index()
    )

    # Inspect the Data
    speakers = set(list(index_df["speaker"]))
    print("Number of Sound Snipptes: ", len(index_df["pos"]))
    print("Number of speakers: ", len(speakers))
    print("Speaker IDs: ", speakers)

    # extract the MFCCs
    print("Extracting MFCCs")
    if args.noise == 0:
        MFCC_dir = "MFCCs"
        dataset_filename = 'feature_dataset.pickle'
        extract_and_store_mfccs(args.audioDir, MFCC_dir, total=len(files))
    else:
        MFCC_dir = f'MFCCs_noise_{str(args.noise).replace(".", "_")}'
        dataset_filename = f'feature_dataset_noise-{str(args.noise).replace(".", "_")}.pickle'
        extract_and_store_mfccs_with_noise(args.audioDir, MFCC_dir, total=len(files), noise_std=float(args.noise))

    # Index the Stimuli
    beginning_stimuli_index = generate_stimuli_index(index_df, "b")
    middle_stimuli_index = generate_stimuli_index(index_df, "m")
    end_stimuli_index = generate_stimuli_index(index_df, "e")

    # build and save the dataset
    print("Building Feature Dataset")
    dataset_b = build_feature_dataset(beginning_stimuli_index, "b", MFCC_dir)
    dataset_m = build_feature_dataset(middle_stimuli_index, "m", MFCC_dir)
    dataset_e = build_feature_dataset(end_stimuli_index, "e", MFCC_dir)

    # Full dataset, for easy access:
    dataset = {"b": dataset_b, "m": dataset_m, "e": dataset_e}

    # store data
    with open(dataset_filename, "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
