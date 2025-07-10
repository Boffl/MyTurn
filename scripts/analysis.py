from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Literal
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import argparse, pickle, json


RANDOM_STATE = 42

def train_random_forest(dataset, pos: Literal["b", "m", "e"]) -> RandomForestClassifier:

    X = dataset[pos][0]
    y = dataset[pos][1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Initialize and train the classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)

    print(f"\n=== Evaluation for {pos} ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["different", "same"]))

    return clf


def cross_eval_random_forest(noise: str, pos: Literal["b", "m", "e"], n_splits: int = 5):
    """
    noise: needs to be of form 'noise-10'
    """

    if noise:
        dataset_filename = f"feature_dataset_{noise}.pickle"

    else:
        dataset_filename = "feature_dataset.pickle"

    try:
        with open(dataset_filename, "rb") as handle:
            dataset = pickle.load(handle)
    except FileNotFoundError:
        print(f"The dataset {dataset_filename} does not exist (yet).")

    X = dataset[pos][0]
    y = dataset[pos][1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    reports = []
    target_results = {"fold": [], "precision": [], "recall": [], "f1-score": [], "support": [], "total": []}
    fold = 1  # starting the fold count at 1

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(
            y_test, y_pred, target_names=["different", "same"], output_dict=True
        )
        report["fold"] = fold
        reports.append(report)

        target_results["fold"].append(fold)
        target_results["total"].append(report["macro avg"]["support"])
        for metric in report["same"]:
            target_results[metric].append(report["same"][metric])
        
        fold += 1
    
    target_results_df = pd.DataFrame(target_results)
    target_results_df.to_csv(f"results/onlyTarget_folds-{n_splits}_{noise}_pos-{pos}.csv")

    # save json
    # with open(f"results/allClasses_folds-{n_splits}_{noise}_pos-{pos}.json", "w", encoding="utf-8") as outjson:
    #    json.dump(reports, outjson)


    # Aggregate classification reports
    metrics = ["precision", "recall", "f1-score", "support"]
    classes = ["different", "same", "macro avg", "weighted avg"]
    mean_results = {}
    std_results = {}

    for cls in classes:
        mean_results[cls] = {}
        std_results[cls] = {}
        for metric in metrics:
            values = [rep[cls][metric] for rep in reports]
            mean_results[cls][metric] = np.mean(values)
            std_results[cls][metric] = np.std(values)

    # Convert to DataFrame for easier viewing
    mean_df = pd.DataFrame(mean_results).T
    mean_df.to_csv(f"results/allClasses_folds-{n_splits}_{noise}_pos-{pos}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", default=0,  help="Take a dataset with noise?")
    parser.add_argument("--folds", default=5, help="Number of folds for crosseval")

    args = parser.parse_args()

    if args.noise:
        noise = f"noise-{args.noise}"
    else: noise = False

    pos_list = ["b", "m", "e"]
    for pos in pos_list:
        print(f"Calculating Crosseval for {pos}...")
        cross_eval_random_forest(noise, pos, args.folds)
    print("done.")
