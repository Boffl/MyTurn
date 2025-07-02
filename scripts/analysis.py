from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Literal
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


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


def cross_eval_random_forest(dataset, pos: Literal["b", "m", "e"], n_splits: int = 10):
    X = dataset[pos][0]
    y = dataset[pos][1]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    reports = []
    cms = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(
            y_test, y_pred, target_names=["different", "same"], output_dict=True
        )
        reports.append(report)
        cms.append(confusion_matrix(y_test, y_pred))

    # Aggregate classification reports
    metrics = ["precision", "recall", "f1-score", "support"]
    classes = ["different", "same"]
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
    std_df = pd.DataFrame(std_results).T

    # Aggregate confusion matrices
    cms = np.array(cms)
    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)

    return mean_df, std_df, mean_cm, std_cm



