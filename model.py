import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split


def extract_features_and_labels(csv_path):
    df = pd.read_csv(csv_path)
    requests = df['number'].tolist()
    labels = df['is_cached'].tolist()

    freq_counter = defaultdict(int)
    features = []
    for i, key in enumerate(requests):
        freq_counter[key] += 1
        features.append([key, freq_counter[key]])

    return np.array(features), np.array(labels)


def train_and_save_best_model(data_csv, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    X, y = extract_features_and_labels(data_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "cat": CatBoostClassifier(verbose=0),
        "lgbm": LGBMClassifier()
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {name}: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    best_path = f"{models_dir}/best_model.pkl"
    joblib.dump(best_model, best_path)
    print(f"Saved best model ({best_model_name}) to {best_path}")


if __name__ == "__main__":
    DATA_CSV = "data/labeled_requests.csv"
    train_and_save_best_model(DATA_CSV)

