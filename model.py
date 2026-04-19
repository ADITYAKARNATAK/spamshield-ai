"""Train, compare, and save the best SMS spam classification model."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import preprocess_text


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
VECTORIZER_PATH = ARTIFACT_DIR / "vectorizer.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


def load_dataset(path: str) -> tuple[pd.Series, pd.Series]:
    """Load the Kaggle SMS Spam Collection CSV."""
    data = pd.read_csv(path, encoding="latin-1")

    if {"v1", "v2"}.issubset(data.columns):
        data = data[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif {"label", "message"}.issubset(data.columns):
        data = data[["label", "message"]]
    else:
        raise ValueError("Dataset must contain columns v1/v2 or label/message.")

    data = data.dropna()
    data["label"] = data["label"].str.lower().map({"ham": 0, "spam": 1})
    data = data.dropna(subset=["label", "message"])
    return data["message"], data["label"].astype(int)


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    """Fit one model and return standard classification metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "name": name,
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def train(data_path: str) -> dict:
    messages, labels = load_dataset(data_path)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(x_train_raw)
    x_test = vectorizer.transform(x_test_raw)

    candidates = [
        ("Naive Bayes", MultinomialNB(alpha=0.35)),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_split=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = [
        evaluate_model(name, model, x_train, x_test, y_train, y_test)
        for name, model in candidates
    ]
    best = max(results, key=lambda item: (item["f1_score"], item["precision"], item["recall"]))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(best["model"], file)
    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)

    metadata = {
        "best_model": best["name"],
        "model_path": str(MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "metrics": [
            {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items() if key != "model"}
            for result in results
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SMS spam classifier.")
    parser.add_argument("--data", default="spam.csv", help="Path to Kaggle spam.csv file.")
    args = parser.parse_args()

    metadata = train(args.data)
    print("\nModel comparison")
    print("----------------")
    for row in metadata["metrics"]:
        print(
            f"{row['name']}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1_score']:.4f}"
        )
    print(f"\nBest model: {metadata['best_model']}")
    print(f"Saved: {MODEL_PATH} and {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
