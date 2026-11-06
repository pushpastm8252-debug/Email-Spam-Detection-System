
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "cleaned_spam_data.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_spam_model.joblib"

METRICS_PATH = MODEL_DIR / "model_metrics.json"

CONFUSION_PATH = MODEL_DIR / "confusion_matrix.png"


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

def load_data():

    print("\nLoading cleaned dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "cleaned_spam_data.csv not found. "
            "Run python preprocess.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "cleaned_message",
        "label_encoded"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    df = df.dropna(
        subset=required_columns
    ).copy()

    X = df["cleaned_message"].astype(str)

    y = df["label_encoded"].astype(int)

    print("Dataset loaded successfully!")
    print("Total messages:", len(df))

    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


# ==========================================
# 3. TRAIN AND TEST SPLIT
# ==========================================

def split_data(X, y):

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test


# ==========================================
# 4. CREATE MACHINE LEARNING MODELS
# ==========================================

def create_models():

    models = {

        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "SVM": SVC(
            kernel="linear",
            probability=True,
            random_state=42
        )
    }

    return models


# ==========================================
# 5. CREATE NLP + TF-IDF PIPELINE
# ==========================================

def create_pipeline(classifier):

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            classifier
        )

    ])

    return pipeline


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

def train_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = {}

    best_f1 = -1

    best_model = None

    best_model_name = None

    for name, classifier in models.items():

        print("\n" + "=" * 45)
        print("Training:", name)
        print("=" * 45)

        pipeline = create_pipeline(classifier)

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print("Accuracy :", round(accuracy * 100, 2), "%")
        print("Precision:", round(precision * 100, 2), "%")
        print("Recall   :", round(recall * 100, 2), "%")
        print("F1 Score :", round(f1 * 100, 2), "%")

        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        }

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Ham", "Spam"],
                zero_division=0
            )
        )

        # Select model using spam-class F1 score
        if f1 > best_f1:

            best_f1 = f1

            best_model = pipeline

            best_model_name = name

            best_predictions = predictions

    return (
        results,
        best_model,
        best_model_name,
        best_predictions
    )


# ==========================================
# 7. SAVE BEST MODEL
# ==========================================

def save_best_model(best_model, best_name):

    print("\nSaving best model...")

    joblib.dump(best_model, MODEL_PATH)

    print("Best model:", best_name)

    print("Model saved at:", MODEL_PATH)


# ==========================================
# 8. SAVE METRICS
# ==========================================

def save_metrics(results, best_name):

    output = {
        "best_model": best_name,
        "models": results
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("Metrics saved at:", METRICS_PATH)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(y_test, predictions):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title("Email Spam Detection - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=150
    )

    plt.close()

    print("Confusion matrix saved:", CONFUSION_PATH)


# ==========================================
# 10. MAIN FUNCTION
# ==========================================

def main():

    print("\n" + "=" * 50)
    print("      EMAIL SPAM DETECTION SYSTEM")
    print("        MACHINE LEARNING TRAINING")
    print("=" * 50)

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = create_models()

    (
        results,
        best_model,
        best_name,
        predictions
    ) = train_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_best_model(best_model, best_name)

    save_metrics(results, best_name)

    save_confusion_matrix(y_test, predictions)

    print("\n" + "=" * 50)
    print("        MODEL TRAINING COMPLETED!")
    print("=" * 50)

    print("\nBEST MODEL:", best_name)

    print("\nAll model results:")

    for name, metrics in results.items():

        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy'] * 100:.2f}% | "
            f"F1={metrics['f1_score'] * 100:.2f}%"
        )

    print("\nReady for Streamlit frontend!")


# ==========================================
# 11. EXECUTION
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nTRAINING ERROR:", error)

        raise
