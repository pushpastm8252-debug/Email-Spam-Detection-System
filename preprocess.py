
import re
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

ZIP_PATH = DATASET_DIR / "sms_spam_collection.zip"

RAW_FILE = DATASET_DIR / "SMSSpamCollection"

RAW_CSV = DATASET_DIR / "raw_spam_data.csv"

CLEANED_CSV = DATASET_DIR / "cleaned_spam_data.csv"


# ==========================================
# 2. DOWNLOAD DATASET
# ==========================================

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)


def download_dataset():

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 0:
        print("Dataset already exists. Skipping download.")
        return

    print("\nDownloading UCI Spam Dataset...")

    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

        print("Download completed!")

        print("\nExtracting dataset...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:

            member = next(
                (
                    name for name in zip_ref.namelist()
                    if Path(name).name == "SMSSpamCollection"
                ),
                None
            )

            if member is None:
                raise FileNotFoundError(
                    "SMSSpamCollection file not found inside ZIP."
                )

            with zip_ref.open(member) as source:
                with open(RAW_FILE, "wb") as destination:
                    destination.write(source.read())

        print("Dataset extracted successfully!")

    except Exception as error:

        print("\nDataset download/extraction failed!")
        raise RuntimeError(str(error)) from error


# ==========================================
# 3. LOAD DATASET
# ==========================================

def load_dataset():

    print("\nLoading dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Dataset file is missing. Run download first."
        )

    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="error"
    )

    if df.empty:
        raise ValueError("Dataset is empty!")

    print("Dataset loaded successfully!")
    print("Original rows:", len(df))

    return df


# ==========================================
# 4. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " EMAIL ",
        text
    )

    # Normalize extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 5. PREPROCESS DATASET
# ==========================================

def preprocess_dataset(df):

    print("\nStarting data preprocessing...")

    df = df.copy()

    # Remove missing values
    df = df.dropna(subset=["label", "message"])

    # Normalize labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Keep valid labels only
    df = df[df["label"].isin(["spam", "ham"])].copy()

    # Clean original message whitespace
    df["message"] = (
        df["message"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages
    df = df[df["message"] != ""].copy()

    # Remove exact duplicate records
    df = df.drop_duplicates(
        subset=["label", "message"]
    ).copy()

    # Clean text
    df["cleaned_message"] = (
        df["message"].apply(clean_text)
    )

    # Remove messages that become empty
    df = df[df["cleaned_message"] != ""].copy()

    # Encode labels
    df["label_encoded"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Final validation
    df = df.dropna(
        subset=["cleaned_message", "label_encoded"]
    ).copy()

    df["label_encoded"] = (
        df["label_encoded"].astype(int)
    )

    df = df.reset_index(drop=True)

    print("Preprocessing completed!")

    return df


# ==========================================
# 6. SAVE DATASETS
# ==========================================

def save_dataset(raw_df, cleaned_df):

    print("\nSaving datasets...")

    raw_df.to_csv(
        RAW_CSV,
        index=False,
        encoding="utf-8"
    )

    cleaned_df.to_csv(
        CLEANED_CSV,
        index=False,
        encoding="utf-8"
    )

    print("Raw dataset saved:", RAW_CSV)
    print("Cleaned dataset saved:", CLEANED_CSV)


# ==========================================
# 7. DATASET SUMMARY
# ==========================================

def show_summary(df):

    print("\n" + "=" * 45)
    print("       EMAIL SPAM DETECTION")
    print("          DATASET SUMMARY")
    print("=" * 45)

    print("\nTotal messages:", len(df))

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nClass percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample cleaned messages:")
    print(
        df[["label", "cleaned_message"]].head(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 45)


# ==========================================
# 8. MAIN FUNCTION
# ==========================================

def main():

    print("\nStarting Spam Dataset Preprocessing...")

    download_dataset()

    raw_df = load_dataset()

    print("\nOriginal dataset shape:", raw_df.shape)

    cleaned_df = preprocess_dataset(raw_df)

    print("\nCleaned dataset shape:", cleaned_df.shape)

    save_dataset(raw_df, cleaned_df)

    show_summary(cleaned_df)

    print("\nSUCCESS: DATA PREPROCESSING COMPLETED!")
    print("Ready for Machine Learning Training.")


# ==========================================
# 9. PROGRAM EXECUTION
# ==========================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nERROR:", error)
        print("Please fix the error and run again.")
