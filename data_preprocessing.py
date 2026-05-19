"""
data_preprocessing.py — Generic and dataset-specific preprocessors.

The generic `preprocess()` function is called by DatasetManager on every
CSV before it is passed to the adapter.  The named functions are kept for
standalone / notebook use.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ─────────────────────────────────────────────────────────────────────
#  Generic preprocessor  (used by DatasetManager)
# ─────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight, schema-agnostic cleaning applied to every loaded CSV.

    Steps
    -----
    1. Drop fully-duplicate rows.
    2. Fill missing object columns with "Unknown".
    3. Fill missing numeric columns with the column median.
    4. Strip leading/trailing whitespace from all column names so that
       'Book-Title ' and 'Book-Title' are treated identically.
    """
    df = df.copy()

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


# ─────────────────────────────────────────────────────────────────────
#  Dataset-specific preprocessors  (standalone / notebook use)
# ─────────────────────────────────────────────────────────────────────

def preprocess_books_data(filepath: str = "datasets/booksdata.csv") -> pd.DataFrame:
    """
    Preprocess the books metadata dataset.

    - Removes duplicates
    - Fills missing values
    - Label-encodes 'authors' and 'publisher'
    - Min-max normalises 'rating' → 'rating_normalized'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)
    print(f"[books]  original shape : {df.shape}")

    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    for col in ["authors", "publisher"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    if "rating" in df.columns:
        scaler = MinMaxScaler()
        df["rating_normalized"] = scaler.fit_transform(df[["rating"]])

    print(f"[books]  final shape    : {df.shape}")
    return df


def preprocess_ratings_data(filepath: str = "datasets/ratings.csv") -> pd.DataFrame:
    """
    Preprocess the user-ratings dataset.

    - Removes duplicate (user, book) pairs
    - Fills missing values
    - Min-max normalises 'rating' → 'rating_normalized'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)
    print(f"[ratings] original shape : {df.shape}")

    # detect user/item columns robustly
    user_col = _find_col(df.columns, ["user_id", "user-id", "userid"])
    item_col = _find_col(df.columns, ["book_id", "isbn", "item_id"])

    if user_col and item_col:
        df = df.drop_duplicates(subset=[user_col, item_col])
    else:
        df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    rating_col = _find_col(df.columns, ["rating", "book-rating", "book_rating", "score"])
    if rating_col:
        scaler = MinMaxScaler()
        df["rating_normalized"] = scaler.fit_transform(df[[rating_col]])

    print(f"[ratings] final shape    : {df.shape}")
    return df


def preprocess_sentiment_data(filepath: str = "datasets/customer_sentiment.csv") -> pd.DataFrame:
    """
    Preprocess the customer-sentiment dataset.

    - Removes duplicates
    - Fills missing values
    - Label-encodes key categorical columns
    - Min-max normalises 'customer_rating' → 'rating_normalized'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    df = pd.read_csv(filepath)
    print(f"[sentiment] original shape : {df.shape}")

    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    for col in ["gender", "age_group", "region", "product_category",
                "purchase_channel", "platform", "sentiment"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    if "customer_rating" in df.columns:
        scaler = MinMaxScaler()
        df["rating_normalized"] = scaler.fit_transform(df[["customer_rating"]])

    print(f"[sentiment] final shape    : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────
#  Internal helper
# ─────────────────────────────────────────────────────────────────────

def _find_col(columns, candidates: list) -> str | None:
    """Return the first column whose normalised name matches a candidate."""
    norm = {c: c.lower().replace("-", "").replace("_", "").replace(" ", "")
            for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace("-", "").replace("_", "").replace(" ", "")
        for col, n in norm.items():
            if key == n:
                return col
    return None


# ─────────────────────────────────────────────────────────────────────
#  Standalone entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Preprocessing Books Data ===")
    books_df = preprocess_books_data()

    print("\n=== Preprocessing Ratings Data ===")
    ratings_df = preprocess_ratings_data()

    print("\n=== Preprocessing Sentiment Data ===")
    sentiment_df = preprocess_sentiment_data()

    print("\nAll datasets preprocessed successfully!")
    print(f"Books    : {books_df.shape}")
    print(f"Ratings  : {ratings_df.shape}")
    print(f"Sentiment: {sentiment_df.shape}")