from typing import Tuple
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def prepare_features(
    df: pd.DataFrame,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[dict, TfidfVectorizer]:
    df = df.copy()
    df["text"] = df["text"].astype(str).apply(clean_text)
    # Map labels
    df["label_bin"] = df["label"].apply(lambda x: 1 if str(x).lower().startswith("spam") else 0)

    # train/val/test split
    train_df, temp = train_test_split(
        df, test_size=(test_size + val_size), stratify=df["label_bin"], random_state=random_state
    )
    val_frac = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp, test_size=val_frac, stratify=temp["label_bin"], random_state=random_state)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label_bin"].values
    y_val = val_df["label_bin"].values
    y_test = test_df["label_bin"].values

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }

    return splits, vectorizer
