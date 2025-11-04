import os
from pathlib import Path
from typing import Union
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download_dataset(dest: Union[str, Path] = "data/sms_spam_no_header.csv") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    df = pd.read_csv(DATA_URL, header=None, names=["label", "text"], encoding_errors="replace")
    df.to_csv(dest, index=False)
    return dest


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df
