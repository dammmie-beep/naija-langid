import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


def nfc(series: pd.Series) -> pd.Series:
    return series.map(lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x)

LANGS = ["hau", "ibo", "yor", "pcm", "eng"]
NAIJASENTI_LANGS = ["hau", "ibo", "yor", "pcm"]  # no English in NaijaSenti
SPLITS = ["train", "dev", "test"]
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_masakhanews():
    print("Downloading MasakhaNEWS...")
    for lang in LANGS:
        for split in SPLITS:
            out = RAW_DIR / f"masakhanews_{lang}_{split}.tsv"
            if out.exists():
                continue
            cached = hf_hub_download(
                "masakhane/masakhanews",
                f"data/{lang}/{split}.tsv",
                repo_type="dataset",
            )
            shutil.copy(cached, out)
            print(f"  {out}")


def download_naijasenti():
    print("Downloading NaijaSenti...")
    base = "https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets"
    for lang in NAIJASENTI_LANGS:
        for split in SPLITS:
            out = RAW_DIR / f"naijasenti_{lang}_{split}.tsv"
            if out.exists():
                continue
            r = requests.get(f"{base}/{lang}/{split}.tsv")
            r.raise_for_status()
            out.write_bytes(r.content)
            print(f"  {out}")


def build_masakhanews_splits():
    print("\nBuilding MasakhaNEWS train/val/test splits...")
    frames = []
    for lang in LANGS:
        for split in SPLITS:
            df = pd.read_csv(RAW_DIR / f"masakhanews_{lang}_{split}.tsv", sep="\t")
            df = df[["text"]].copy()
            df["lang"] = lang
            df["source"] = "masakhanews"
            df["source_split"] = split
            frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    pooled["text"] = nfc(pooled["text"])
    before = len(pooled)
    pooled = pooled.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  Pooled {before} rows → {len(pooled)} after deduplication")

    train_parts, val_parts, test_parts = [], [], []
    for lang in LANGS:
        lang_df = pooled[pooled["lang"] == lang]
        tr, temp = train_test_split(lang_df, test_size=0.20, random_state=42)
        va, te = train_test_split(temp, test_size=0.50, random_state=42)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    def shuffle_concat(parts):
        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    train = shuffle_concat(train_parts)
    val = shuffle_concat(val_parts)
    test = shuffle_concat(test_parts)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        counts = df["lang"].value_counts().to_dict()
        print(f"  {name}: {len(df)} rows — {counts}")


def build_naijasenti_stress_test():
    print("\nBuilding NaijaSenti stress test...")
    frames = []
    for lang in NAIJASENTI_LANGS:
        for split in SPLITS:
            df = pd.read_csv(RAW_DIR / f"naijasenti_{lang}_{split}.tsv", sep="\t")
            df = df[["tweet"]].rename(columns={"tweet": "text"})
            df["lang"] = lang
            df["source"] = "naijasenti"
            df["source_split"] = split
            frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    pooled["text"] = nfc(pooled["text"])
    before = len(pooled)
    pooled = pooled.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  Pooled {before} rows → {len(pooled)} after deduplication")

    pooled.to_csv(PROCESSED_DIR / "naijasenti_stress_test.csv", index=False)
    counts = pooled["lang"].value_counts().to_dict()
    print(f"  Saved {len(pooled)} rows — {counts}")


if __name__ == "__main__":
    download_masakhanews()
    download_naijasenti()
    build_masakhanews_splits()
    build_naijasenti_stress_test()
