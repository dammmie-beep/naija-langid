"""
Sentence-level processing of the article-level processed files.

Input:  data/processed/{train,val,test}.csv  +  naijasenti_stress_test.csv
Output: data/processed/sentences_{train,val,test}.csv
        data/processed/sentences_naijasenti_stress_test.csv
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
MIN_CHARS = 20
MIN_WORDS = 3

# Sentence boundary: one or more sentence-ending punctuation marks followed by
# whitespace, OR one or more newlines.
_SENT_RE = re.compile(r"(?<=[.?!])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def is_long_enough(text: str) -> bool:
    return len(text) >= MIN_CHARS and len(text.split()) >= MIN_WORDS


def nfc(series: pd.Series) -> pd.Series:
    return series.map(lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x)


# ── Step 1: sentence-split each article split ────────────────────────────────

def build_sentence_splits():
    print("Splitting articles into sentences...")
    split_dfs = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
        rows = []
        for _, row in df.iterrows():
            for sent in split_sentences(str(row["text"])):
                rows.append({
                    "text": sent,
                    "lang": row["lang"],
                    "source": row["source"],
                    "source_split": row["source_split"],
                    "split": split,
                })
        split_dfs[split] = pd.DataFrame(rows)
        print(f"  {split}: {len(df)} articles → {len(split_dfs[split])} sentences")

    # NaijaSenti tweets are already sentence-length; splitting still normalises
    # any multi-sentence tweets.
    senti = pd.read_csv(PROCESSED_DIR / "naijasenti_stress_test.csv")
    senti_rows = []
    for _, row in senti.iterrows():
        for sent in split_sentences(str(row["text"])):
            senti_rows.append({
                "text": sent,
                "lang": row["lang"],
                "source": row["source"],
                "source_split": row["source_split"],
                "split": "stress_test",
            })
    senti_df = pd.DataFrame(senti_rows)
    print(f"  stress_test: {len(senti)} tweets → {len(senti_df)} sentences")

    return split_dfs, senti_df


# ── Step 2: NFC + length filter ──────────────────────────────────────────────

def filter_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = nfc(df["text"])
    before = len(df)
    df = df[df["text"].map(is_long_enough)].reset_index(drop=True)
    print(f"  {label}: {before} → {len(df)} after length filter (dropped {before - len(df)})")
    return df


# ── Step 3: deduplication within + across splits ─────────────────────────────

def deduplicate_across(split_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Remove duplicates within each split, then remove from val/test any sentence
    already seen in train, and from test any sentence already seen in val.
    Priority: train > val > test (sentences stay in the earliest split).
    """
    result = {}
    seen: set[str] = set()
    for split in ["train", "val", "test"]:
        df = split_dfs[split].drop_duplicates(subset="text").reset_index(drop=True)
        before = len(df)
        df = df[~df["text"].isin(seen)].reset_index(drop=True)
        seen.update(df["text"].tolist())
        print(f"  {split}: {before} → {len(df)} after cross-split dedup (dropped {before - len(df)})")
        result[split] = df
    return result


def deduplicate_senti(senti_df: pd.DataFrame, train_texts: set[str]) -> pd.DataFrame:
    before = len(senti_df)
    senti_df = senti_df.drop_duplicates(subset="text").reset_index(drop=True)
    # Also remove any stress-test sentence that leaked into training data.
    senti_df = senti_df[~senti_df["text"].isin(train_texts)].reset_index(drop=True)
    print(f"  stress_test: {before} → {len(senti_df)} after dedup (dropped {before - len(senti_df)})")
    return senti_df


# ── Step 4: boilerplate inspection ───────────────────────────────────────────

def show_top_sentences(split_dfs: dict[str, pd.DataFrame], n: int = 20):
    train = split_dfs["train"]
    print(f"\nTop {n} most frequent sentences per language (train set):")
    for lang in sorted(train["lang"].unique()):
        counts = train[train["lang"] == lang]["text"].value_counts().head(n)
        print(f"\n  [{lang}]")
        for text, cnt in counts.items():
            preview = text[:90].replace("\n", " ")
            print(f"    {cnt:>4}x  {preview}")


# ── Step 5: summary table ─────────────────────────────────────────────────────

def print_summary(split_dfs: dict[str, pd.DataFrame], senti_df: pd.DataFrame):
    print("\nRow counts per language per split (sentences):")
    all_langs = sorted(set().union(*[df["lang"].unique() for df in split_dfs.values()]))
    header = f"{'lang':<6}" + "".join(f"{s:>10}" for s in ["train", "val", "test", "stress_test"])
    print("  " + header)
    print("  " + "-" * len(header))
    for lang in all_langs:
        row = f"{lang:<6}"
        for split, df in split_dfs.items():
            row += f"{len(df[df['lang']==lang]):>10,}"
        st_count = len(senti_df[senti_df["lang"] == lang]) if lang in senti_df["lang"].values else 0
        row += f"{st_count:>10,}"
        print("  " + row)
    totals = f"{'TOTAL':<6}"
    for df in split_dfs.values():
        totals += f"{len(df):>10,}"
    totals += f"{len(senti_df):>10,}"
    print("  " + "-" * len(header))
    print("  " + totals)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    split_dfs, senti_df = build_sentence_splits()

    print("\nApplying length filter...")
    split_dfs = {s: filter_df(df, s) for s, df in split_dfs.items()}
    senti_df = filter_df(senti_df, "stress_test")

    print("\nDeduplicating...")
    split_dfs = deduplicate_across(split_dfs)
    all_train_texts = set(split_dfs["train"]["text"].tolist())
    senti_df = deduplicate_senti(senti_df, all_train_texts)

    show_top_sentences(split_dfs)

    print("\nSaving...")
    for split, df in split_dfs.items():
        out = PROCESSED_DIR / f"sentences_{split}.csv"
        df.to_csv(out, index=False)
        print(f"  {out}")
    senti_out = PROCESSED_DIR / "sentences_naijasenti_stress_test.csv"
    senti_df.to_csv(senti_out, index=False)
    print(f"  {senti_out}")

    print_summary(split_dfs, senti_df)
