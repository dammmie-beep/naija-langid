# Data card — naija-langid

## Sources

| Dataset | HuggingFace / URL | Task | Splits used |
|---------|-------------------|------|-------------|
| MasakhaNEWS | `masakhane/masakhanews` | News topic classification | train, dev, test → pooled and re-split |
| NaijaSenti | GitHub: `hausanlp/NaijaSenti` (TSV files) | Twitter sentiment | train, dev, test → pooled, kept separate |

## Languages

| Code | Language | Present in MasakhaNEWS | Present in NaijaSenti |
|------|----------|------------------------|-----------------------|
| eng | English | Yes | No |
| hau | Hausa | Yes | Yes |
| ibo | Igbo | Yes | Yes |
| pcm | Nigerian Pidgin | Yes | Yes |
| yor | Yorùbá | Yes | Yes |

## Licences

- **MasakhaNEWS**: CC BY 4.0
- **NaijaSenti**: CC BY 4.0

## Raw file inventory

24 TSV files in `data/raw/`, named `{dataset}_{lang}_{split}.tsv`.  
The original split name is preserved in the filename so any row can be traced back to its source.

## Cleaning steps (in order)

1. Download raw TSVs with original split names preserved.
2. Extract the `text` column (MasakhaNEWS article body) or `tweet` column (NaijaSenti), renaming to `text`.
3. NFC-normalise all text strings.
4. Deduplicate on `text` at article level, within each dataset.
5. Re-split MasakhaNEWS with stratified 80 / 10 / 10 by language (random seed 42).
6. Split articles into sentences using regex on `[.?!]` + whitespace and on newlines.
7. Filter fragments: drop rows with < 20 characters or < 3 words.
8. Deduplicate on `text` at sentence level; cross-split dedup with priority train > val > test.
9. Remove from NaijaSenti stress test any sentence already present in train.

## Duplicate counts removed

| Stage | Dataset | Rows before | Rows after | Dropped |
|-------|---------|-------------|------------|---------|
| Article-level dedup | MasakhaNEWS | 13,409 | 13,280 | 129 |
| Article-level dedup | NaijaSenti | 63,550 | 61,135 | 2,415 |
| Length filter | MasakhaNEWS (sentences) | 262,047 | 258,008 | 4,039 |
| Length filter | NaijaSenti (sentences) | 87,597 | 72,186 | 15,411 |
| Cross-split dedup (val) | MasakhaNEWS (sentences) | 25,026 | 24,556 | 470 |
| Cross-split dedup (test) | MasakhaNEWS (sentences) | 25,425 | 24,838 | 587 |
| Train-overlap removal | NaijaSenti (sentences) | 72,186 | 71,813 | 373 |

## Final row counts — sentence-level files

### MasakhaNEWS (training data)

| lang | train | val | test | total |
|------|------:|----:|-----:|------:|
| eng | 91,982 | 11,178 | 11,502 | 114,662 |
| hau | 40,775 | 4,780 | 4,857 | 50,412 |
| yor | 24,353 | 3,084 | 2,980 | 30,417 |
| ibo | 23,156 | 2,987 | 3,044 | 29,187 |
| pcm | 22,426 | 2,527 | 2,455 | 27,408 |
| **total** | **202,692** | **24,556** | **24,838** | **252,086** |

### NaijaSenti (stress test, not used in training)

| lang | rows |
|------|-----:|
| hau | 24,049 |
| yor | 21,631 |
| ibo | 17,684 |
| pcm | 8,449 |
| **total** | **71,813** |

## Known issues

- **Class imbalance**: English has ~4× more training sentences than Pidgin. Per-language accuracy should be reported separately; macro-averaged metrics will mask weakness on Pidgin.
- **Regex sentence splitting**: Abbreviations (e.g. "Dr.", "Vol.") and decimal numbers produce false splits. No language-specific tokeniser was used.
- **NaijaSenti has no English**: The stress test covers only four languages. English model performance under Twitter-style distribution is untested.
- **Code-mixing**: Nigerian Pidgin shares vocabulary with English. Some PCM sentences may be miscategorised as English by a weak model.
- **MasakhaNEWS is news domain only**: The training distribution does not cover social media, SMS, or speech transcripts.
