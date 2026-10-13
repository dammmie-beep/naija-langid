# Data decisions log

One line per choice, with rationale.

| # | Decision | Why |
|---|----------|-----|
| 1 | **NFC-normalise all text before deduplication** | Yoruba and Hausa tone marks can arrive in NFC or NFD form depending on the scraper. Without normalisation two visually identical strings compare unequal and survive as duplicates. |
| 2 | **Deduplicate article-level before sentence splitting** | Splitting first and then deduping would catch duplicated sentences but miss the case where two splits contain the same full article sliced differently. Deduping at article level is the safer, earlier gate. |
| 3 | **NaijaSenti kept entirely separate as a stress test** | The data comes from Twitter (short, informal, code-mixed), was annotated for sentiment not language, and has no English. Mixing it into training would let the model memorise tweet-specific vocabulary rather than learning robust language features. It is more useful as an out-of-domain probe. |
| 4 | **Regex sentence splitting on `[.?!]` + whitespace / newlines** | Simple, zero-dependency, and language-agnostic — important because a Yoruba-aware tokeniser does not exist off the shelf. Accepted trade-off: abbreviations and decimal numbers occasionally produce false splits. |
| 5 | **Length filter: ≥ 20 characters AND ≥ 3 words** | Removes URL fragments, lone punctuation, partial header lines, and other noise that carries no language signal. Threshold chosen by inspection; both conditions must hold because a long URL can pass the char test but fail the word test. |
| 6 | **Cross-split deduplication (train > val > test priority)** | A sentence that appears in both train and test inflates test accuracy. Sentences are assigned to the earliest split they appear in and removed from later splits. |
| 7 | **English included as a fifth class** | BBC Pidgin articles are partly code-mixed with English. Including English as a class gives the model a target to assign English-dominant segments to, rather than forcing it to pick the nearest Nigerian language. |
| 8 | **Headlines not used (full article body only)** | Decision deferred pending confirmation of best approach. Headlines would add ~13 k shorter samples and bring training length distribution closer to tweet length, but that choice interacts with the sentence-splitting step and was left to a later decision. |
