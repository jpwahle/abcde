# ABCDE v1.1 — HasBPM correction and moral features

Dataset: [Hugging Face v1.1](https://huggingface.co/datasets/jpwahle/abcde/tree/v1.1). See the [dataset card](https://huggingface.co/datasets/jpwahle/abcde), [feature definitions](https://huggingface.co/datasets/jpwahle/abcde/blob/v1.1/releases/v1.1/FEATURES.md), and [citation guide](CITATIONS.md).

This release extends the published ABCDE snapshot with the committed whole-word body-part matching correction and 48 moral-foundation features per text field. It contains 264,210,076 enriched text records across 31 source tables, plus three unchanged demographic-user tables.

## Changes

- Recalculates `HasBPM` and the five possessive BPM columns, including answer/reasoning prefixes. Removes 34,056,405 prior positive flags; changes 8 known-false flags to true and recomputes 2,325 missing flags.
- Adds averages, matched-token counts, and presence/count features for Care, Loyalty, Purity, and Authority. Dimension relevance counts nonneutral words from either pole; opposite poles can cancel the average while the presence flag remains set.
- Uses the existing lexicon conventions: lowercase whitespace tokens, fine ordinal scores from −3 to +3, and repeated-token counts.
- Provides 31 source-specific Parquet configurations for Hugging Face loading. The three demographic-user tables remain byte-identical TSV copies.
- Adds cumulative feature/source citation guidance, including Daniela Teodorescu et al. (2026) for age data, both VAD papers, TUSC, and the compilation paper or resource for each AI dataset.

The source moral lexicons are unpublished and excluded from the release. No lexicon files or raw lexicon entries are uploaded to Hugging Face or GitHub.

## Validation

All 28 tests passed. Full release verification passed HF source Git/LFS hashes, SHA-256 checksums, Parquet page checksums, file inventory and row counts, moral-feature invariants on every row, and reference feature checks on samples from every source. Each output shard was read back and every unchanged field compared with its parsed source.

## Compatibility and inherited limitations

Historical TSV field values are preserved as strings, including IDs, decimal precision, empty strings, and literal NA-like values. Updated BPM flags are boolean; new moral averages are float64 and counts/flags are int64. Cast historical numeric fields explicitly for analysis.

MAGE retains the character-spaced text present in the upstream snapshot, limiting lexicon coverage. STAR-1 retains all 1,000 original rows, with separate answer/reasoning BPM and moral features; the upstream file had no header or historical feature columns.

Source revision: `d6797a1c0c935de66c6a022e3d0ed1ea705c4c10`.

## Loading

```python
from datasets import load_dataset

rows = load_dataset(
    "jpwahle/abcde", "tusc-country-posts",
    revision="v1.1", split="train", streaming=True,
)
```

Choose the source-specific configuration listed on Hugging Face. The `train` name is a storage split, not an experimental split.
