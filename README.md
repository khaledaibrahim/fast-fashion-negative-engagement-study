# fast-fashion-negative-engagement-study

Reproducible text analytics pipeline for Study 1 of a mixed-methods fast-fashion project, exploring low service quality, negative past experience, brand hate, and brand switching in customer reviews.

## What This Project Does

- Loads the Excel review dataset from a configured source path.
- Filters the exploratory sample to `1-3` star reviews.
- Cleans and standardizes review text and metadata.
- Produces descriptive summaries, sentiment scores, emotion proxies, aspect coding, topic models, and monthly trend outputs.
- Saves publication-ready tables and figures to `outputs/`.

## Study Role

This repository is the reproducible `Study 1` companion to a broader mixed-methods paper. Study 1 explores naturally occurring negative customer reviews to identify the main low-service-quality failures, emotional responses, and escalation patterns in fast fashion. Those exploratory findings then feed into `Study 2`, which tests the confirmatory structural model around low service quality, negative past experience, brand hate, and brand switching.

The same repository is also intended to function as a portfolio-style data science case study, so the workflow emphasizes transparency, reproducibility, and publication-facing outputs.

## Project Structure

- `config/default.yml`: analysis parameters and dictionaries.
- `scripts/run_analysis.py`: end-to-end entry point.
- `src/fast_fashion_analysis/`: reusable Python modules.
- `data/processed/`: generated cleaned data and model outputs.
- `outputs/figures/`: saved charts.
- `outputs/tables/`: saved summary tables.
- `reports/`: manuscript-facing narrative drafts and support documents.

## Quick Start

1. Install dependencies:

```bash
/opt/anaconda3/bin/python -m pip install -r requirements.txt
```

2. Run the pipeline:

```bash
/opt/anaconda3/bin/python scripts/run_analysis.py --config config/default.yml
```

## Notes

- The raw Excel file is referenced by absolute path in the config to avoid duplicating source data during setup.
- The current LDA topic labels are publication-facing labels derived from the normalized top-term outputs.
- The pipeline includes VADER sentiment, lexicon-based emotion/aspect scoring, sklearn LDA, BERTopic with a local sklearn embedder, and coherence diagnostics.
- The project currently assumes the Conda Python at `/opt/anaconda3/bin/python` because the BERTopic stack is installed there.
- In this local environment, BERTopic required a dependency-level patch to disable problematic numba caching inside `umap` and `pynndescent`. That patch is environment-specific rather than part of the project code.
- For manuscript support, see `reports/study1_manuscript_support.md`. Earlier working drafts are kept under `reports/archive/`.
- The GitHub-facing review-level export is `outputs/tables/study1_scored_reviews_public.csv`, which excludes reviewer identity/contact columns and redacts email addresses and phone numbers from the text fields.
- GitHub may not preview large CSV files inline. For easier download, the repository also includes `outputs/tables/study1_scored_reviews_public.zip`.
