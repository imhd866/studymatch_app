# StudyMatch

StudyMatch is a Streamlit app that recommends academic papers from a local arXiv subset and optionally enriches the results with live arXiv lookups and LLM-generated paper assessments.

## What it does

- Accepts a research topic, abstract, or free-form interest query.
- Searches a local paper index using SPECTER-based embeddings.
- Expands the query with TF-IDF terms from the nearest abstracts.
- Re-ranks for domain keywords and diversifies results with MMR.
- Pulls a few live results from arXiv and merges them into the candidate pool.
- Optionally scores each recommendation with a Groq-hosted model for a short groundedness verdict.

## Project structure

- `app.py`: Streamlit UI.
- `models_utils.py`: retrieval, query expansion, re-ranking, and diversification.
- `agents/agentic_recommender.py`: live arXiv fetches, arXiv ID verification, embedding generation, and optional LLM scoring.
- `data/cleaned_arxiv_subset.csv`: local paper metadata.
- `data/specter_embeddings.npz`: local embedding matrix aligned with the CSV.
- `data/embedding_cache.json`: cached embeddings for repeated live papers and queries.
- `run_local.py`: convenience launcher for Streamlit.

## Requirements

- Python 3.10+ recommended.
- Internet access is needed for:
  - live arXiv fetching,
  - Groq groundedness scoring,
  - first-time download of the Hugging Face embedding model.

## Setup

1. Create and activate a virtual environment.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Configure environment variables.

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `.env` if you want groundedness scoring. If you leave it blank, the app still runs and skips that step gracefully.

## Running the app

Use either command:

```powershell
streamlit run app.py
```

or:

```powershell
py run_local.py
```

Then open the local URL shown by Streamlit in your browser.

## How recommendations are built

1. The app loads the local arXiv subset and precomputed SPECTER embeddings.
2. The query is embedded with the same model.
3. Nearby papers are used to extract a few TF-IDF expansion terms.
4. The expanded query is scored against the local embedding matrix.
5. Top candidates receive a lightweight keyword bonus.
6. MMR diversification selects a final set of recommendations.
7. A few live arXiv results are embedded and merged into the candidate pool.
8. Each final paper can be checked for arXiv link validity and optional groundedness feedback.

## Notes and limitations

- The first run can be slow because the embedding model may need to download.
- Without a configured Groq API key, groundedness scoring is skipped.
- If live arXiv calls fail, the app falls back to local recommendations.
- The local dataset is only as broad as the included CSV and embedding files.

## Troubleshooting

- `No installed Python found`: install Python and ensure `py` or `python` is available on PATH.
- `Missing required data file`: verify the `data/` directory contains both the CSV and `.npz` files.
- Hugging Face model download issues: re-run with working internet access, then try again.
- Groq auth errors: check `STUDYMATCH_GROQ_API_KEY` in `.env`.

## Security

- Do not commit real API keys to `.env`.
- Rotate any key that has already been stored in the repository or shared outside your machine.
