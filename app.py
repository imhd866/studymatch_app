import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agents.agentic_recommender import assess_recommendations
from models_utils import augment_with_live_arxiv

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "cleaned_arxiv_subset.csv"
EMBEDDINGS_PATH = DATA_DIR / "specter_embeddings.npz"
SEARCH_HISTORY_PATH = DATA_DIR / "search_history.json"
MAX_HISTORY_ITEMS = 20


@st.cache_data(show_spinner=False)
def load_local_index():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)["embeddings"]
    return df, embeddings


def load_search_history():
    if not SEARCH_HISTORY_PATH.exists():
        return []

    try:
        with open(SEARCH_HISTORY_PATH, "r", encoding="utf-8") as file_handle:
            history = json.load(file_handle)
        return history if isinstance(history, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_search_history(history):
    SEARCH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEARCH_HISTORY_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(history[:MAX_HISTORY_ITEMS], file_handle, ensure_ascii=True, indent=2)


def remember_search(query_text, results):
    history = load_search_history()
    record = {
        "query": query_text,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    history = [item for item in history if item.get("query") != query_text]
    history.insert(0, record)
    save_search_history(history)
    return record


def render_results(results):
    if not results:
        st.warning("No matching papers were found for that query.")
        return

    st.success(f"Found {len(results)} recommendations.")
    for paper in results:
        st.markdown(f"**[{paper['title']}](https://arxiv.org/abs/{paper['id']})**")
        st.markdown(f"*{paper.get('authors', '')}*")
        st.markdown(f"`{paper.get('categories', 'N/A')}` - Score: `{float(paper.get('score', 0)):.4f}`")
        st.markdown(paper["abstract"][:500] + "...\n")

        if paper.get("groundedness"):
            st.markdown(f"**Agent verdict:** {paper['groundedness']}")
        if "link_verified" in paper:
            status = "Valid" if paper["link_verified"] else "Broken"
            st.markdown(f"**Link check:** {status}")
        st.markdown("---")


st.set_page_config(page_title="StudyMatch", layout="centered")
st.title("StudyMatch: Academic Paper Recommender")
st.markdown("Paste a research topic, abstract, or interest to get personalized recommendations.")

if "active_record" not in st.session_state:
    history = load_search_history()
    st.session_state.active_record = history[0] if history else None

history = load_search_history()

with st.sidebar:
    st.subheader("Past Searches")
    if history:
        options = {
            f"{item['query'][:45]}{'...' if len(item['query']) > 45 else ''}": item for item in history
        }
        selected_label = st.selectbox("Reopen a saved search", list(options.keys()))
        if st.button("Load Selected Search", use_container_width=True):
            st.session_state.active_record = options[selected_label]
            st.rerun()
    else:
        st.caption("Your recent searches will appear here after you run one.")

query_default = st.session_state.active_record["query"] if st.session_state.active_record else ""
query = st.text_area("Enter your research query:", value=query_default, height=200)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Generating recommendations..."):
        try:
            df, embeddings = load_local_index()
            results = augment_with_live_arxiv(query, df, embeddings)
            enriched_results = assess_recommendations(results)
            st.session_state.active_record = remember_search(query.strip(), enriched_results)
        except FileNotFoundError as exc:
            st.error(f"Missing required data file: {exc}")
        except Exception as exc:
            st.error(f"Unable to generate recommendations: {exc}")

if st.session_state.active_record:
    saved_at = st.session_state.active_record.get("saved_at")
    if saved_at:
        st.caption(f"Showing saved results from {saved_at.replace('T', ' ').replace('+00:00', ' UTC')}")
    render_results(st.session_state.active_record.get("results", []))
