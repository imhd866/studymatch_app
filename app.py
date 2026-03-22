from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agents.agentic_recommender import assess_recommendations
from models_utils import augment_with_live_arxiv

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "cleaned_arxiv_subset.csv"
EMBEDDINGS_PATH = DATA_DIR / "specter_embeddings.npz"


@st.cache_data(show_spinner=False)
def load_local_index():
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)["embeddings"]
    return df, embeddings


st.set_page_config(page_title="StudyMatch", layout="centered")
st.title("StudyMatch: Academic Paper Recommender")
st.markdown("Paste a research topic, abstract, or interest to get personalized recommendations.")

query = st.text_area("Enter your research query:", height=200)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Generating recommendations..."):
        try:
            df, embeddings = load_local_index()
            results = augment_with_live_arxiv(query, df, embeddings)
            enriched_results = assess_recommendations(results)

            if not enriched_results:
                st.warning("No matching papers were found for that query.")
            else:
                st.success(f"Found {len(enriched_results)} recommendations.")
                for paper in enriched_results:
                    st.markdown(f"**[{paper['title']}](https://arxiv.org/abs/{paper['id']})**")
                    st.markdown(f"*{paper['authors']}*")
                    st.markdown(f"`{paper['categories']}` - Score: `{paper['score']:.4f}`")
                    st.markdown(paper["abstract"][:500] + "...\n")

                    if paper.get("groundedness"):
                        st.markdown(f"**Agent verdict:** {paper['groundedness']}")
                    if "link_verified" in paper:
                        status = "Valid" if paper["link_verified"] else "Broken"
                        st.markdown(f"**Link check:** {status}")
                    st.markdown("---")
        except FileNotFoundError as exc:
            st.error(f"Missing required data file: {exc}")
        except Exception as exc:
            st.error(f"Unable to generate recommendations: {exc}")
