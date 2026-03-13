import streamlit as st
import pandas as pd
import numpy as np
from models_utils import generate_recommendations
from agents.agentic_recommender import assess_recommendations

st.set_page_config(page_title="\U0001F4DA StudyMatch", layout="centered")
st.title("\U0001F4DA StudyMatch: Academic Paper Recommender")
st.markdown("Paste a research topic, abstract, or interest to get personalized recommendations.")

query = st.text_area("Enter your research query:", height=200)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Generating recommendations..."):
        try:
            df = pd.read_csv("data/cleaned_arxiv_subset.csv")
            embeddings = np.load("data/specter_embeddings.npz")["embeddings"]
            results = generate_recommendations(query, df, embeddings)

            # Call agentic pipeline
            enriched_results = assess_recommendations(results)

            st.success("Results:")
            for paper in enriched_results:
                st.markdown(f"**[{paper['title']}](https://arxiv.org/abs/{paper['id']})**")
                st.markdown(f"*{paper['authors']}*")
                st.markdown(f"`{paper['categories']}` — Score: `{paper['score']:.4f}`")
                st.markdown(paper['abstract'][:500] + "...\n")

                # Show agent-generated insight
                if 'groundedness' in paper:
                    st.markdown(f"**Agent Verdict:** {paper['groundedness']}")
                if 'link_verified' in paper:
                    st.markdown(f"**Link Check:** {'✅ Valid' if paper['link_verified'] else '❌ Broken' }")
                st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")
