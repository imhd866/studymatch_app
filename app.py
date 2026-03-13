import streamlit as st
import pandas as pd
import numpy as np
from models_utils import generate_recommendations

st.set_page_config(page_title="📚 StudyMatch", layout="centered")
st.title("📚 StudyMatch: Academic Paper Recommender")
st.markdown("Paste a research topic, abstract, or interest to get personalized recommendations.")

query = st.text_area("Enter your research query:", height=200)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Generating recommendations..."):
        try:
            df = pd.read_csv("data/cleaned_arxiv_subset.csv")
            embeddings = np.load("data/specter_embeddings.npz")["embeddings"]
            results = generate_recommendations(query, df, embeddings)

            st.success("Results:")
            for _, paper in results.iterrows():
                st.markdown(f"**[{paper['title']}](https://arxiv.org/abs/{paper['id']})**")
                st.markdown(f"*{paper['authors']}*")
                st.markdown(f"`{paper['categories']}` — Score: `{paper['score']:.4f}`")
                st.markdown(paper['abstract'][:500] + "...\n")
                st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")