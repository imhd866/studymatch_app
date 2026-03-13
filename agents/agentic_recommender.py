# 📁 File: agents/agentic_recommender.py

from langchain_core.tools import tool
from langchain_groq import ChatGroq
import requests

# === Set up Groq LLM ===
llm = ChatGroq(
    groq_api_key="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",  # <- replace with environment variable in production
    model_name="mixtral-8x7b-32768"
)

# === Tool 1: Verify arXiv link ===
@tool
def verify_arxiv_link(paper_id: str) -> str:
    """Check if the arXiv paper link is valid and accessible."""
    url = f"https://arxiv.org/abs/{paper_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"✅ Valid link: {url}"
        else:
            return f"❌ Invalid or unavailable: {url}"
    except Exception as e:
        return f"⚠️ Error checking {url}: {e}"

# === Tool 2: Compute groundedness ===
@tool
def compute_groundedness(info: dict) -> str:
    """Score groundedness of a paper based on its title and abstract."""
    title = info.get("title", "")
    abstract = info.get("abstract", "")
    prompt = f"Rate groundedness (0-10) and explain.\nTitle: {title}\nAbstract: {abstract}"
    return llm.invoke(prompt).content

# === Main enrichment function ===
def assess_recommendations(papers_df):
    """
    Enhance recommended papers by verifying arXiv links and scoring groundedness.
    Calls tools directly instead of using agent planner.
    """
    enriched = []
    for _, row in papers_df.iterrows():
        paper_id = row['id']
        title = row['title']
        abstract = row['abstract']

        try:
            link_result = verify_arxiv_link.invoke(paper_id)
        except Exception as e:
            link_result = f"⚠️ Link check error: {e}"

        try:
            groundedness_result = compute_groundedness.invoke({
                "title": title,
                "abstract": abstract
            })
        except Exception as e:
            groundedness_result = f"⚠️ Groundedness error: {e}"

        enriched.append({
            "id": paper_id,
            "title": title,
            "authors": row['authors'],
            "categories": row['categories'],
            "abstract": abstract,
            "score": row['score'],
            "groundedness": groundedness_result,
            "link_verified": "✅" in link_result
        })

    return enriched
