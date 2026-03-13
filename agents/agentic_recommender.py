from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatGroq
from langchain.agents import AgentType
from langchain.tools import tool
import requests
import re

# === Agent LLM ===
llm = ChatGroq(
    groq_api_key="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",  # replace with your actual key or set as env var
    model_name="mixtral-8x7b-32768"
)

# === Tool 1: Link Verifier ===
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

# === Tool 2: Groundedness Checker ===
@tool
def compute_groundedness(title: str, abstract: str) -> str:
    """Compute a groundedness score using the LLM's reasoning on title + abstract."""
    prompt = f"Rate how grounded the following abstract is in real, measurable research (0-10), and explain why.\n\nTitle: {title}\nAbstract: {abstract}"
    return llm.predict(prompt)

# === Agent Setup ===
agent = initialize_agent(
    tools=[verify_arxiv_link, compute_groundedness],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# === Entry point ===
def assess_recommendations(papers_df):
    enriched = []
    for _, row in papers_df.iterrows():
        result = agent.run(f"Check link for paper ID {row['id']}. Then analyze groundedness of: '{row['title']}'\n\n{row['abstract'][:400]}...")
        enriched.append((row['id'], row['title'], result))
    return enriched
