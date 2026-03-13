from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import requests

# === Set up Groq LLM ===
llm = ChatGroq(
    groq_api_key="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",  # Replace with env var in production
    model_name="openai/gpt-oss-120b"
)

# === Define Tools ===
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

@tool
def compute_groundedness(info: dict) -> str:
    """Score groundedness of a paper based on its title and abstract."""
    title = info.get("title", "")
    abstract = info.get("abstract", "")
    prompt = f"Rate groundedness (0-10) and explain.\nTitle: {title}\nAbstract: {abstract}"
    return llm.invoke(prompt).content

# === Build ReAct Agent ===
tools = [verify_arxiv_link, compute_groundedness]
agent = create_agent(model=llm, tools=tools)
agent_executor = agent.with_config({"recursion_limit": 100})

# === Entry Function ===
def assess_recommendations(papers_df):
    enriched = []
    for _, row in papers_df.iterrows():
        try:
            # Pass prompt inside a HumanMessage and inside a 'messages' key
            response = agent_executor.invoke({
                "messages": [
                    HumanMessage(content=(
                        f"1. Check arXiv link for paper ID {row['id']}\n"
                        f"2. Rate how grounded this research is:\n"
                        f"{row['title']}\n{row['abstract'][:500]}..."
                    ))
                ]
            })

            # The LangChain LangGraph agent returns the last message object
            if isinstance(response, list) and len(response) > 0:
                output = response[-1].content
            elif hasattr(response, "content"):
                output = response.content
            else:
                output = "⚠️ No valid response received"
                
        except Exception as e:
            output = f"⚠️ Agent error: {e}"

        enriched.append({
            "id": row['id'],
            "title": row['title'],
            "authors": row['authors'],
            "categories": row['categories'],
            "abstract": row['abstract'],
            "score": row['score'],
            "groundedness": output,
            "link_verified": "✅" in output
        })

    return enriched