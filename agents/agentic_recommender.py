# ✅ Updated agentic_recommender.py with fixes

from langchain_core.tools import tool
from langchain_groq import ChatGroq
import requests
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import hashlib
import os
import json
import re
from sklearn.metrics.pairwise import cosine_similarity

# === Set up Groq LLM ===
llm = ChatGroq(
    groq_api_key="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",  # Replace with env var
    model_name="openai/gpt-oss-120b"
)

# === Set up local embedding model ===
MODEL_NAME = "allenai/specter2_base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# === Embedding Cache ===
CACHE_PATH = "data/embedding_cache.json"
MAX_CACHE_ENTRIES = 5000
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

def normalize(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

def embed_text(text):
    text = normalize(text)
    key = hashlib.md5(text.encode()).hexdigest()
    if key in embedding_cache:
        return np.array(embedding_cache[key])
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    embedding_cache[key] = emb.tolist()
    if len(embedding_cache) > MAX_CACHE_ENTRIES:
        embedding_cache.pop(next(iter(embedding_cache)))
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(embedding_cache, f)
    return emb

# === Tool 1: Robust arXiv link verifier ===
@tool
def verify_arxiv_link(paper_id: str) -> str:
    """Check if the arXiv paper ID is valid using the arXiv API."""
    paper_id = re.sub(r"v\d+$", "", paper_id)
    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and "<entry>" in response.text:
            return f"✅ Valid arXiv ID: {paper_id}"
        else:
            return f"❌ Invalid or missing arXiv ID: {paper_id}"
    except Exception as e:
        return f"⚠️ Error contacting arXiv API: {e}"

# === Tool 2: Groundedness scorer ===
@tool
def compute_groundedness(title: str, abstract: str) -> str:
    """Score groundedness of a paper based on its title and abstract."""
    prompt = f"Rate groundedness (0–10) and explain in bullet points.\nTitle: {title}\nAbstract: {abstract}"
    return llm.invoke(prompt).content

# === Tool 3: Live paper fetcher from arXiv ===
@tool
def fetch_arxiv_results(query: str) -> list:
    """Search arXiv for relevant papers and return metadata."""
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=25"
    try:
        response = requests.get(url)
        entries = ET.fromstring(response.content).findall(".//{http://www.w3.org/2005/Atom}entry")
        results = []
        for entry in entries:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            arxiv_id_raw = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id_raw)
            authors = ", ".join(
                author.find("{http://www.w3.org/2005/Atom}name").text
                for author in entry.findall("{http://www.w3.org/2005/Atom}author")
            )
            results.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors
            })
        return results
    except Exception as e:
        return [{"error": f"⚠️ Error querying arXiv API: {e}"}]

# === Optional reranker ===
def rerank_arxiv_results(query, papers, top_n=10):
    query_embed = embed_text(query).reshape(1, -1)
    paper_texts = [p["title"] + " " + p["abstract"] for p in papers]
    paper_vecs = np.array([embed_text(txt) for txt in paper_texts])
    sims = cosine_similarity(query_embed, paper_vecs).flatten()
    top_indices = sims.argsort()[::-1][:top_n]
    return [papers[i] | {"score": float(sims[i])} for i in top_indices]

# === Main function to assess papers ===
def assess_recommendations(papers_df):
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
            "authors": row.get("authors", ""),
            "categories": row.get("categories", ""),
            "abstract": abstract,
            "score": row.get("score", 0),
            "groundedness": groundedness_result,
            "link_verified": "✅" in link_result
        })

    return enriched
