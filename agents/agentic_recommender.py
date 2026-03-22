import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from transformers import AutoModel, AutoTokenizer

load_dotenv(override=True)

MODEL_NAME = "allenai/specter2_base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PATH = Path("data/embedding_cache.json")
ARXIV_LINK_CACHE_PATH = Path("data/arxiv_link_cache.json")
MAX_CACHE_ENTRIES = 5000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as file_handle:
        embedding_cache = json.load(file_handle)
else:
    embedding_cache = {}

if os.path.exists(ARXIV_LINK_CACHE_PATH):
    with open(ARXIV_LINK_CACHE_PATH, "r", encoding="utf-8") as file_handle:
        arxiv_link_cache = json.load(file_handle)
else:
    arxiv_link_cache = {}


def get_llm():
    # Re-read on each call so Streamlit restarts and env changes are picked up.
    groq_api_key = (os.getenv("STUDYMATCH_GROQ_API_KEY") or "").strip().strip("\"'")
    groq_model_name = (os.getenv("STUDYMATCH_GROQ_MODEL") or "llama-3.3-70b-versatile").strip()

    if not groq_api_key:
        return None
    return ChatGroq(groq_api_key=groq_api_key, model_name=groq_model_name)


def normalize(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def normalize_title(text):
    lowered = normalize(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def embed_text(text):
    text = normalize(text)
    key = hashlib.md5(text.encode()).hexdigest()
    if key in embedding_cache:
        return np.array(embedding_cache[key])

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    embedding_cache[key] = embedding.tolist()
    if len(embedding_cache) > MAX_CACHE_ENTRIES:
        embedding_cache.pop(next(iter(embedding_cache)))
    with open(CACHE_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(embedding_cache, file_handle)

    return embedding


def verify_arxiv_link(paper_id):
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": paper_id},
            timeout=5,
        )
        response.raise_for_status()
        return "<entry>" in response.text
    except requests.RequestException:
        return False


def resolve_arxiv_link(title, fallback_id=None):
    normalized_title = normalize_title(title)
    cached = arxiv_link_cache.get(normalized_title)
    if cached is not None:
        return cached

    if fallback_id and verify_arxiv_link(fallback_id):
        resolved = {
            "id": fallback_id,
            "url": f"https://arxiv.org/abs/{fallback_id}",
            "verified": True,
        }
        arxiv_link_cache[normalized_title] = resolved
        with open(ARXIV_LINK_CACHE_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(arxiv_link_cache, file_handle)
        return resolved

    resolved = {
        "id": fallback_id or "",
        "url": f"https://arxiv.org/abs/{fallback_id}" if fallback_id else "",
        "verified": False,
    }

    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{title}"', "start": 0, "max_results": 5},
            timeout=10,
        )
        response.raise_for_status()
        entries = ET.fromstring(response.content).findall(".//{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            api_title = entry.find("{http://www.w3.org/2005/Atom}title")
            api_id = entry.find("{http://www.w3.org/2005/Atom}id")
            if api_title is None or api_id is None:
                continue

            candidate_title = api_title.text.strip()
            candidate_id = re.sub(r"v\d+$", "", api_id.text.split("/")[-1])
            if normalize_title(candidate_title) == normalized_title:
                resolved = {
                    "id": candidate_id,
                    "url": f"https://arxiv.org/abs/{candidate_id}",
                    "verified": True,
                }
                break
    except (requests.RequestException, ET.ParseError):
        pass

    arxiv_link_cache[normalized_title] = resolved
    with open(ARXIV_LINK_CACHE_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(arxiv_link_cache, file_handle)
    return resolved


def compute_groundedness(title, abstract):
    llm = get_llm()
    if llm is None:
        return None

    prompt = (
        "Rate the groundedness of this paper from 0 to 10 and explain your reasoning in short bullets.\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )
    try:
        return llm.invoke(prompt).content
    except Exception as exc:
        if "invalid_api_key" in str(exc) or "Invalid API Key" in str(exc):
            return (
                "Groundedness check failed: the Groq API key loaded by the app is invalid. "
                "If you are running a deployed Streamlit app, update the secret in Streamlit's app settings; "
                "a local .env file will not override cloud secrets."
            )
        raise


def fetch_arxiv_results(query):
    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": 5},
            timeout=10,
        )
        response.raise_for_status()
        entries = ET.fromstring(response.content).findall(".//{http://www.w3.org/2005/Atom}entry")
        results = []
        for entry in entries:
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
            authors = ", ".join(
                author.find("{http://www.w3.org/2005/Atom}name").text
                for author in entry.findall("{http://www.w3.org/2005/Atom}author")
            )
            categories = " ".join(
                category.attrib.get("term", "")
                for category in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
            ).strip() or "N/A"
            results.append(
                {
                    "id": re.sub(r"v\d+$", "", arxiv_id),
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": categories,
                }
            )
        return results
    except (requests.RequestException, ET.ParseError) as exc:
        return [{"error": f"Error querying arXiv API: {exc}"}]


def assess_recommendations(papers_df):
    enriched = []
    for _, row in papers_df.iterrows():
        paper_id = row["id"]
        title = row["title"]
        abstract = row["abstract"]

        try:
            link_resolution = resolve_arxiv_link(title, fallback_id=paper_id)
        except Exception:
            link_resolution = {
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "verified": False,
            }

        try:
            groundedness_result = compute_groundedness(title, abstract)
        except Exception as exc:
            groundedness_result = f"Groundedness check failed: {exc}"

        enriched.append(
            {
                "id": link_resolution["id"] or paper_id,
                "title": title,
                "authors": row.get("authors", ""),
                "categories": row.get("categories", ""),
                "abstract": abstract,
                "score": row.get("score", 0),
                "groundedness": groundedness_result,
                "link_verified": link_resolution["verified"],
                "resolved_url": link_resolution["url"],
            }
        )

    return enriched
