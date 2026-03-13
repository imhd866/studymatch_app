import numpy as np
import json
import os
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from agents.agentic_recommender import fetch_arxiv_results, embed_text

MODEL_NAME = "allenai/specter2_base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

KEYWORDS = ['spiking', 'neuromorphic', 'Josephson', 'superconduct', 'quantum', 'edge']
TOP_N = 10


def embed_text_local(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def expand_query(query_text, df, embeddings, top_n=10, max_terms=5):
    q_vec = embed_text_local(query_text).reshape(1, -1)
    sims = cosine_similarity(q_vec, embeddings)[0]
    top_docs = df.iloc[sims.argsort()[::-1][:top_n]]
    corpus = top_docs['abstract'].tolist()
    tfidf = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf.fit(corpus)
    keywords = tfidf.get_feature_names_out()[:max_terms]
    return query_text + " " + " ".join(keywords), q_vec


def rerank(results, scores):
    reranked = []
    for idx, (_, row) in enumerate(results.iterrows()):
        bonus = sum(1 for kw in KEYWORDS if kw.lower() in row['title'].lower() or kw.lower() in row['abstract'].lower())
        reranked.append((idx, scores[idx] + 0.01 * bonus))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in reranked]


def mmr_diversify(query_vec, candidate_vecs, top_k=10, lambda_param=0.7):
    selected = []
    remaining = list(range(len(candidate_vecs)))
    sims_to_query = cosine_similarity(candidate_vecs, query_vec).flatten()
    first = np.argmax(sims_to_query)
    selected.append(first)
    remaining.remove(first)
    for _ in range(1, top_k):
        mmr_scores = []
        for i in remaining:
            sim_to_query = sims_to_query[i]
            sim_to_selected = max(cosine_similarity(candidate_vecs[i].reshape(1, -1), candidate_vecs[selected])[0])
            mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((i, mmr_score))
        if not mmr_scores:
            break
        next_doc = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(next_doc)
        remaining.remove(next_doc)
    return selected


def generate_recommendations(query, df, embeddings, top_n=TOP_N):
    df = df.reset_index(drop=True)
    expanded, q_vec = expand_query(query, df, embeddings)
    q_embed = embed_text_local(expanded).reshape(1, -1)
    sims = cosine_similarity(q_embed, embeddings)[0]
    top_indices = sims.argsort()[::-1][:50]
    reranked = rerank(df.iloc[top_indices], sims[top_indices])
    candidate_vecs = embeddings[top_indices]
    diversified = mmr_diversify(q_embed, candidate_vecs, top_k=top_n)
    final_indices = [top_indices[i] for i in diversified]
    return df.iloc[final_indices].assign(score=sims[final_indices])


def augment_with_live_arxiv(query, df_static, embeddings_static, top_n=10, cache_path="embedding_cache.json"):
    # Load or create embedding cache
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    # Step 1: Fetch fresh papers from arXiv
    raw_results = fetch_arxiv_results.invoke(query)
    new_rows = []
    new_embeddings = []

    if isinstance(raw_results, str):
        return df_static  # if fetch failed or no results

    for paper in raw_results.split("\n\n"):
        match = re.search(r"\((\d{4}\.\d{4,5})\)", paper)
        if not match:
            continue
        arxiv_id = match.group(1)
        if arxiv_id in cache:
            cached = cache[arxiv_id]
        else:
            title_match = re.search(r"• (.*?) \(", paper)
            abstract_match = re.search(r"Abstract: (.*?)\.\.\.", paper, re.DOTALL)
            if not title_match or not abstract_match:
                continue
            title = title_match.group(1).strip()
            abstract = abstract_match.group(1).strip()
            embed = embed_text(title + " " + abstract).tolist()
            cached = {
                "embedding": embed,
                "title": title,
                "abstract": abstract,
                "authors": "Unknown",
                "categories": "N/A"
            }
            cache[arxiv_id] = cached
        new_rows.append({
            "id": arxiv_id,
            "title": cached["title"],
            "abstract": cached["abstract"],
            "authors": cached.get("authors", "Unknown"),
            "categories": cached.get("categories", "N/A")
        })
        new_embeddings.append(np.array(cached["embedding"]))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    df_live = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_static, df_live], ignore_index=True)
    embeddings_live = np.array(new_embeddings)
    all_embeddings = np.vstack([embeddings_static, embeddings_live]) if embeddings_live.size else embeddings_static

    return generate_recommendations(query, df_combined, all_embeddings, top_n=top_n)
