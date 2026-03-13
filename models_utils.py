# ✅ Refactored models_utils.py for StudyMatch

import numpy as np
import json
import os
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from agents.agentic_recommender import fetch_arxiv_results, embed_text

KEYWORDS = ['spiking', 'neuromorphic', 'Josephson', 'superconduct', 'quantum', 'edge']
TOP_N = 10


def expand_query(query_text, df, embeddings, top_n=10, max_terms=5):
    q_vec = embed_text(query_text).reshape(1, -1)
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
    q_embed = embed_text(expanded).reshape(1, -1)
    sims = cosine_similarity(q_embed, embeddings)[0]
    top_indices = sims.argsort()[::-1][:50]
    reranked = rerank(df.iloc[top_indices], sims[top_indices])
    candidate_vecs = embeddings[top_indices]
    diversified = mmr_diversify(q_embed, candidate_vecs, top_k=top_n)
    final_indices = [top_indices[i] for i in diversified]
    return df.iloc[final_indices].assign(score=sims[final_indices])


def augment_with_live_arxiv(query, df_static, embeddings_static, top_n=10, cache_path="embedding_cache.json"):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    new_papers = fetch_arxiv_results.invoke({"query": query})
    if not isinstance(new_papers, list):
        return df_static

    new_rows = []
    new_embeddings = []

    for paper in new_papers:
        if not isinstance(paper, dict) or "id" not in paper:
            continue
        pid = paper["id"]
        if pid not in cache:
            embed = embed_text(paper["title"] + " " + paper["abstract"]).tolist()
            cache[pid] = {
                "embedding": embed,
                "title": paper["title"],
                "abstract": paper["abstract"],
                "authors": paper.get("authors", "Unknown"),
                "categories": paper.get("categories", "N/A")
            }

        cached = cache[pid]
        new_rows.append({
            "id": pid,
            "title": cached["title"],
            "abstract": cached["abstract"],
            "authors": cached["authors"],
            "categories": cached["categories"]
        })
        new_embeddings.append(np.array(cached["embedding"]))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    if not new_rows:
        return df_static

    df_live = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_static, df_live], ignore_index=True)
    embeddings_live = np.array(new_embeddings)
    all_embeddings = np.vstack([embeddings_static, embeddings_live]) if embeddings_static.size else embeddings_live

    return generate_recommendations(query, df_combined, all_embeddings, top_n=top_n)
