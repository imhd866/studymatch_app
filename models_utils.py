import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/specter2_base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

KEYWORDS = ['spiking', 'neuromorphic', 'Josephson', 'superconduct', 'quantum', 'edge']
TOP_N = 10

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :].squeeze().cpu().numpy()

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