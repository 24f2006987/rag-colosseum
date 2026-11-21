
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ABSTAIN_THRESHOLD = 0.28

app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load knowledge base
# -----------------------------

with open(KB_PATH, "r", encoding="utf-8") as file:
    passages = json.load(file)

if not isinstance(passages, list) or not passages:
    raise ValueError("knowledge_base.json must contain a non-empty JSON list.")

if not all(isinstance(p, str) for p in passages):
    raise ValueError("Every knowledge-base passage must be a string.")


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer(MODEL_NAME)

passage_embeddings = model.encode(
    passages,
    convert_to_numpy=True,
    normalize_embeddings=True
)
# Keyword-based retrieval
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

passage_tfidf = vectorizer.fit_transform(passages)


# -----------------------------
# Prepare sentences
# -----------------------------

sentence_data = []

for passage_id, passage in enumerate(passages):
    sentences = re.split(r"(?<=[.!?])\s+", passage.strip())

    for sentence in sentences:
        if sentence.strip():
            sentence_data.append({
                "passage_id": passage_id,
                "text": sentence.strip()
            })

sentence_embeddings = model.encode(
    [item["text"] for item in sentence_data],
    convert_to_numpy=True,
    normalize_embeddings=True
)


# -----------------------------
# Request and response
# -----------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


# -----------------------------
# Health check
# -----------------------------

@app.get("/")
def home():
    return {"status": "RAG API is running"}


# -----------------------------
# RAG query endpoint
# -----------------------------


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    # Semantic similarity
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    semantic_scores = passage_embeddings @ query_embedding

    # Keyword similarity
    query_tfidf = vectorizer.transform([question])
    keyword_scores = (passage_tfidf @ query_tfidf.T).toarray().flatten()

    # Combine both scores
    combined_scores = (
        0.55 * semantic_scores +
        0.45 * keyword_scores
    )
    print("\n--- RETRIEVAL DEBUG ---")
    for i in np.argsort(combined_scores)[::-1][:5]:
        print(
            "Passage:", i,
            "| Semantic:", round(float(semantic_scores[i]), 4),
            "| Keyword:", round(float(keyword_scores[i]), 4),
            "| Combined:", round(float(combined_scores[i]), 4)
            )

    # Rank passages
    ranked_indices = np.argsort(combined_scores)[::-1]

    relevant = [
        int(i) for i in ranked_indices
        if combined_scores[i] >= 0.30
    ][:request.top_k]

    if not relevant:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.0
        )

    scores = [
        round(float(np.clip(combined_scores[i], 0, 1)), 4)
        for i in relevant
    ]

    # Extract relevant sentences from selected passages


    # Prepare source IDs
    sources = relevant

    # Extract complete relevant passages
    answer = " ".join(
        passages[passage_id]
        for passage_id in sources
    )

    # Calculate confidence
    confidence = scores[0] if scores else 0.0

    return QueryResponse(
        answer=answer,
        sources=sources,
        scores=scores,
        confidence=confidence
    )
