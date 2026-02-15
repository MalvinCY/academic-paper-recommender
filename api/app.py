"""
FastAPI REST API for the Academic Paper Recommender.

All data (embeddings, FAISS index, paper metadata) is loaded once at
startup and held in memory for low-latency query responses.

Endpoints:
    GET /                          API info and available endpoints
    GET /health                    Health check and data status
    GET /recommend/paper/{id}      Paper-to-paper recommendations
    GET /random                    Random paper from the dataset
    GET /search/title/{query}      Keyword search in paper titles
    GET /stats                     Dataset statistics

Author: Malvin Siew
"""

import os
import logging

import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Academic Paper Recommender",
    description=(
        "Discover relevant research papers using SPECTER2 embeddings "
        "and FAISS similarity search"
    ),
    version="1.0.0",
)

# CORS middleware — allows cross-origin requests from any client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data loading — runs once at startup
# ---------------------------------------------------------------------------
logger.info("Loading data...")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_normalized.npy")
INDEX_PATH = os.path.join(DATA_DIR, "papers.index")
PAPERS_PATH = os.path.join(DATA_DIR, "papers_with_embeddings.pkl")

logger.info("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
logger.info(f"Loaded {len(embeddings)} embeddings")

logger.info("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
logger.info(f"FAISS index loaded ({index.ntotal} vectors)")

logger.info("Loading papers...")
df = pd.read_pickle(PAPERS_PATH)
logger.info(f"Loaded {len(df)} papers")

logger.info("=" * 50)
logger.info("API ready to serve requests")
logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def recommend_by_index(query_idx: int, k: int = 10) -> list[dict]:
    """Return the k most similar papers to the paper at *query_idx*.

    L2 distances on normalised vectors are converted to cosine
    similarities via sim = 1 - (dist^2 / 2).
    """
    query_vector = embeddings[query_idx : query_idx + 1].astype("float32")

    # k+1 because the query itself will be the top result
    distances, indices = index.search(query_vector, k + 1)

    # Drop the first result (the query paper itself)
    result_indices = indices[0][1:]
    result_distances = distances[0][1:]

    # Convert L2 distances on normalised vectors to cosine similarities
    similarities = 1 - (result_distances**2) / 2

    recommendations = []
    for idx, sim in zip(result_indices, similarities):
        paper = df.iloc[idx]
        recommendations.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "categories": paper["categories"],
                "published": paper["published"],
                "similarity": float(sim),
                "abstract": paper["abstract"][:300] + "...",
            }
        )

    return recommendations


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    """Return API metadata and available endpoints."""
    return {
        "message": "Academic Paper Recommender API",
        "version": "1.0.0",
        "description": (
            "Discover relevant research papers using SPECTER2 embeddings "
            "and FAISS similarity search"
        ),
        "endpoints": {
            "health": "GET /health — API health check",
            "recommend": "GET /recommend/paper/{paper_id} — Get recommendations",
            "random": "GET /random — Get a random paper",
            "search": "GET /search/title/{query} — Search papers by title",
            "docs": "GET /docs — Interactive API documentation",
        },
        "statistics": {
            "total_papers": len(df),
            "date_range": f"{df['published'].min()} to {df['published'].max()}",
            "categories": len(df["primary_category"].unique()),
        },
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Return health status and data-loading confirmation."""
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "embeddings_loaded": embeddings is not None,
        "papers_loaded": df is not None,
        "n_papers": len(df),
        "index_size": index.ntotal,
    }


@app.get("/recommend/paper/{paper_id}", tags=["Recommendations"])
def recommend_from_paper(
    paper_id: str,
    k: int = Query(
        5, ge=1, le=20, description="Number of recommendations to return"
    ),
):
    """Return the top-k most similar papers to the given *paper_id*."""
    matching_papers = df[df["paper_id"] == paper_id]

    if len(matching_papers) == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Paper '{paper_id}' not found in database. "
                "Try /random to get a valid paper ID."
            ),
        )

    query_idx = matching_papers.index[0]
    query_paper = matching_papers.iloc[0]

    recommendations = recommend_by_index(query_idx, k)

    return {
        "query_paper": {
            "paper_id": query_paper["paper_id"],
            "title": query_paper["title"],
            "categories": query_paper["categories"],
            "published": query_paper["published"],
            "abstract": query_paper["abstract"][:300] + "...",
        },
        "recommendations": recommendations,
        "count": len(recommendations),
    }


@app.get("/random", tags=["Exploration"])
def random_paper():
    """Return a random paper from the dataset."""
    paper = df.sample(1).iloc[0]
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "categories": paper["categories"],
        "primary_category": paper["primary_category"],
        "published": paper["published"],
        "authors": paper["authors"][:5],
        "abstract": paper["abstract"][:500] + "...",
        "pdf_url": paper["pdf_url"],
    }


@app.get("/search/title/{query}", tags=["Search"])
def search_by_title(
    query: str,
    limit: int = Query(
        10, ge=1, le=50, description="Maximum number of results"
    ),
):
    """Case-insensitive keyword search across paper titles."""
    matches = df[df["title"].str.contains(query, case=False, na=False)]
    total_matches = len(matches)
    matches = matches.head(limit)

    results = []
    for _, paper in matches.iterrows():
        results.append(
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "categories": paper["categories"],
                "published": paper["published"],
                "abstract": paper["abstract"][:200] + "...",
            }
        )

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "total_matches": total_matches,
    }


@app.get("/stats", tags=["Info"])
def get_statistics():
    """Return summary statistics for the paper dataset."""
    return {
        "total_papers": len(df),
        "date_range": {
            "earliest": df["published"].min(),
            "latest": df["published"].max(),
        },
        "categories": {
            "total_unique": len(df["primary_category"].unique()),
            "top_5": df["primary_category"].value_counts().head(5).to_dict(),
        },
        "papers_per_year": (
            df["year"].value_counts().sort_index().tail(5).to_dict()
        ),
        "average_abstract_length": int(df["abstract_length"].mean()),
    }


# ---------------------------------------------------------------------------
# Entrypoint for local development
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
