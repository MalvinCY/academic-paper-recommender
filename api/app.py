"""
Academic Paper Recommender API

FastAPI application for recommending research papers using FAISS similarity search.

Features:
- Paper-to-paper recommendations
- Random paper exploration
- Title keyword search
- Interactive API documentation

Note: Text-based search (user query → embeddings) planned for v2.0
"""

import os
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= INITIALIZE APP =============
app = FastAPI(
    title="Academic Paper Recommender",
    description="Discover relevant research papers using SPECTER2 embeddings and FAISS similarity search",
    version="1.0.0",
)

# Add CORS middleware (allows frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= LOAD DATA AT STARTUP =============
logger.info("Loading data...")

# Paths (relative to api/ directory)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings_normalized.npy")
INDEX_PATH = os.path.join(DATA_DIR, "papers.index")
PAPERS_PATH = os.path.join(DATA_DIR, "papers_with_embeddings.pkl")

# Load normalized embeddings
logger.info("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH)
logger.info(f"✓ Loaded {len(embeddings)} embeddings")

# Load FAISS index
logger.info("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
logger.info(f"✓ FAISS index loaded ({index.ntotal} vectors)")

# Load papers dataframe
logger.info("Loading papers...")
df = pd.read_pickle(PAPERS_PATH)
logger.info(f"✓ Loaded {len(df)} papers")

logger.info("=" * 50)
logger.info("✓ API ready to serve requests!")
logger.info("=" * 50)

# ============= HELPER FUNCTIONS =============


def recommend_by_index(query_idx: int, k: int = 10):
    """
    Recommend papers given a paper index

    Args:
        query_idx: Index of the query paper in the dataframe
        k: Number of recommendations to return

    Returns:
        List of recommended papers with similarity scores
    """
    # Get query vector
    query_vector = embeddings[query_idx : query_idx + 1].astype("float32")

    # Search FAISS index (k+1 to include query itself)
    distances, indices = index.search(query_vector, k + 1)

    # Skip first result (query itself)
    result_indices = indices[0][1:]
    result_distances = distances[0][1:]

    # Convert L2 distances to cosine similarities
    # For normalized vectors: cosine_sim = 1 - (L2_distance² / 2)
    similarities = 1 - (result_distances**2) / 2

    # Build recommendations list
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


# ============= API ENDPOINTS =============


@app.get("/", tags=["Info"])
def root():
    """API information and available endpoints"""
    return {
        "message": "Academic Paper Recommender API",
        "version": "1.0.0",
        "description": "Discover relevant research papers using SPECTER2 embeddings and FAISS similarity search",
        "endpoints": {
            "health": "GET /health - API health check",
            "recommend": "GET /recommend/paper/{paper_id} - Get paper recommendations",
            "random": "GET /random - Get a random paper",
            "search": "GET /search/title/{query} - Search papers by title",
            "docs": "GET /docs - Interactive API documentation",
        },
        "statistics": {
            "total_papers": len(df),
            "date_range": f"{df['published'].min()} to {df['published'].max()}",
            "categories": len(df["primary_category"].unique()),
        },
        "note": "Text-based search (semantic query → papers) coming in v2.0",
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Check API health and data status"""
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
    k: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
):
    """
    Get paper recommendations based on a paper ID

    Args:
        paper_id: arXiv paper ID (e.g., "2301.07041", "1706.03762v1")
        k: Number of recommendations (1-20)

    Returns:
        Query paper info and list of recommended papers with similarity scores

    Example paper IDs to try:
    - 2301.07041 (recent ML paper)
    - 1706.03762 (Transformer paper)
    - 2005.14165 (GPT-3 paper)
    """
    # Find paper by ID
    matching_papers = df[df["paper_id"] == paper_id]

    if len(matching_papers) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found in database. Try /random to get a valid paper ID.",
        )

    # Get paper index and metadata
    query_idx = matching_papers.index[0]
    query_paper = matching_papers.iloc[0]

    # Get recommendations
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
    """
    Get a random paper from the database

    Useful for:
    - Exploring the dataset
    - Getting valid paper IDs to test recommendations
    - Discovery

    Returns:
        Random paper with full metadata
    """
    paper = df.sample(1).iloc[0]
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "categories": paper["categories"],
        "primary_category": paper["primary_category"],
        "published": paper["published"],
        "authors": paper["authors"][:5],  # First 5 authors
        "abstract": paper["abstract"][:500] + "...",
        "pdf_url": paper["pdf_url"],
    }


@app.get("/search/title/{query}", tags=["Search"])
def search_by_title(
    query: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
):
    """
    Simple keyword search in paper titles

    Args:
        query: Search keywords (case-insensitive)
        limit: Maximum results to return (1-50)

    Returns:
        List of papers with matching titles

    Example queries:
    - "transformer"
    - "reinforcement learning"
    - "computer vision"
    """
    # Search for keyword in titles (case-insensitive)
    matches = df[df["title"].str.contains(query, case=False, na=False)]

    # Limit results
    matches = matches.head(limit)

    # Format results
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
        "total_matches": len(df[df["title"].str.contains(query, case=False, na=False)]),
    }


@app.get("/stats", tags=["Info"])
def get_statistics():
    """
    Get dataset statistics

    Returns:
        Various statistics about the paper database
    """
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
        "papers_per_year": df["year"].value_counts().sort_index().tail(5).to_dict(),
        "average_abstract_length": int(df["abstract_length"].mean()),
    }


# ============= RUN SERVER =============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
