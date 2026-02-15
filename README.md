# Academic Paper Recommender

A content-based recommendation system for discovering relevant academic papers using semantic similarity. Given any paper in the dataset, the system retrieves the most closely related papers by comparing abstract embeddings — no collaborative filtering or user history required.

**Live demo:** [academic-paper-recommender.streamlit.app](https://academic-paper-recommender.streamlit.app/)

## How It Works

1. **Data Collection** — 9,280 papers collected from the [arXiv API](https://arxiv.org/help/api) across six ML/AI categories, using a balanced sampling strategy (recent + relevant) for temporal coverage.

2. **Embedding Generation** — Each abstract is encoded into a 768-dimensional vector using [SPECTER2](https://github.com/allenai/specter) (Allen AI), a transformer model trained on citation relationships between academic papers.

3. **Similarity Search** — Embeddings are normalised to unit length and indexed with [FAISS](https://github.com/facebookresearch/faiss) for fast nearest-neighbour retrieval. Cosine similarity scores are derived from L2 distances on the normalised vectors.

4. **Deployment** — A Streamlit web app provides the interactive frontend; a FastAPI REST API provides programmatic access.

## Features

- **Paper-to-paper recommendations** — retrieve the most similar papers to any entry in the dataset
- **Title keyword search** — find papers by keywords in the title
- **Random exploration** — discover papers with a random pick and its neighbours
- **Category browsing** — filter and paginate by arXiv discipline and subcategory
- **REST API** — all recommendation functionality exposed via FastAPI with interactive docs at `/docs`

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Embeddings | SPECTER2 (`allenai/specter2_base`) | Trained on citation graphs — suited to academic paper similarity |
| Search | FAISS (`IndexFlatL2`) | Millisecond nearest-neighbour search; scales to millions of vectors |
| Web App | Streamlit | Built-in caching and state management; rapid prototyping |
| API | FastAPI | Auto-generated OpenAPI docs; type-validated endpoints |
| Data Source | arXiv API | Free, open access to 2M+ paper metadata |

## Project Structure

```
academic-paper-recommender/
├── notebooks/
│   ├── 01_data_collection.ipynb        # arXiv API collection and cleaning
│   ├── 02_embedding_generation.ipynb   # SPECTER2 embedding pipeline
│   └── 03_recommendation_engine.ipynb  # FAISS index building and evaluation
├── api/
│   ├── __init__.py
│   └── app.py                          # FastAPI REST API
├── data/
│   ├── raw/                            # Original arXiv data (gitignored)
│   └── processed/                      # Embeddings, FAISS index (gitignored)
├── streamlit_app.py                    # Streamlit web interface
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/MalvinCY/academic-paper-recommender.git
cd academic-paper-recommender

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

To reproduce the full pipeline from scratch, run the notebooks in order (`01` → `02` → `03`). Embedding generation requires `sentence-transformers` and `torch`, and takes roughly 45 minutes on CPU.

To run the Streamlit app locally (requires the processed data files):

```bash
streamlit run streamlit_app.py
```

To start the FastAPI server:

```bash
uvicorn api.app:app --reload --port 8001
```

## Dataset

- **9,280 papers** after deduplication and quality filtering (600-character abstract minimum)
- **6 arXiv categories:** cs.AI, cs.LG, cs.CL, cs.CV, cs.IR, stat.ML
- **Date range:** 2008–2026
- **102 unique primary categories** (papers are frequently cross-listed)

## Key Design Decisions

- **SPECTER2 over SciBERT** — SciBERT is a strong scientific language model but was not trained for sentence-level similarity. SPECTER2 was trained on citation relationships, which aligns directly with the recommendation objective.

- **Balanced data collection** — Collecting papers sorted by both recency and relevance provides better temporal coverage than either strategy alone, avoiding a dataset skewed entirely towards recent or landmark papers.

- **L2 on normalised vectors** — Rather than using FAISS's cosine similarity index directly, all vectors are normalised to unit length and searched with `IndexFlatL2`. This is better optimised in FAISS, and the conversion (`sim = 1 - dist²/2`) is straightforward.

- **600-character abstract threshold** — Very short abstracts produce poor embeddings due to insufficient semantic content. This threshold removes only 2.9% of papers.

## Future Improvements

- **Text-to-paper search** — encode a free-text query with SPECTER2 at inference time to find relevant papers without requiring a seed paper
- **Larger dataset** — scale beyond 9,280 papers; the FAISS architecture supports millions of vectors with approximate index types
- **Citation-aware re-ranking** — combine embedding similarity with citation network features

## Author

**Malvin Siew** — [GitHub](https://github.com/MalvinCY)

## Licence

MIT — see [LICENSE](LICENSE) for details.
