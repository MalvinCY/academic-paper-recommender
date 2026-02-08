# Academic Paper Recommender

Content-based recommendation system for discovering relevant research papers using semantic similarity.

## Project Overview

Helps researchers discover relevant academic papers based on:
- Paper abstracts or keywords
- Semantic similarity (SciBERT embeddings)
- Fast similarity search (FAISS)

**Status:** In Development

## Features (Planned)

- [ ] Data collection from arXiv API
- [ ] SciBERT embeddings for scientific text
- [ ] FAISS-based similarity search
- [ ] REST API for recommendations
- [ ] Multiple input modes (paper ID, abstract, keywords)

## Tech Stack

- **Embeddings:** SciBERT (sentence-transformers)
- **Search:** FAISS (Facebook AI Similarity Search)
- **API:** FastAPI
- **Data Source:** arXiv API

## Project Structure
```
academic-paper-recommender/
├── notebooks/          # Development & experimentation
├── src/               # Core logic (reusable code)
├── api/               # FastAPI application
├── data/              # Raw & processed data
└── models/            # Trained models & indices
```

## Setup
```bash
# Clone repository
git clone https://github.com/MalvinCY/academic-paper-recommender.git
cd academic-paper-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Progress

- [x] Project setup
- [ ] Data collection (arXiv API)
- [ ] Embedding generation
- [ ] Recommendation engine
- [ ] API development
- [ ] Deployment

## Author

**Malvin Siew**
- GitHub: [@MalvinCY](https://github.com/MalvinCY)

---

*This project complements my [Patient Sentiment Analysis](https://github.com/MalvinCY/patient-sentiment-classifier) project, demonstrating versatility across NLP classification and information retrieval tasks.*
