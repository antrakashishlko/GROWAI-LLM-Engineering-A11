# GROWAI LLM Engineering - Assignment 11

## Production LLM API with Evals & Observability

A production-ready **Retrieval-Augmented Generation (RAG) API** built with **FastAPI, ChromaDB, Ollama, hybrid retrieval, reranking, response caching, streaming, Docker and DeepEval**.

This project demonstrates how to move an LLM application from a development prototype toward a **containerized, evaluated and observable production-oriented API**.

## Features

* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* Semantic search using embeddings
* BM25 keyword retrieval
* Hybrid retrieval with Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Local LLM inference using Ollama
* FastAPI REST API
* Server-Sent Events (SSE) streaming
* Exact-match response caching
* JSONL-based request observability
* Docker containerization
* Docker Compose orchestration
* Health-check endpoint
* DeepEval-based automated evaluation
* Answer Relevancy, Faithfulness and Contextual Precision evaluation

## Architecture

```text
User
 │
 ▼
FastAPI
 │
 ├── Request Validation
 │
 ├── Cache Lookup
 │
 └── RAG Pipeline
       │
       ├── ChromaDB Vector Search
       ├── BM25 Keyword Search
       ├── Reciprocal Rank Fusion
       └── Cross-Encoder Reranking
                │
                ▼
           Relevant Context
                │
                ▼
           Ollama LLM
                │
        ┌───────┴────────┐
        ▼                ▼
   Standard API      SSE Streaming
        │                │
        └───────┬────────┘
                ▼
             Response
```

## Tech Stack

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Application development         |
| FastAPI               | REST API framework              |
| Pydantic              | Request and response validation |
| LangChain             | RAG and LLM pipeline components |
| ChromaDB              | Vector database                 |
| Sentence Transformers | Text embeddings and reranking   |
| BM25                  | Keyword-based retrieval         |
| Ollama                | Local LLM inference             |
| Docker                | Containerization                |
| Docker Compose        | Multi-service orchestration     |
| DeepEval              | LLM evaluation                  |
| PyTorch               | ML runtime dependency           |

## RAG Pipeline

The system uses a hybrid retrieval architecture rather than relying on a single retrieval method.

```text
Document
   ↓
Text Chunking
   ↓
Embeddings ──────────────┐
   ↓                     │
ChromaDB Vector Search   │
                         ├──► Reciprocal Rank Fusion
BM25 Keyword Search ─────┘
                         ↓
                  Top Retrieved Chunks
                         ↓
                 Cross-Encoder Reranking
                         ↓
                    Top 3 Contexts
                         ↓
                    Ollama LLM
                         ↓
                  Grounded Response
```

### Retrieval Components

**ChromaDB**
Stores document embeddings and performs semantic similarity search.

**BM25**
Provides keyword-based retrieval to complement semantic search.

**Reciprocal Rank Fusion (RRF)**
Combines vector-search and BM25 rankings to improve retrieval robustness.

**Cross-Encoder Reranker**
Re-ranks retrieved documents according to query-document relevance before sending context to the LLM.

## API Endpoints

### Health Check

```http
GET /health
```

Returns the current API service health status.

### Chat

```http
POST /chat
```

Accepts a user question and returns a RAG-generated response.

Example request:

```json
{
  "question": "What is artificial intelligence?"
}
```

### Streaming Chat

```http
POST /chat/stream
```

Returns the generated response progressively using **Server-Sent Events (SSE)**.

This provides a more responsive experience for longer LLM responses.

## Response Caching

The API implements an exact-match in-memory cache.

```text
User Query
    ↓
Cache Lookup
    │
    ├── Cache Hit
    │      ↓
    │   Cached Response
    │
    └── Cache Miss
           ↓
       RAG Pipeline
           ↓
        Ollama LLM
           ↓
      Store Response
           ↓
        API Response
```

This avoids repeated LLM inference for identical questions.

During testing, the first request required full RAG + LLM processing, while a repeated request was served directly from the cache.

## Observability

Each request is recorded in JSON Lines format with:

* Timestamp
* HTTP method
* Request path
* User question
* Generated response
* Request latency
* Estimated input tokens
* Estimated output tokens
* Model name
* Cache-hit status

Runtime logs are written to:

```text
logs/requests.jsonl
```

## Docker Deployment

The application is containerized using Docker and orchestrated with Docker Compose.

### Services

```text
Docker Compose
│
├── API
│   └── FastAPI + RAG + Ollama client
│
└── ChromaDB
    └── Vector database
```

### Start the Application

```text
docker compose up -d
```

### Check Services

```text
docker compose ps
```

### Stop Services

```text
docker compose down
```

The API is exposed on port `8000` and ChromaDB on port `8001`.

## LLM Evaluation

The system includes automated evaluation using **DeepEval**.

Ten test cases were created to evaluate the RAG application's answer quality.

### Evaluation Metrics

| Metric               | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| Answer Relevancy     | Measures how relevant the generated answer is to the question   |
| Faithfulness         | Measures whether the answer is supported by retrieved context   |
| Contextual Precision | Measures the relevance and ranking quality of retrieved context |

### Latest Evaluation Results

| Metric               | Pass Rate |
| -------------------- | --------: |
| Answer Relevancy     |    70.00% |
| Faithfulness         |    60.00% |
| Contextual Precision |   100.00% |

The results indicate strong contextual retrieval performance, while answer faithfulness and relevancy provide clear areas for further optimization.

## Evaluation Workflow

```text
Evaluation Questions
        ↓
RAG Retrieval
        ↓
Document Reranking
        ↓
LLM Answer Generation
        ↓
DeepEval
        ↓
┌───────────────────────────┐
│ Answer Relevancy          │
│ Faithfulness              │
│ Contextual Precision      │
└───────────────────────────┘
        ↓
Evaluation Results
```

## Project Structure

```text
LLM_Production_API_MLOps/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── cache.py
│   ├── rag.py
│   ├── rag_pipeline.py
│   └── logging_middleware.py
│
├── data/
│   └── document.txt
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   └── test_cases.py
│
├── logs/
│   └── requests.jsonl
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
```

### Core Files

**app/main.py**
Defines the FastAPI application and `/health`, `/chat` and `/chat/stream` endpoints.

**app/models.py**
Defines Pydantic request and response models.

**app/rag_pipeline.py**
Implements document loading, chunking, embeddings, ChromaDB retrieval, BM25 search, RRF, cross-encoder reranking, prompt construction and Ollama inference.

**app/rag.py**
Provides the application-level interface to the RAG pipeline.

**app/cache.py**
Implements exact-match in-memory response caching.

**app/logging_middleware.py**
Handles request observability and JSONL logging.

**data/document.txt**
Contains the source knowledge used by the RAG system.

**evaluation/evaluate.py**
Runs the DeepEval evaluation pipeline.

**evaluation/test_cases.py**
Contains the 10 evaluation test cases and expected outputs.

**Dockerfile**
Defines the API container image.

**docker-compose.yml**
Orchestrates the FastAPI API and ChromaDB services.

## Local Setup

### 1. Clone the Repository

```text
git clone <your-repository-url>
cd LLM_Production_API_MLOps
```

### 2. Create Virtual Environment

```text
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```text
pip install -r requirements.txt
```

### 4. Start Ollama

Ensure Ollama is installed and running with the required model.

Check available models:

```text
ollama list
```

### 5. Run the Application

Using Docker Compose:

```text
docker compose up -d
```

## API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://localhost:8000/docs
```

The health endpoint can be accessed at:

```text
http://localhost:8000/health
```

## Production Considerations

This project demonstrates several important components involved in productionizing LLM applications:

* API-based LLM serving
* RAG-based knowledge grounding
* Hybrid information retrieval
* Document reranking
* Response caching
* Streaming inference
* Containerized deployment
* Multi-service orchestration
* Automated LLM evaluation
* Request-level observability

For a larger production deployment, the architecture could be extended with authentication, distributed caching, persistent logging, monitoring, rate limiting, CI/CD and cloud deployment.

## Future Improvements

* Replace in-memory cache with Redis
* Add authentication and API authorization
* Add rate limiting
* Improve retrieval and reranking strategies
* Expand the evaluation dataset
* Add automated regression testing
* Add structured monitoring and dashboards
* Add CI/CD pipeline
* Add persistent observability storage
* Deploy the application to a cloud environment

## Assignment

GROWAI LLM Engineering & Generative AI - Assignment 11
