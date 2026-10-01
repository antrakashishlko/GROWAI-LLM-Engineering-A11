"""
This project implements a production-oriented RAG API using FastAPI,
ChromaDB, and Ollama. It provides standard and streaming chat endpoints,
response caching, request-level observability, and automated LLM evaluation
with DeepEval for measuring answer quality and retrieval performance.
"""

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .cache import get_cached_response, cache_response
from .logging_middleware import write_request_log
from .models import ChatRequest, ChatResponse
from .rag import get_rag_response


app = FastAPI(
    title="ChefMate Production LLM API",
    description="Production RAG API with FastAPI, caching and observability",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """
    Health check endpoint used to verify that the API is running.
    """
    return {
        "status": "healthy",
        "service": "ChefMate Production LLM API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, chat_request: ChatRequest):
    """
    Generate an answer using the RAG pipeline.
    Uses exact-match caching and records observability data.
    """

    question = chat_request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start_time = time.perf_counter()

    # Check cache
    cached_answer = get_cached_response(question)

    if cached_answer is not None:
        answer = cached_answer
        cache_hit = True

    else:
        try:
            answer = get_rag_response(question)
            cache_response(question, answer)
            cache_hit = False

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing failed: {str(error)}"
            )

    # Calculate request latency
    latency = time.perf_counter() - start_time

    # Save observability log
    write_request_log(
        request=request,
        question=question,
        answer=answer,
        latency=latency,
        cache_hit=cache_hit
    )

    return ChatResponse(answer=answer)


def stream_text(text: str):
    """
    Stream the response text in small chunks.
    """

    words = text.split()

    for word in words:
        yield f"data: {word} \n\n"


@app.post("/chat/stream")
def chat_stream(request: Request, chat_request: ChatRequest):
    """
    Streaming version of the chat endpoint using Server-Sent Events.
    """

    question = chat_request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start_time = time.perf_counter()

    cached_answer = get_cached_response(question)

    if cached_answer is not None:
        answer = cached_answer
        cache_hit = True

    else:
        try:
            answer = get_rag_response(question)
            cache_response(question, answer)
            cache_hit = False

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing failed: {str(error)}"
            )

    latency = time.perf_counter() - start_time

    write_request_log(
        request=request,
        question=question,
        answer=answer,
        latency=latency,
        cache_hit=cache_hit
    )

    return StreamingResponse(
        stream_text(answer),
        media_type="text/event-stream"
    )
