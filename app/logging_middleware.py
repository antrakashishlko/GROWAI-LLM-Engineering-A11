import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "requests.jsonl"


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a simple word-based approximation.
    This is not the model's exact tokenizer.
    """
    if not text:
        return 0

    return len(text.split())


def write_request_log(
    request: Request,
    question: str,
    answer: str,
    latency: float,
    cache_hit: bool,
    model_name: str = "qwen3:0.6b"
):
    """
    Store request information as one JSON object per line.
    """

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "question": question,
        "answer": answer,
        "latency_seconds": round(latency, 4),
        "estimated_input_tokens": estimate_tokens(question),
        "estimated_output_tokens": estimate_tokens(answer),
        "model": model_name,
        "cache_hit": cache_hit
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")