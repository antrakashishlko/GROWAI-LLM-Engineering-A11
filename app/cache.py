# Simple in-memory cache
response_cache = {}

def get_cached_response(question: str):
    """
    Return the cached response if the exact question exists.
    Otherwise, return None.
    """
    return response_cache.get(question)

def cache_response(question: str, answer: str):
    """
    Store the question and its answer in the cache.
    """
    response_cache[question] = answer
