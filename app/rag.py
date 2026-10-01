from .rag_pipeline import answer_question


def get_rag_response(question: str) -> str:
    """
    Generate an answer for the user's question
    using the existing RAG pipeline.
    """
    return answer_question(question)