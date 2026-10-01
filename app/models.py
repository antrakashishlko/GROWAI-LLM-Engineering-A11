from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User's question"
    )


class ChatResponse(BaseModel):
    answer: str