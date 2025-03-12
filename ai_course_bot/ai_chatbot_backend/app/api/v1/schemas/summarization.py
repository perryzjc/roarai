from typing import List, Optional

from pydantic import BaseModel, Field
from app.api.v1.schemas.completion import Message


class SummarizationRequest(BaseModel):
    """Request schema for summarizing a chat conversation."""
    messages: List[Message] = Field(..., description="The chat messages to summarize")
    max_length: Optional[int] = Field(500, description="Maximum length of the summary in characters")
    temperature: Optional[float] = Field(0.7, description="Temperature for the summarization model")


class SummarizationResponse(BaseModel):
    """Response schema for summarized chat conversation."""
    summary: str = Field(..., description="The generated summary of the chat conversation")
    input_message_count: int = Field(..., description="Number of messages in the input conversation") 