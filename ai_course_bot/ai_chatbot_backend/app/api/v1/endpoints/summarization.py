from fastapi import APIRouter, Depends

from app.api.v1.schemas.summarization import SummarizationRequest, SummarizationResponse
from app.api.v1.services.summarization_service import SummarizationService
from app.dependencies.model import get_model_pipeline


router = APIRouter()


@router.post("", response_model=SummarizationResponse)
async def summarize_chat(
    request: SummarizationRequest,
    pipeline=Depends(get_model_pipeline)
) -> SummarizationResponse:
    """
    Summarize a chat conversation.
    
    Args:
        request: The summarization request containing messages to summarize
        pipeline: The model pipeline dependency
        
    Returns:
        SummarizationResponse with the generated summary
    """
    return await SummarizationService.summarize_conversation(
        messages=request.messages,
        pipeline=pipeline,
        max_length=request.max_length,
        temperature=request.temperature
    )
