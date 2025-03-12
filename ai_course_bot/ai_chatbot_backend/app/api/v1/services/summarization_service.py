import json
from typing import List

from app.api.v1.schemas.completion import Message
from app.api.v1.schemas.summarization import SummarizationResponse
from app.api.v1.services.rag_selector import format_chat_msg


class SummarizationService:
    """Service for summarizing chat conversations."""

    @staticmethod
    def get_summarization_prompt(messages: List[Message]) -> str:
        """
        Create a prompt for summarizing the chat conversation as a one-line title.

        Args:
            messages: List of chat messages to summarize

        Returns:
            A string prompt for the summarization model
        """
        formatted_messages = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        # Updated prompt to instruct the model to generate a short title
        return f"""
You are a helpful assistant that creates a short title for a conversation.
Generate a one-line, descriptive title that captures the main topic or question of the conversation.
Conversation:
{formatted_messages}

Title:
"""

    @staticmethod
    async def summarize_conversation(
            messages: List[Message],
            pipeline,
            max_length: int = 30,  # Reduced max_length for a short summary/title
            temperature: float = 0.7
    ) -> SummarizationResponse:
        """
        Summarize a conversation using the AI model.

        Args:
            messages: List of messages to summarize
            pipeline: The model pipeline to use
            max_length: Maximum length of the summary (set lower for a title)
            temperature: Temperature for the model

        Returns:
            SummarizationResponse containing the summary
        """
        if not messages:
            return SummarizationResponse(
                summary="No messages to summarize.",
                input_message_count=0
            )

        # Create system message with updated summarization instructions
        system_message = Message(
            role="system",
            content=SummarizationService.get_summarization_prompt(messages)
        )

        # Process with the model
        formatted_messages = format_chat_msg([system_message])
        prompt = formatted_messages[-1].content

        # Get response from model
        try:
            # Call pipeline directly (works with both real and mock pipelines)
            response_stream = pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                stream=False,
            )

            # For mock pipeline, we need to collect the tokens from the stream
            tokens = []
            for json_str in response_stream:
                try:
                    data = json.loads(json_str)
                    if data.get("type") == "token":
                        tokens.append(data.get("data", ""))
                except Exception as e:
                    continue

            summary = "".join(tokens).strip()

            # If summary is empty, provide a fallback
            if not summary:
                summary = "Failed to generate a meaningful title."

            return SummarizationResponse(
                summary=summary,
                input_message_count=len(messages)
            )

        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Failed to generate summary: {str(e)}"
            return SummarizationResponse(
                summary=error_msg,
                input_message_count=len(messages)
            )
