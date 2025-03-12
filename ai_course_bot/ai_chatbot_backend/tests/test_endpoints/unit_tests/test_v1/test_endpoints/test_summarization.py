import pytest
from app.api.v1.schemas.completion import Message


@pytest.fixture
def conversation(tai_trivia_question):
    return [
        Message(
            role="user",
            content=tai_trivia_question
        ),
        Message(
            role="assistant",
            content=(
                "TAI (Teaching Assistant Intelligence) is a project at UC Berkeley focused on "
                "developing AI tools to assist teaching and learning in diverse courses. "
                "It aims to provide personalized help to students and reduce the workload on human teaching assistants."
            )
        )
    ]


def test_summarize_chat(client_unit, conversation):
    """
    Test the summarization endpoint with a standard conversation payload.
    Verifies that the response includes a non-empty summary containing expected keywords,
    and that the input message count matches the number of provided messages.
    """
    payload = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in conversation],
        "max_length": 200,
        "temperature": 0.7,
    }

    response = client_unit.post("/v1/summarization", json=payload)
    assert response.status_code == 200, "Expected status code 200."

    data = response.json()
    assert "summary" in data, "Response should include a 'summary' field."
    assert "input_message_count" in data, "Response should include an 'input_message_count' field."
    assert data["input_message_count"] == len(conversation), "Mismatch in input message count."

    summary = data["summary"].strip()
    assert summary, "Summary should not be empty."

    summary_lower = summary.lower()
    for keyword in ["tai", "berkeley", "teaching"]:
        assert keyword in summary_lower, f"Expected keyword '{keyword}' in summary."


def test_summarize_empty_chat(client_unit):
    """
    Test the summarization endpoint with an empty conversation.
    Ensures that the endpoint returns a proper fallback message when no messages are provided.
    """
    payload = {
        "messages": [],
        "max_length": 100,
        "temperature": 0.5,
    }

    response = client_unit.post("/v1/summarization", json=payload)
    assert response.status_code == 200, "Expected status code 200 for empty conversation."

    data = response.json()
    assert "summary" in data, "Response should include a 'summary' field."
    assert "input_message_count" in data, "Response should include an 'input_message_count' field."
    assert data["input_message_count"] == 0, "Input message count should be 0 for empty conversation."
    assert "No messages to summarize" in data["summary"], "Expected fallback message for empty conversation."


def test_summarize_chat_max_length(client_unit, tai_trivia_question):
    """
    Test that the summarization endpoint honors the max_length parameter.
    Uses a small max_length to verify that the output summary is truncated appropriately.
    """
    conversation = [
        Message(role="user", content=tai_trivia_question),
        Message(role="assistant", content="Detailed explanation about TAI and its benefits.")
    ]

    # Set a very low max_length to force truncation in the output.
    max_length = 5
    payload = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in conversation],
        "max_length": max_length,
        "temperature": 0.7,
    }

    response = client_unit.post("/v1/summarization", json=payload)
    assert response.status_code == 200, "Expected status code 200."

    data = response.json()
    summary = data["summary"].strip()
    assert summary, "Summary should not be empty."

    # Validate that the summary token count does not exceed max_length.
    token_count = len(summary.split())
    assert token_count <= max_length, f"Summary token count ({token_count}) exceeds max_length ({max_length})."
