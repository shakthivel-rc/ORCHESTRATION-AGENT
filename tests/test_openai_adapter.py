from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from switchboard import ClientCapabilities, LLMRequest, PromptSegment
from switchboard.engine.schema import build_wire_schema
from switchboard.providers import SHIPPED_ADAPTERS, resolve_client
from switchboard.providers.openai_adapter import OpenAIAdapter


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message
    finish_reason: str = "stop"
    logprobs: Any = None


class _Usage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 4


class _Response(BaseModel):
    id: str = "resp_1"
    model: str = "gpt-5-nano"
    choices: list[_Choice]
    usage: _Usage = Field(default_factory=_Usage)


class _Completions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _Response:
        self.kwargs = kwargs
        return _Response(
            choices=[
                _Choice(
                    message=_Message(
                        content='{"rationale":"r","kind":"route","route":"refund","args":null}'
                    )
                )
            ]
        )


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _Client:
    def __init__(self) -> None:
        self.chat = _Chat()


def test_openai_adapter_is_shipped() -> None:
    assert "openai" in SHIPPED_ADAPTERS


def test_openai_adapter_maps_request_to_openai_chat_completion() -> None:
    client = _Client()
    adapter = OpenAIAdapter(
        "gpt-5-nano",
        client=client,
        aclient=object(),
        capabilities=ClientCapabilities(structured="grammar", logprobs=True, caching="implicit"),
    )
    wire = build_wire_schema(["refund"], mode="dynamic")
    request = LLMRequest(
        segments=(PromptSegment(role="user", content="refund please"),),
        output_schema=wire,
        want_logprobs=True,
    )

    result = adapter.complete(request)

    assert result.parsed is not None
    assert result.raw_text.startswith('{"rationale"')
    assert result.usage.input_tokens == 10
    assert client.chat.completions.kwargs is not None
    assert client.chat.completions.kwargs["response_format"]["type"] == "json_schema"
    assert client.chat.completions.kwargs["logprobs"] is True


def test_resolve_client_builds_openai_adapter_with_injected_client() -> None:
    client = _Client()
    resolved = resolve_client(
        "openai:gpt-5-nano",
        client=client,
        aclient=object(),
        capabilities=ClientCapabilities(structured="grammar"),
    )

    assert isinstance(resolved, OpenAIAdapter)
