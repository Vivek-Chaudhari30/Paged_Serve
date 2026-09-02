"""OpenAI-compatible request and response schemas.

Compatibility is not politeness, it is leverage. Every load generator, client
library, and eval harness already speaks this protocol, so matching it means
vLLM's own benchmark script can be pointed at this server — which is the most
credible comparison the project can produce, precisely because it is not one we
wrote.

The schemas accept the fields OpenAI defines and quietly ignore the ones this
engine does not implement, rather than rejecting them. A client that sends
``presence_penalty`` should get a completion, not a 422 about a knob it did not
know was unsupported. What is *not* implemented is documented here rather than
silently accepted and silently ignored — see ``UNSUPPORTED``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from pagedserve.model.sampler import SamplingParams

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "ErrorResponse",
    "UNSUPPORTED",
    "make_id",
]

# Accepted for compatibility and not acted on. Listed so the gap is a documented
# fact rather than a surprise discovered in someone's eval results.
UNSUPPORTED = ("presence_penalty", "frequency_penalty", "logit_bias", "logprobs")


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class SamplingFields(BaseModel):
    """The sampling knobs shared by both endpoints."""

    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False

    # Accepted, ignored. See UNSUPPORTED.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def stop_strings(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        if isinstance(self.stop, str):
            return (self.stop,)
        return tuple(self.stop)

    def to_sampling_params(self, stop_token_ids: tuple[int, ...] = ()) -> SamplingParams:
        """Translate to the engine's parameters.

        OpenAI's default temperature is 1.0 (sampling); the engine's is 0.0
        (greedy). The API default wins here, because a client that sends no
        temperature expects OpenAI's behaviour, not ours.
        """
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop_token_ids=stop_token_ids,
            stop_strings=self.stop_strings(),
            n=self.n,
            seed=self.seed,
        )


class CompletionRequest(SamplingFields):
    """``POST /v1/completions``."""

    model: str
    prompt: str | list[str]

    def prompts(self) -> list[str]:
        return [self.prompt] if isinstance(self.prompt, str) else list(self.prompt)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""


class ChatCompletionRequest(SamplingFields):
    """``POST /v1/chat/completions``."""

    model: str
    messages: list[ChatMessage]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None
    logprobs: None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: make_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatMessageDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage | None = None
    delta: ChatMessageDelta | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: make_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ErrorResponse(BaseModel):
    error: dict[str, Any]

    @classmethod
    def make(cls, message: str, kind: str = "invalid_request_error") -> ErrorResponse:
        return cls(error={"message": message, "type": kind})


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pagedserve"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
