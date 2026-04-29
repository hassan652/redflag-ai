"""
LLM provider abstraction layer.

Provides a unified interface for different LLM backends (Gemini, Groq)
so the agent can switch providers without changing any orchestration code.

Usage:
    provider = create_provider("groq")  # or "gemini"
    provider.add_message("user", "What action should I take?")
    action = await provider.get_structured_action(system_prompt)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import Action

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# =============================================================================
# Provider Pricing (per million tokens)
# =============================================================================

PRICING: dict[str, dict[str, float]] = {
    "gemini-3-flash-preview": {
        "input": 0.075,
        "output": 0.30,
    },
    # Groq models
    "llama-3.3-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    },
    "llama-3.1-8b-instant": {
        "input": 0.05,
        "output": 0.08,
    },
    "llama-3.1-70b-versatile": {
        "input": 0.59,
        "output": 0.79,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input": 0.11,
        "output": 0.34,
    },
    "compound-beta": {
        "input": 0.0,
        "output": 0.0,
    },
    "mixtral-8x7b-32768": {
        "input": 0.24,
        "output": 0.24,
    },
}


# =============================================================================
# Token Usage Tracking
# =============================================================================


@dataclass
class TokenUsage:
    """
    Track token usage and costs across the session.

    Provider-agnostic — works with any LLM backend.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    # Track content sizes
    tool_result_chars: int = 0
    documents_parsed: int = 0
    documents_scanned: int = 0

    # Provider info
    provider_name: str = ""
    model_name: str = ""

    def add_api_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage from an API call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.api_calls += 1

    def add_tool_result(self, result: str, tool_name: str) -> None:
        """Record metrics from a tool execution."""
        self.tool_result_chars += len(result)
        if tool_name == "parse_file":
            self.documents_parsed += 1
        elif tool_name == "scan_folder":
            self.documents_scanned += result.count("│ [")
        elif tool_name == "preview_file":
            self.documents_parsed += 1

    def _calculate_cost(self) -> tuple[float, float, float]:
        """Calculate estimated costs based on the model's pricing."""
        pricing = PRICING.get(self.model_name, {"input": 0.0, "output": 0.0})
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost, output_cost, input_cost + output_cost

    def summary(self) -> str:
        """Generate a formatted summary of token usage and costs."""
        input_cost, output_cost, total_cost = self._calculate_cost()

        return f"""
═══════════════════════════════════════════════════════════════
                      TOKEN USAGE SUMMARY
═══════════════════════════════════════════════════════════════
  Provider:            {self.provider_name}
  Model:               {self.model_name}
  API Calls:           {self.api_calls}
  Prompt Tokens:       {self.prompt_tokens:,}
  Completion Tokens:   {self.completion_tokens:,}
  Total Tokens:        {self.total_tokens:,}
───────────────────────────────────────────────────────────────
  Documents Scanned:   {self.documents_scanned}
  Documents Parsed:    {self.documents_parsed}
  Tool Result Chars:   {self.tool_result_chars:,}
───────────────────────────────────────────────────────────────
  Est. Cost ({self.model_name}):
    Input:  ${input_cost:.4f}
    Output: ${output_cost:.4f}
    Total:  ${total_cost:.4f}
═══════════════════════════════════════════════════════════════
"""


# =============================================================================
# Abstract Provider
# =============================================================================


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Each provider must handle:
    - Chat history management
    - Structured JSON output (returns an Action model)
    - Token usage tracking
    """

    token_usage: TokenUsage

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        ...

    @abstractmethod
    async def get_structured_action(self, system_prompt: str) -> Action | None:
        """
        Send the conversation to the LLM and get a structured Action response.

        Args:
            system_prompt: The system instruction for the LLM.

        Returns:
            A parsed Action model, or None if the response failed.
        """
        ...

    @abstractmethod
    def get_raw_history(self) -> list[dict[str, str]]:
        """Return the conversation history in a provider-agnostic format."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear the conversation history and reset token tracking."""
        ...


# =============================================================================
# Gemini Provider
# =============================================================================


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider.

    Uses the google-genai SDK with native structured JSON output
    via response_schema.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
    ) -> None:
        from google.genai import Client as GenAIClient
        from google.genai.types import HttpOptions

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY not found: export it or pass api_key= to the constructor."
            )

        self._client = GenAIClient(
            api_key=api_key,
            http_options=HttpOptions(api_version="v1beta"),
        )
        self._model = model
        self._chat_history: list[Any] = []  # list[Content]
        self.token_usage = TokenUsage(provider_name="Gemini", model_name=model)

    def add_message(self, role: str, content: str) -> None:
        from google.genai.types import Content, Part

        self._chat_history.append(
            Content(role=role, parts=[Part.from_text(text=content)])
        )

    async def get_structured_action(self, system_prompt: str) -> Action | None:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=self._chat_history,
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": Action,
            },
        )

        # Track tokens
        if response.usage_metadata:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        if response.candidates is not None:
            if response.candidates[0].content is not None:
                self._chat_history.append(response.candidates[0].content)
            if response.text is not None:
                return Action.model_validate_json(response.text)

        return None

    def get_raw_history(self) -> list[dict[str, str]]:
        history = []
        for content in self._chat_history:
            role = content.role or "user"
            text = ""
            if content.parts:
                text = content.parts[0].text or ""
            history.append({"role": role, "content": text})
        return history

    def reset(self) -> None:
        self._chat_history.clear()
        self.token_usage = TokenUsage(
            provider_name="Gemini", model_name=self._model
        )


# =============================================================================
# Groq Provider
# =============================================================================

# JSON schema for the Action model — injected into the system prompt for Groq
# since Groq doesn't support response_schema like Gemini does.
ACTION_JSON_SCHEMA = """
You MUST respond with ONLY a valid JSON object matching this exact schema:

{
  "action": <one of the action types below>,
  "reason": "<your reasoning for choosing this action>"
}

Action types (use exactly ONE):

1. Tool call:
   {"action": {"tool_name": "<tool>", "tool_input": [{"parameter_name": "<name>", "parameter_value": "<value>"}, ...]}, "reason": "..."}
   Valid tool_name values: "read", "grep", "glob", "scan_folder", "preview_file", "parse_file", "read_section"

2. Navigate into directory:
   {"action": {"directory": "<path>"}, "reason": "..."}

3. Ask human a question:
   {"action": {"question": "<your question>"}, "reason": "..."}

4. Stop with final answer:
   {"action": {"final_result": "<your complete answer with citations>"}, "reason": "..."}

CRITICAL: Return ONLY the JSON object. No markdown fences, no explanation outside the JSON.
"""


class GroqProvider(LLMProvider):
    """
    Groq LLM provider.

    Uses the Groq SDK (OpenAI-compatible) with JSON mode.
    The Action schema is embedded in the system prompt since
    Groq doesn't support native response_schema.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        from groq import AsyncGroq

        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError(
                "GROQ_API_KEY not found: export it or pass api_key= to the constructor."
            )

        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._chat_history: list[dict[str, str]] = []
        self.token_usage = TokenUsage(provider_name="Groq", model_name=model)

    def add_message(self, role: str, content: str) -> None:
        # Map Gemini's "model" role to OpenAI's "assistant" role
        if role == "model":
            role = "assistant"
        self._chat_history.append({"role": role, "content": content})

    async def get_structured_action(self, system_prompt: str) -> Action | None:
        # Build messages: system prompt + JSON schema + chat history
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt + "\n\n" + ACTION_JSON_SCHEMA,
            },
            *self._chat_history,
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096,
        )

        # Track tokens
        if response.usage:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

        # Extract and parse the response
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            raw_text = choice.message.content.strip()

            # Add assistant response to history
            self._chat_history.append({"role": "assistant", "content": raw_text})

            # Parse JSON — handle potential markdown fences
            json_text = raw_text
            if json_text.startswith("```"):
                # Strip markdown code fences
                lines = json_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                json_text = "\n".join(lines)

            try:
                return Action.model_validate_json(json_text)
            except Exception:
                # Try to extract JSON from the response
                try:
                    start = json_text.index("{")
                    end = json_text.rindex("}") + 1
                    return Action.model_validate_json(json_text[start:end])
                except (ValueError, Exception):
                    return None

        return None

    def get_raw_history(self) -> list[dict[str, str]]:
        return list(self._chat_history)

    def reset(self) -> None:
        self._chat_history.clear()
        self.token_usage = TokenUsage(
            provider_name="Groq", model_name=self._model
        )


# =============================================================================
# Factory
# =============================================================================

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "groq": "llama-3.3-70b-versatile",
}


def create_provider(
    provider: str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider name — "gemini" or "groq".
        model: Model name override. If None, uses the default for the provider.
        api_key: API key override. If None, reads from env vars.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        return GeminiProvider(
            api_key=api_key,
            model=model or DEFAULT_MODELS["gemini"],
        )
    elif provider == "groq":
        return GroqProvider(
            api_key=api_key,
            model=model or DEFAULT_MODELS["groq"],
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: gemini, groq"
        )
