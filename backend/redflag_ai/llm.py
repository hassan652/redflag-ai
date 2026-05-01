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

import re


def _parse_to_action(raw_text: str) -> Action | None:
    """Clean LLM output and attempt to parse it as an Action.

    Handles:
    - ``<think>...</think>`` blocks from reasoning/thinking models
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    """
    text = raw_text.strip()

    # Strip <think>...</think> blocks (qwen3 and other thinking models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Direct parse
    try:
        return Action.model_validate_json(text)
    except Exception:
        pass

    # Fallback: extract the outermost JSON object
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return Action.model_validate_json(text[start:end])
    except Exception:
        return None


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


DIRECT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question directly and clearly. "
    "Use plain language unless the user asks for something more technical.\n\n"
    "When documents are provided, you MUST cite your sources for every factual claim "
    "using this inline format: [Source: filename, Section/Page reference]\n\n"
    "Example:\n"
    "> The purchase price is $125 million [Source: master_agreement.pdf, Section 2.1], "
    "payable at closing [Source: master_agreement.pdf, Section 3.1].\n\n"
    "At the end of your answer, include a '## Sources Consulted' section listing every "
    "document you referenced. If a document does not contain information relevant to the "
    "question, explicitly say so rather than guessing."
)


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
   MANDATORY: final_result MUST contain [Source: filename, Section] inline citations for EVERY factual claim. Answers without citations will be rejected.

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
            self._chat_history.append({"role": "assistant", "content": raw_text})
            return _parse_to_action(raw_text)

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

# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (GPT-4o, o1, etc.) using the openai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-5.4-mini",
        base_url: str | None = None,
        env_var_name: str = "OPENAI_API_KEY",
        provider_label: str = "OpenAI",
    ) -> None:
        from openai import AsyncOpenAI

        if api_key is None:
            api_key = os.getenv(env_var_name)
        if api_key is None:
            raise ValueError(f"{env_var_name} not found: export it or pass api_key= to the constructor.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._chat_history: list[dict[str, str]] = []
        self.token_usage = TokenUsage(provider_name=provider_label, model_name=model)

    def add_message(self, role: str, content: str) -> None:
        if role == "model":
            role = "assistant"
        self._chat_history.append({"role": role, "content": content})

    async def get_structured_action(self, system_prompt: str) -> Action | None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt + "\n\n" + ACTION_JSON_SCHEMA},
            *self._chat_history,
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=4096,
        )
        if response.usage:
            self.token_usage.add_api_call(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            raw_text = choice.message.content.strip()
            self._chat_history.append({"role": "assistant", "content": raw_text})
            return _parse_to_action(raw_text)
        return None

    def get_raw_history(self) -> list[dict[str, str]]:
        return list(self._chat_history)

    def reset(self) -> None:
        self._chat_history.clear()
        provider_name = self.token_usage.provider_name or "OpenAI"
        self.token_usage = TokenUsage(provider_name=provider_name, model_name=self._model)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider using the OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-5.4-mini") -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            env_var_name="OPENROUTER_API_KEY",
            provider_label="OpenRouter",
        )


# =============================================================================
# Claude (Anthropic) Provider
# =============================================================================

class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider (claude-3-5-haiku, claude-3-7-sonnet, etc.)."""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022") -> None:
        from anthropic import AsyncAnthropic

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("ANTHROPIC_API_KEY not found: export it or pass api_key= to the constructor.")

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._chat_history: list[dict[str, str]] = []
        self.token_usage = TokenUsage(provider_name="Claude", model_name=model)

    def add_message(self, role: str, content: str) -> None:
        if role == "model":
            role = "assistant"
        self._chat_history.append({"role": role, "content": content})

    async def get_structured_action(self, system_prompt: str) -> Action | None:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt + "\n\n" + ACTION_JSON_SCHEMA,
            messages=self._chat_history,  # type: ignore[arg-type]
        )
        self.token_usage.add_api_call(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
        if response.content:
            raw_text = response.content[0].text.strip()  # type: ignore[attr-defined]
            self._chat_history.append({"role": "assistant", "content": raw_text})
            json_text = raw_text
            if json_text.startswith("```"):
                lines = [l for l in json_text.split("\n") if not l.strip().startswith("```")]
                json_text = "\n".join(lines)
            try:
                return Action.model_validate_json(json_text)
            except Exception:
                try:
                    start = json_text.index("{")
                    end = json_text.rindex("}") + 1
                    return Action.model_validate_json(json_text[start:end])
                except Exception:
                    return None
        return None

    def get_raw_history(self) -> list[dict[str, str]]:
        return list(self._chat_history)

    def reset(self) -> None:
        self._chat_history.clear()
        self.token_usage = TokenUsage(provider_name="Claude", model_name=self._model)


# =============================================================================
# Factory
# =============================================================================

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3-flash-preview",
    "groq": "qwen/qwen3-32b",
    "openai": "gpt-5.4-mini",
    "openrouter": "openai/gpt-5.4-mini",
    "claude": "claude-3-5-haiku-20241022",
}


def create_provider(
    provider: str = "openrouter",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance."""
    provider = provider.lower().strip()

    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model=model or DEFAULT_MODELS["gemini"])
    elif provider == "groq":
        return GroqProvider(api_key=api_key, model=model or DEFAULT_MODELS["groq"])
    elif provider == "openrouter":
        return OpenRouterProvider(api_key=api_key, model=model or DEFAULT_MODELS["openrouter"])
    elif provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model or DEFAULT_MODELS["openai"])
    elif provider == "claude":
        return ClaudeProvider(api_key=api_key, model=model or DEFAULT_MODELS["claude"])
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: openrouter, groq, openai, claude, gemini"
        )


async def generate_text(
    prompt: str,
    provider: str = "openrouter",
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str = DIRECT_CHAT_SYSTEM_PROMPT,
) -> tuple[str, TokenUsage]:
    """Generate a direct text response without the filesystem workflow."""
    provider_name = provider.lower().strip()
    resolved_model = model or DEFAULT_MODELS.get(provider_name, "")

    if provider_name == "gemini":
        p = GeminiProvider(api_key=api_key, model=resolved_model)
        response = await p._client.aio.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config={"system_instruction": system_prompt},
        )
        if response.usage_metadata:
            p.token_usage.add_api_call(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )
        return response.text or "", p.token_usage

    if provider_name == "groq":
        p = GroqProvider(api_key=api_key, model=resolved_model)
        response = await p._client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        if response.usage:
            p.token_usage.add_api_call(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message and choice.message.content else ""
        return text, p.token_usage

    if provider_name == "openrouter":
        p = OpenRouterProvider(api_key=api_key, model=resolved_model)
        response = await p._client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=4096,
        )
        if response.usage:
            p.token_usage.add_api_call(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message and choice.message.content else ""
        return text, p.token_usage

    if provider_name == "openai":
        p = OpenAIProvider(api_key=api_key, model=resolved_model)
        response = await p._client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=4096,
        )
        if response.usage:
            p.token_usage.add_api_call(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        choice = response.choices[0] if response.choices else None
        text = choice.message.content if choice and choice.message and choice.message.content else ""
        return text, p.token_usage

    if provider_name == "claude":
        p = ClaudeProvider(api_key=api_key, model=resolved_model)
        response = await p._client.messages.create(
            model=resolved_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        p.token_usage.add_api_call(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
        text = response.content[0].text if response.content else ""  # type: ignore[attr-defined]
        return text, p.token_usage

    raise ValueError(f"Unknown LLM provider: '{provider_name}'. Supported: openrouter, groq, openai, claude, gemini")
