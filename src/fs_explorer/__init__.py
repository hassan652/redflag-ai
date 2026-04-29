"""
FsExplorer - AI-powered filesystem exploration agent.

This package provides an intelligent agent that can explore filesystems,
parse documents, and answer questions about their contents using
pluggable LLM providers (Google Gemini, Groq) and MarkItDown for document parsing.

Example usage:
    >>> from fs_explorer import FsExplorerAgent, workflow
    >>> agent = FsExplorerAgent(provider="groq")  # or "gemini"
    >>> # Use with the workflow for full exploration
    >>> result = await workflow.run(start_event=InputEvent(task="Find the purchase price"))
"""

from .agent import FsExplorerAgent
from .llm import (
    LLMProvider,
    GeminiProvider,
    GroqProvider,
    TokenUsage,
    create_provider,
)
from .workflow import (
    workflow,
    FsExplorerWorkflow,
    InputEvent,
    ExplorationEndEvent,
    ToolCallEvent,
    GoDeeperEvent,
    AskHumanEvent,
    HumanAnswerEvent,
    IngestEvent,
    RawAnswerEvent,
    get_agent,
    reset_agent,
    set_provider,
)
from .models import Action, ActionType, Tools, Citation, RiskItem, KeyTerm, StructuredAnswer
from .router import QueryType, classify_query, get_strategy
from .workspace import WorkspaceContext, ingest_folder, get_workspace_context
from .verifier import verify_answer

__all__ = [
    # Agent
    "FsExplorerAgent",
    # LLM Providers
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "TokenUsage",
    "create_provider",
    # Workflow
    "workflow",
    "FsExplorerWorkflow",
    "InputEvent",
    "ExplorationEndEvent",
    "ToolCallEvent",
    "GoDeeperEvent",
    "AskHumanEvent",
    "HumanAnswerEvent",
    "IngestEvent",
    "RawAnswerEvent",
    "get_agent",
    "reset_agent",
    "set_provider",
    # Models
    "Action",
    "ActionType",
    "Tools",
    "Citation",
    "RiskItem",
    "KeyTerm",
    "StructuredAnswer",
    # Router
    "QueryType",
    "classify_query",
    "get_strategy",
    # Workspace
    "WorkspaceContext",
    "ingest_folder",
    "get_workspace_context",
    # Verifier
    "verify_answer",
]

