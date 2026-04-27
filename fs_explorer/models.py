"""
Pydantic models for FsExplorer agent actions.

This module defines the structured data models used to represent
the actions the agent can take during filesystem exploration.
"""

from pydantic import BaseModel, Field
from typing import TypeAlias, Literal, Any


# =============================================================================
# Type Aliases
# =============================================================================

Tools: TypeAlias = Literal[
    "read", "grep", "glob", "scan_folder", "preview_file", "parse_file", "read_section"
]
"""Available tool names that the agent can invoke."""

ActionType: TypeAlias = Literal["stop", "godeeper", "toolcall", "askhuman"]
"""Types of actions the agent can take."""


# =============================================================================
# Action Models
# =============================================================================

class StopAction(BaseModel):
    """
    Action indicating the task is complete.
    
    Used when the agent has gathered enough information to provide
    a final answer to the user's query.
    """
    
    final_result: str = Field(
        description="Final result of the operation with the answer to the user's query"
    )


class AskHumanAction(BaseModel):
    """
    Action requesting clarification from the user.
    
    Used when the agent needs additional information or context
    to proceed with the task.
    """
    
    question: str = Field(
        description="Clarification question to ask the user"
    )


class GoDeeperAction(BaseModel):
    """
    Action to navigate into a subdirectory.
    
    Used when the agent needs to explore a subdirectory
    to find relevant files.
    """
    
    directory: str = Field(
        description="Path to the directory to navigate into"
    )


class ToolCallArg(BaseModel):
    """
    A single argument for a tool call.
    
    Represents a parameter name-value pair to pass to a tool.
    """
    
    parameter_name: str = Field(
        description="Name of the parameter"
    )
    parameter_value: Any = Field(
        description="Value for the parameter"
    )


class ToolCallAction(BaseModel):
    """
    Action to invoke a filesystem tool.
    
    Used when the agent needs to read files, search for patterns,
    or parse documents to gather information.
    """
    
    tool_name: Tools = Field(
        description="Name of the tool to invoke"
    )
    tool_input: list[ToolCallArg] = Field(
        description="Arguments to pass to the tool"
    )

    def to_fn_args(self) -> dict[str, Any]:
        """
        Convert tool input to a dictionary for function calls.
        
        Returns:
            Dictionary mapping parameter names to values.
        """
        return {arg.parameter_name: arg.parameter_value for arg in self.tool_input}


class Action(BaseModel):
    """
    Container for an agent action with reasoning.
    
    Wraps any of the specific action types (stop, go deeper,
    tool call, ask human) along with the agent's explanation
    for why this action was chosen.
    """
    
    action: ToolCallAction | GoDeeperAction | StopAction | AskHumanAction = Field(
        description="The specific action to take"
    )
    reason: str = Field(
        description="Explanation for why this action was chosen"
    )

    def to_action_type(self) -> ActionType:
        """
        Get the type of this action.
        
        Returns:
            The action type string: "toolcall", "godeeper", "askhuman", or "stop".
        """
        if isinstance(self.action, ToolCallAction):
            return "toolcall"
        elif isinstance(self.action, GoDeeperAction):
            return "godeeper"
        elif isinstance(self.action, AskHumanAction):
            return "askhuman"
        else:
            return "stop"


# =============================================================================
# Structured Output Models (for legal-grade answers)
# =============================================================================


class Citation(BaseModel):
    """
    A verified citation linking a claim to a source document.
    
    Every factual claim in the answer should have a citation
    with the exact file, section, and quoted text.
    """
    
    claim: str = Field(
        description="The factual claim being made"
    )
    source_file: str = Field(
        description="Filename of the source document"
    )
    source_section: str = Field(
        default="",
        description="Section/article/exhibit reference within the document"
    )
    source_quote: str = Field(
        default="",
        description="Exact quoted text from the source that supports the claim"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    is_amended: bool = Field(
        default=False,
        description="Whether this section was modified by an amendment"
    )
    amendment_source: str | None = Field(
        default=None,
        description="If amended, which amendment document modified this section"
    )


class RiskItem(BaseModel):
    """
    A single risk identified during document analysis.
    
    Used in risk analysis output format to present findings
    as a structured risk matrix.
    """
    
    risk: str = Field(
        description="Description of the risk"
    )
    severity: str = Field(
        description="Risk severity: High, Medium, or Low"
    )
    category: str = Field(
        default="General",
        description="Risk category (e.g., Financial, Legal, Operational, Regulatory)"
    )
    source: str = Field(
        default="",
        description="Source document and section"
    )
    mitigation: str | None = Field(
        default=None,
        description="Any mitigation measures mentioned in the documents"
    )


class KeyTerm(BaseModel):
    """
    A key term or provision extracted from documents.
    
    Used for structured term sheets and deal summaries.
    """
    
    term: str = Field(
        description="Name of the term (e.g., Purchase Price, Closing Date)"
    )
    value: str = Field(
        description="The value or description of the term"
    )
    conditions: str | None = Field(
        default=None,
        description="Any conditions or qualifications on this term"
    )
    source_file: str = Field(
        default="",
        description="Source document filename"
    )
    source_section: str = Field(
        default="",
        description="Section reference within the document"
    )


class StructuredAnswer(BaseModel):
    """
    A structured legal analysis response.
    
    Provides a professional-grade deliverable that lawyers can
    use directly in memos, reports, and client communications.
    """
    
    summary: str = Field(
        description="2-3 sentence executive summary answering the question"
    )
    key_findings: list[Citation] = Field(
        default_factory=list,
        description="Main factual findings with full citations"
    )
    risks: list[RiskItem] = Field(
        default_factory=list,
        description="Identified risks with severity ratings"
    )
    key_terms: list[KeyTerm] = Field(
        default_factory=list,
        description="Extracted key terms and provisions"
    )
    timeline: list[dict] = Field(
        default_factory=list,
        description="Chronological events with dates and sources"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps — what the documents do NOT specify"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended next steps or areas requiring further review"
    )
