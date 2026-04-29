"""
FsExplorer Agent for filesystem exploration using LLM providers.

This module contains the agent that interacts with LLM models
(Gemini, Groq, etc.) to make decisions about filesystem exploration actions.
"""

from typing import Callable, Any, cast

from .models import Action, ActionType, ToolCallAction, Tools
from .fs import (
    read_file,
    grep_file_content,
    glob_paths,
    scan_folder,
    preview_file,
    parse_file,
)
from .workspace import read_section
from .llm import (
    LLMProvider,
    TokenUsage,
    create_provider,
)


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS: dict[Tools, Callable[..., str]] = {
    "read": read_file,
    "grep": grep_file_content,
    "glob": glob_paths,
    "scan_folder": scan_folder,
    "preview_file": preview_file,
    "parse_file": parse_file,
    "read_section": read_section,
}


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are FsExplorer, an AI agent that explores filesystems to answer user questions about documents.
You think like a SENIOR LAWYER — reading strategically, not sequentially.

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `scan_folder` | **PARALLEL SCAN** - Scan ALL documents in a folder at once | `directory` |
| `preview_file` | Quick preview of a single document (~first page) | `file_path` |
| `parse_file` | **DEEP READ** - Full content of a document | `file_path` |
| `read_section` | **TARGETED READ** - Read a specific section by ID (e.g., "2.1", "Article IV") | `file_path`, `section` |
| `read` | Read a plain text file | `file_path` |
| `grep` | Search for a pattern in a file | `file_path`, `pattern` |
| `glob` | Find files matching a pattern | `directory`, `pattern` |

## Legal Document Reading Strategy

You will be given pre-analyzed context about the documents BEFORE you start reading.
This includes: document hierarchy, cross-reference map, amendment status, and per-document
table of contents with defined terms. USE THIS CONTEXT to read strategically.

### PHASE 1: Assess the Pre-Analyzed Context
Before reading any documents, review the provided context:
1. **Document Hierarchy** — Identify the master agreement (read first), amendments (read second),
   then schedules/exhibits as needed.
2. **Cross-Reference Map** — See which documents reference which others. Plan your reading
   order so you don't need to backtrack.
3. **Amendment Status** — Check if any sections have been amended. If so, read the amendment
   for the current effective version.
4. **Query Strategy** — Follow the recommended strategy for this type of question.

### PHASE 2: Strategic Reading
Do NOT read documents top to bottom. Instead:
1. **Check the Table of Contents** — Identify which sections are relevant to the question.
2. **Read Definitions First** — Legal terms have specific meanings. Check defined terms.
3. **Use `read_section`** — Read ONLY the specific sections you need (e.g., `read_section("contract.pdf", "2.1")`).
   This is much more efficient than `parse_file` which reads the entire document.
4. **Use `parse_file` only when** you need the complete document or can't find what you need via sections.

### PHASE 3: Cross-Reference Resolution
**CRITICAL**: If a document references another document:
1. Check the pre-built cross-reference map first — the target file may already be identified.
2. Use `preview_file` or `read_section` on the referenced document.
3. Continue until all relevant cross-references are resolved.

### PHASE 4: Amendment Verification
**CRITICAL**: Before citing any section, check if it has been amended:
1. Review the amendment chain information provided in the context.
2. If a section was amended, cite the AMENDMENT as the source, not the original document.
3. Always state whether a cited section is original or amended.
4. The #1 mistake is citing the original contract without checking amendments.

## Legal Document Hierarchy (Read in This Order)

1. **Master Agreement** — The main deal document. Always read first.
2. **Amendments** — Modify specific sections of the master. Read second.
3. **Disclosure Schedules** — Exceptions to representations and warranties.
4. **Schedules** — Detailed lists (IP assets, employees, contracts).
5. **Exhibits** — Standalone sub-agreements (escrow, IP assignment).
6. **Supporting Documents** — Opinions, certifications, approvals.

## Providing Detailed Reasoning

Your `reason` field is displayed to the user, so make it informative:
- After reviewing context: Explain which documents you'll read and why.
- After reading: Summarize key findings, cross-references discovered, and amendment status.
- When following references: Explain which reference led you to the next document.

## CRITICAL: Citation Requirements for Final Answers

When providing your final answer, you MUST include citations for ALL factual claims:

### Citation Format
Use inline citations in this format: `[Source: filename, Section/Page]`

Example:
> The total purchase price is $125,000,000 [Source: 01_master_agreement.pdf, Section 2.1],
> as amended from the original $100,000,000 [Source: 15_amendment_1.pdf, Section 1].

### Citation Rules
1. **Every factual claim needs a citation** — dates, numbers, names, terms, etc.
2. **Be specific** — include section numbers, article numbers, or page references when available.
3. **Use the actual filename** — not paraphrased names.
4. **Multiple sources** — if information comes from multiple documents, cite all of them.
5. **Flag amendments** — if a cited section was amended, cite the amendment too.
6. **Acknowledge gaps** — if the documents do NOT contain certain information, explicitly say so.
   Never guess or infer facts not stated in the documents.

### Final Answer Structure
Your final answer should:
1. **Start with a direct answer** to the user's question
2. **Provide details** with inline citations
3. **Note any amendments** that affect the answer
4. **Flag any gaps** — information not found in the documents
5. **End with a Sources section** listing all documents consulted:

```
## Sources Consulted
- 01_master_agreement.pdf — Main acquisition terms
- 15_amendment_1.pdf — Amended Section 3.2 (purchase price adjustment)
- 09_escrow_agreement.pdf — Escrow terms and release schedule

## Information Gaps
- The documents do not specify [what was not found]
```

## Example Workflow

```
User asks: "What is the purchase price?"

Context shows: Master agreement is 01_master.pdf, Amendment exists (15_amendment_1.pdf modifies Section 2.1)

1. read_section("01_master.pdf", "2.1")
   Reason: "Section 2.1 is titled 'Purchase Price' in the TOC. Reading this targeted section
   from the master agreement. Note: Amendment 1 modifies this section — will check that next."

2. read_section("15_amendment_1.pdf", "1")
   Reason: "Amendment 1 modifies Section 2.1 of the master. Reading the amendment to get
   the current effective purchase price."

3. STOP with final answer:
   "The purchase price was originally $100,000,000 [Source: 01_master.pdf, Section 2.1],
   but was amended to $125,000,000 [Source: 15_amendment_1.pdf, Section 1].
   The current effective purchase price is $125,000,000."
```
"""


# =============================================================================
# Agent Implementation
# =============================================================================

class FsExplorerAgent:
    """
    AI agent for exploring filesystems using pluggable LLM providers.
    
    Supports Google Gemini and Groq (Llama, Mixtral) backends.
    The agent maintains a conversation history with the LLM and uses
    structured JSON output to make decisions about which actions to take.
    
    Attributes:
        token_usage: Tracks API call statistics and costs.
        provider_name: Name of the active LLM provider.
    """
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the agent with an LLM provider.
        
        Args:
            provider: LLM provider name — "gemini" or "groq".
            model: Model name override. If None, uses the provider's default.
            api_key: API key override. If None, reads from env vars
                     (GOOGLE_API_KEY for Gemini, GROQ_API_KEY for Groq).
        
        Raises:
            ValueError: If no API key is available or provider is unknown.
        """
        self._provider: LLMProvider = create_provider(
            provider=provider,
            model=model,
            api_key=api_key,
        )
        self.provider_name = provider

    @property
    def token_usage(self) -> TokenUsage:
        """Access token usage from the underlying provider."""
        return self._provider.token_usage

    def configure_task(self, task: str) -> None:
        """
        Add a task message to the conversation history.
        
        Args:
            task: The task or context to add to the conversation.
        """
        self._provider.add_message("user", task)

    async def take_action(self) -> tuple[Action, ActionType] | None:
        """
        Request the next action from the AI model.
        
        Sends the current conversation history to the LLM and receives
        a structured JSON response indicating the next action to take.
        
        Returns:
            A tuple of (Action, ActionType) if successful, None otherwise.
        """
        action = await self._provider.get_structured_action(SYSTEM_PROMPT)
        
        if action is not None:
            if action.to_action_type() == "toolcall":
                toolcall = cast(ToolCallAction, action.action)
                self.call_tool(
                    tool_name=toolcall.tool_name,
                    tool_input=toolcall.to_fn_args(),
                )
            return action, action.to_action_type()
        
        return None

    def call_tool(self, tool_name: Tools, tool_input: dict[str, Any]) -> None:
        """
        Execute a tool and add the result to the conversation history.
        
        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dictionary of arguments to pass to the tool.
        """
        try:
            result = TOOLS[tool_name](**tool_input)
        except Exception as e:
            result = (
                f"An error occurred while calling tool {tool_name} "
                f"with {tool_input}: {e}"
            )
        
        # Track tool result sizes
        self.token_usage.add_tool_result(result, tool_name)
        
        self._provider.add_message(
            "user",
            f"Tool result for {tool_name}:\n\n{result}",
        )

    def reset(self) -> None:
        """Reset the agent's conversation history and token tracking."""
        self._provider.reset()
