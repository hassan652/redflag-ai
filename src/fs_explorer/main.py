"""
CLI entry point for the FsExplorer agent.

Provides a command-line interface for running filesystem exploration tasks
with rich, detailed output showing each step of the workflow.
"""

import json
import asyncio
from datetime import datetime

from typer import Typer, Option
from typing import Annotated
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .workflow import (
    workflow,
    InputEvent,
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

app = Typer()


# Tool icons for visual distinction
TOOL_ICONS = {
    "scan_folder": "📂",
    "preview_file": "👁️",
    "parse_file": "📖",
    "read": "📄",
    "grep": "🔍",
    "glob": "🔎",
}

# Phase detection based on tool usage
PHASE_DESCRIPTIONS = {
    "scan_folder": ("Phase 1", "Parallel Document Scan", "cyan"),
    "preview_file": ("Phase 1/2", "Quick Preview", "cyan"),
    "parse_file": ("Phase 2", "Deep Dive", "green"),
    "read": ("Reading", "Text File", "blue"),
    "grep": ("Searching", "Pattern Match", "yellow"),
    "glob": ("Finding", "File Search", "yellow"),
}


def format_tool_panel(event: ToolCallEvent, step_number: int) -> Panel:
    """Create a richly formatted panel for a tool call event."""
    tool_name = event.tool_name
    icon = TOOL_ICONS.get(tool_name, "🔧")
    phase_info = PHASE_DESCRIPTIONS.get(tool_name, ("Action", "Tool Call", "yellow"))
    phase_label, phase_desc, color = phase_info
    
    # Build the content
    lines = []
    
    # Tool and target info
    if "directory" in event.tool_input:
        target = event.tool_input["directory"]
        lines.append(f"**Target Directory:** `{target}`")
    elif "file_path" in event.tool_input:
        target = event.tool_input["file_path"]
        lines.append(f"**Target File:** `{target}`")
    
    # Additional parameters
    other_params = {k: v for k, v in event.tool_input.items() 
                    if k not in ("directory", "file_path")}
    if other_params:
        lines.append(f"**Parameters:** `{json.dumps(other_params)}`")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Reasoning (this is the key part for visibility)
    lines.append("**Agent's Reasoning:**")
    lines.append("")
    lines.append(event.reason)
    
    content = "\n".join(lines)
    
    # Create title with step number and phase
    title = f"{icon} Step {step_number}: {tool_name} [{phase_label}: {phase_desc}]"
    
    return Panel(
        Markdown(content),
        title=title,
        title_align="left",
        border_style=f"bold {color}",
        padding=(1, 2),
    )


def format_navigation_panel(event: GoDeeperEvent, step_number: int) -> Panel:
    """Create a panel for directory navigation events."""
    content = f"""**Navigating to:** `{event.directory}`

---

**Agent's Reasoning:**

{event.reason}
"""
    return Panel(
        Markdown(content),
        title=f"📁 Step {step_number}: Navigate to Directory",
        title_align="left",
        border_style="bold magenta",
        padding=(1, 2),
    )


def print_workflow_header(console: Console, task: str) -> None:
    """Print a header showing the task being executed."""
    console.print()
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", justify="right")
    header.add_column()
    
    # Show provider info
    from .workflow import _AGENT_PROVIDER, _AGENT_MODEL
    provider_display = _AGENT_PROVIDER.capitalize()
    if _AGENT_MODEL:
        provider_display += f" ({_AGENT_MODEL})"
    
    header.add_row("🤖 FsExplorer Agent", "")
    header.add_row("🧠 Provider:", provider_display)
    header.add_row("📋 Task:", task)
    header.add_row("🕐 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(Panel(header, border_style="bold blue", title="Starting Exploration", title_align="left"))
    console.print()


def print_workflow_summary(console: Console, agent, step_count: int) -> None:
    """Print a summary of the workflow execution."""
    usage = agent.token_usage
    
    # Create summary table
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", justify="right")
    summary.add_column()
    
    summary.add_row("Provider:", usage.provider_name or agent.provider_name)
    summary.add_row("Model:", usage.model_name or "default")
    summary.add_row("Total Steps:", str(step_count))
    summary.add_row("API Calls:", str(usage.api_calls))
    summary.add_row("Documents Scanned:", str(usage.documents_scanned))
    summary.add_row("Documents Parsed:", str(usage.documents_parsed))
    summary.add_row("", "")
    summary.add_row("Prompt Tokens:", f"{usage.prompt_tokens:,}")
    summary.add_row("Completion Tokens:", f"{usage.completion_tokens:,}")
    summary.add_row("Total Tokens:", f"{usage.total_tokens:,}")
    summary.add_row("", "")
    
    # Cost calculation
    input_cost, output_cost, total_cost = usage._calculate_cost()
    summary.add_row("Est. Input Cost:", f"${input_cost:.4f}")
    summary.add_row("Est. Output Cost:", f"${output_cost:.4f}")
    summary.add_row("Est. Total Cost:", f"${total_cost:.4f}")
    
    console.print()
    console.print(Panel(
        summary,
        title="📊 Workflow Summary",
        title_align="left",
        border_style="bold blue",
    ))


async def run_workflow(task: str) -> None:
    """
    Execute the exploration workflow with detailed step-by-step output.
    
    Args:
        task: The user's task/question to answer.
    """
    console = Console()
    
    # Reset agent for fresh state
    reset_agent()
    
    # Print header
    print_workflow_header(console, task)
    
    step_number = 0
    handler = workflow.run(start_event=InputEvent(task=task))
    
    with console.status(status="[bold cyan]🔄 Analyzing task...") as status:
        async for event in handler.stream_events():
            if isinstance(event, IngestEvent):
                step_number += 1
                ingest_panel = Panel(
                    Markdown(
                        f"**{event.message}**\n\n"
                        f"Documents found: {event.documents_found}\n\n"
                        f"---\n\n{event.hierarchy_summary[:800]}"
                    ),
                    title=f"📥 Step {step_number}: Document Ingestion & Analysis",
                    title_align="left",
                    border_style="bold blue",
                    padding=(1, 2),
                )
                console.print(ingest_panel)
                console.print()
                status.update("[bold cyan]🔄 Agent is analyzing documents...")
            
            elif isinstance(event, RawAnswerEvent):
                status.update("[bold yellow]🔍 Verifying citations against source documents...")
            
            elif isinstance(event, ToolCallEvent):
                step_number += 1
                
                # Update status based on tool
                icon = TOOL_ICONS.get(event.tool_name, "🔧")
                if event.tool_name == "scan_folder":
                    status.update(f"[bold cyan]{icon} Scanning documents in parallel...")
                elif event.tool_name == "parse_file":
                    status.update(f"[bold green]{icon} Reading document in detail...")
                elif event.tool_name == "preview_file":
                    status.update(f"[bold cyan]{icon} Quick preview of document...")
                else:
                    status.update(f"[bold yellow]{icon} Executing {event.tool_name}...")
                
                # Print the detailed panel
                panel = format_tool_panel(event, step_number)
                console.print(panel)
                console.print()
                
                status.update("[bold cyan]🔄 Processing results...")
                
            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                panel = format_navigation_panel(event, step_number)
                console.print(panel)
                console.print()
                status.update("[bold cyan]🔄 Exploring directory...")
                
            elif isinstance(event, AskHumanEvent):
                status.stop()
                console.print()
                
                # Create a nice prompt panel
                question_panel = Panel(
                    Markdown(f"**Question:** {event.question}\n\n**Why I'm asking:** {event.reason}"),
                    title="❓ Human Input Required",
                    title_align="left",
                    border_style="bold red",
                )
                console.print(question_panel)
                
                answer = console.input("[bold cyan]Your answer:[/] ")
                while answer.strip() == "":
                    console.print("[bold red]Please provide an answer.[/]")
                    answer = console.input("[bold cyan]Your answer:[/] ")
                
                handler.ctx.send_event(HumanAnswerEvent(response=answer.strip()))
                console.print()
                status.start()
                status.update("[bold cyan]🔄 Processing your response...")
        
        # Get final result
        result = await handler
        status.update("[bold green]✨ Preparing final answer...")
        await asyncio.sleep(0.1)
        status.stop()
    
    # Print final result with prominent styling
    console.print()
    if result.final_result:
        final_panel = Panel(
            Markdown(result.final_result),
            title="✅ Final Answer",
            title_align="left",
            border_style="bold green",
            padding=(1, 2),
        )
        console.print(final_panel)
    elif result.error:
        error_panel = Panel(
            Text(result.error, style="bold red"),
            title="❌ Error",
            title_align="left",
            border_style="bold red",
        )
        console.print(error_panel)
    
    # Print workflow summary
    agent = get_agent()
    print_workflow_summary(console, agent, step_number)


@app.command()
def main(
    task: Annotated[
        str,
        Option(
            "--task",
            "-t",
            help="Task that the FsExplorer Agent has to perform while exploring the current directory.",
        ),
    ],
    provider: Annotated[
        str,
        Option(
            "--provider",
            "-p",
            help="LLM provider to use: 'gemini' or 'groq'.",
        ),
    ] = "groq",
    model: Annotated[
        str | None,
        Option(
            "--model",
            "-m",
            help="Model name override (e.g. 'llama-3.3-70b-versatile' for Groq).",
        ),
    ] = None,
) -> None:
    """
    Explore the filesystem to answer questions about documents.
    
    The agent will scan, analyze, and parse relevant documents to provide
    comprehensive answers with source citations.
    
    Supports multiple LLM providers:
      --provider gemini  (default, requires GOOGLE_API_KEY)
      --provider groq    (requires GROQ_API_KEY)
    """
    set_provider(provider, model)
    asyncio.run(run_workflow(task))
