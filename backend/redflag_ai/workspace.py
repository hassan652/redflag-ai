"""
Workspace context management and ingest pipeline.

Orchestrates the full document ingestion pipeline:
1. Parse all documents with MarkItDown (via fs.py)
2. Extract structure (TOC, definitions, sections) for each
3. Build document hierarchy (master → amendments → schedules → exhibits)
4. Extract and resolve cross-references between documents
5. Detect amendment chains and build effective clause versions

The resulting WorkspaceContext is stored and used by the agent
to make smarter decisions about what to read.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .document_structure import (
    DocumentStructure,
    parse_document_structure,
    format_structure_for_agent,
)
from .hierarchy import (
    DocumentHierarchy,
    build_hierarchy,
    format_hierarchy,
)
from .reference_map import (
    ReferenceMap,
    resolve_references,
    format_reference_map,
)
from .versioning import (
    AmendmentChain,
    detect_amendments,
    format_amendment_chains,
)
from .fs import SUPPORTED_EXTENSIONS


# =============================================================================
# Workspace Context
# =============================================================================


@dataclass
class WorkspaceContext:
    """
    Pre-processed workspace state, built at ingest time.

    Contains all the structural analysis that the agent needs to
    make smart reading decisions — hierarchy, cross-references,
    amendment chains, and per-document structure.
    """

    directory: str
    structures: dict[str, DocumentStructure] = field(
        default_factory=dict
    )  # file_path -> structure
    hierarchy: DocumentHierarchy | None = None
    reference_map: ReferenceMap | None = None
    amendment_chains: dict[str, AmendmentChain] = field(
        default_factory=dict
    )  # master_file -> chain


# =============================================================================
# Module-Level Context Storage
# =============================================================================

_WORKSPACE_CONTEXT: WorkspaceContext | None = None


def get_workspace_context() -> WorkspaceContext | None:
    """Get the current workspace context, if one has been ingested."""
    return _WORKSPACE_CONTEXT


def reset_workspace_context() -> None:
    """Reset the workspace context (useful for testing or new sessions)."""
    global _WORKSPACE_CONTEXT
    _WORKSPACE_CONTEXT = None


# =============================================================================
# Ingest Pipeline
# =============================================================================


def _parse_and_structure_file(file_path: str) -> tuple[str, DocumentStructure | None]:
    """
    Parse a single file and extract its structure.

    Called in parallel for all documents in a folder.

    Returns:
        Tuple of (file_path, DocumentStructure or None on error).
    """
    try:
        from .fs import get_parsed_content

        markdown_content = get_parsed_content(file_path)
        structure = parse_document_structure(file_path, markdown_content)
        return file_path, structure
    except Exception as e:
        print(f"  ⚠️  Failed to parse {os.path.basename(file_path)}: {e}")
        return file_path, None


def ingest_folder(
    directory: str,
    max_workers: int = 4,
) -> WorkspaceContext:
    """
    Run the full ingest pipeline on a folder of documents.

    This is called ONCE before the agent starts exploring.
    It pre-processes all documents so the agent has enriched context.

    Pipeline:
    1. Find all supported documents
    2. Parse each with MarkItDown (parallel, with caching)
    3. Extract structure (TOC, definitions, sections, cross-refs)
    4. Build document hierarchy
    5. Resolve cross-references to actual files
    6. Detect amendment chains

    Args:
        directory: Path to the folder to ingest.
        max_workers: Number of parallel workers for parsing.

    Returns:
        A WorkspaceContext with all pre-processed data.
    """
    global _WORKSPACE_CONTEXT

    abs_directory = os.path.abspath(directory)

    # Step 1: Find all supported document files
    doc_files: list[str] = []
    for item in os.listdir(abs_directory):
        item_path = os.path.join(abs_directory, item)
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                doc_files.append(item_path)

    if not doc_files:
        # No documents found — return empty context
        ctx = WorkspaceContext(directory=abs_directory)
        _WORKSPACE_CONTEXT = ctx
        return ctx

    # Step 2 & 3: Parse and extract structure (parallel)
    structures: dict[str, DocumentStructure] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_parse_and_structure_file, f): f for f in doc_files
        }
        for future in as_completed(futures):
            file_path, structure = future.result()
            if structure is not None:
                structures[file_path] = structure

    # Step 4: Build document hierarchy
    doc_hierarchy = build_hierarchy(structures)

    # Step 5: Resolve cross-references
    all_refs: dict[str, list[dict]] = {}
    for file_path, structure in structures.items():
        if structure.cross_references:
            all_refs[file_path] = structure.cross_references

    ref_map = resolve_references(all_refs, list(structures.keys()))

    # Step 6: Detect amendment chains
    amendment_chains = detect_amendments(structures, doc_hierarchy.nodes)

    # Build and store the workspace context
    ctx = WorkspaceContext(
        directory=abs_directory,
        structures=structures,
        hierarchy=doc_hierarchy,
        reference_map=ref_map,
        amendment_chains=amendment_chains,
    )
    _WORKSPACE_CONTEXT = ctx

    return ctx


# =============================================================================
# Read Section Tool
# =============================================================================


def read_section(file_path: str, section: str) -> str:
    """
    Read a specific section from a pre-parsed document.

    This is a tool the agent can use instead of parse_file when it only
    needs a specific section (e.g., "Section 2.1" or "Article IV").
    Much more token-efficient than reading the entire document.

    Args:
        file_path: Path to the document file.
        section: Section identifier (e.g., "2.1", "Article IV", "4.2(b)").

    Returns:
        The section content, or available sections if not found.
    """
    ctx = get_workspace_context()

    if ctx is None or file_path not in ctx.structures:
        # Fallback: try to parse the file on demand
        try:
            from .fs import get_parsed_content

            markdown = get_parsed_content(file_path)
            structure = parse_document_structure(file_path, markdown)
        except Exception as e:
            return f"Error: Could not parse {file_path}: {e}"
    else:
        structure = ctx.structures[file_path]

    # Direct section lookup
    if section in structure.sections:
        content = structure.sections[section]

        # Check if this section has been amended
        amendment_note = ""
        if ctx and ctx.amendment_chains:
            for chain in ctx.amendment_chains.values():
                if section in chain.effective_clauses:
                    clause = chain.effective_clauses[section]
                    if clause.is_amended:
                        source = clause.amendment_history[-1]["source"]
                        amendment_note = (
                            f"\n\n⚠️ AMENDMENT NOTICE: This section was modified by "
                            f"{source}. The text above is from the ORIGINAL document. "
                            f"Check the amendment for the current effective version."
                        )

        return (
            f"=== Section {section} of {os.path.basename(file_path)} ===\n\n"
            f"{content}{amendment_note}"
        )

    # Fuzzy matching: try partial matches
    matches: list[tuple[str, str]] = []
    section_lower = section.lower().strip()

    for sec_id, content in structure.sections.items():
        sec_lower = sec_id.lower().strip()
        if section_lower in sec_lower or sec_lower in section_lower:
            matches.append((sec_id, content))

    if len(matches) == 1:
        sec_id, content = matches[0]
        return (
            f"=== Section {sec_id} of {os.path.basename(file_path)} ===\n"
            f"(matched from query: '{section}')\n\n{content}"
        )
    elif len(matches) > 1:
        match_list = ", ".join(m[0] for m in matches)
        return (
            f"Multiple sections match '{section}': {match_list}. "
            f"Please be more specific."
        )

    # Section not found — show available sections
    available = ", ".join(sorted(structure.sections.keys())[:30])
    return (
        f"Section '{section}' not found in {os.path.basename(file_path)}.\n"
        f"Available sections: {available}\n\n"
        f"Tip: Use parse_file to read the full document, or try a different section ID."
    )


# =============================================================================
# Formatting for Agent
# =============================================================================


def format_workspace_context(ctx: WorkspaceContext, max_chars: int = 40_000, compact_override: bool = False) -> str:
    """
    Format the full workspace context as enriched context for the AI agent.

    This is shown to the agent at the START of exploration, giving it
    the hierarchy, reference map, amendment status, and document summaries
    before it reads a single document.

    Args:
        max_chars: Hard character ceiling for the returned string. When many
                   documents are present the per-doc summaries switch to a
                   compact format (TOC only) and the result is capped to keep
                   the agent's prompt within the model's context window.
        compact_override: When True (user explicitly toggled compact mode),
                          force compact summaries regardless of document count.
                          When False (default), compact is only auto-applied
                          when there are >15 documents.
    """
    # Compact per-doc summaries when the user explicitly requested it,
    # or when there are too many documents to fit in the model's context window.
    compact = compact_override or (len(ctx.structures) > 15)

    sections: list[str] = []

    # Document count summary
    sections.append(
        f"📊 WORKSPACE ANALYSIS: {len(ctx.structures)} documents ingested from "
        f"{os.path.basename(ctx.directory)}"
        + (" (compact summaries — use read_section/parse_file for full detail)" if compact else "")
    )
    sections.append("")

    # Hierarchy
    if ctx.hierarchy:
        sections.append(format_hierarchy(ctx.hierarchy, ctx.structures))
        sections.append("")

    # Reference map
    if ctx.reference_map:
        sections.append(format_reference_map(ctx.reference_map))
        sections.append("")

    # Amendment chains
    if ctx.amendment_chains:
        sections.append(format_amendment_chains(ctx.amendment_chains))
        sections.append("")

    # Per-document summaries (brief)
    sections.append("📄 DOCUMENT SUMMARIES")
    sections.append("=" * 50)
    reading_order = (
        ctx.hierarchy.reading_order if ctx.hierarchy else list(ctx.structures.keys())
    )
    for file_path in reading_order:
        if file_path in ctx.structures:
            sections.append(
                format_structure_for_agent(ctx.structures[file_path], compact=compact)
            )
            sections.append("")

    result = "\n".join(sections)

    # Hard cap: truncate at a newline boundary to avoid splitting mid-section.
    if len(result) > max_chars:
        truncated = result[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            truncated = truncated[:last_newline]
        result = truncated + "\n\n[... workspace context truncated — use read_section or parse_file to access remaining documents ...]"

    return result
