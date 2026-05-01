"""
Post-answer citation verification.

After the AI agent produces an answer, this module parses out every citation,
checks it against the actual source documents, and flags any mismatches.
This prevents the #1 risk in legal AI: hallucinated citations.

Verification strategy (hybrid):
1. Fast word-overlap check first (free, instant).
2. For citations in the "unsure zone" (overlap 0.2–0.5), escalate to an LLM
   for semantic verification — asks the model whether the source text supports
   the claim.
"""

import asyncio
import re
import os
from dataclasses import dataclass, field


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class CitationCheck:
    """Result of verifying a single citation."""

    claim: str  # The factual claim being made
    cited_file: str  # File referenced in the citation
    cited_section: str | None  # Section referenced (if any)
    status: str = ""  # "verified", "unverified", "file_not_found", "section_not_found"
    source_text: str | None = None  # Actual text from the cited location
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class VerificationReport:
    """Complete verification report for an answer."""

    total_citations: int = 0
    verified: int = 0
    unverified: int = 0
    file_not_found: int = 0
    section_not_found: int = 0
    checks: list[CitationCheck] = field(default_factory=list)
    overall_confidence: float = 0.0


# =============================================================================
# Citation Parsing
# =============================================================================

# Pattern: [Source: filename, Section X.Y] or [Source: filename, Page X]
CITATION_PATTERN = re.compile(
    r"\[Source:\s*"  # Opening [Source:
    r"([^,\]]+)"  # Filename (group 1)
    r"(?:,\s*"  # Optional comma separator
    r"((?:Section|Article|Exhibit|Schedule|Page|§)\s*[^\]]+))?"  # Section reference (group 2)
    r"\]",  # Closing ]
    re.IGNORECASE,
)

# Pattern to extract the claim before a citation
CLAIM_PATTERN = re.compile(
    r"([^.!?\n\[]{10,200})"  # Text before the citation (10-200 chars)
    r"\s*\[Source:",  # Followed by a citation
    re.IGNORECASE,
)


def _parse_citations(answer_text: str) -> list[dict]:
    """
    Extract all citations from the agent's answer text.

    Returns a list of dicts with: claim, file, section.
    """
    citations: list[dict] = []

    for match in CITATION_PATTERN.finditer(answer_text):
        cited_file = match.group(1).strip()
        cited_section = match.group(2).strip() if match.group(2) else None

        # Extract the claim: the full sentence/bullet ending at this citation.
        # Walk backwards from the citation start to find the nearest sentence/line boundary.
        before_text = answer_text[: match.start()].rstrip()
        # Find the last sentence boundary (newline, period+space, colon) within 600 chars
        window = before_text[-600:]
        # Prefer the last newline (captures the full bullet point)
        last_newline = window.rfind("\n")
        if last_newline != -1:
            claim = window[last_newline:].strip(" \t-*•")
        else:
            # Fall back to last sentence
            last_period = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            claim = window[last_period + 2:].strip() if last_period != -1 else window.strip()

        # Strip trailing citation artifacts that bled into the claim
        claim = re.sub(r"\[Source:.*$", "", claim).strip()

        citations.append(
            {
                "claim": claim,
                "file": cited_file,
                "section": cited_section,
            }
        )

    return citations


# =============================================================================
# Verification
# =============================================================================


def _find_file_in_structures(
    cited_filename: str, structures: dict
) -> str | None:
    """Find the actual file path for a cited filename."""
    # Exact basename match
    for file_path in structures:
        if os.path.basename(file_path) == cited_filename:
            return file_path

    # Fuzzy match (remove common variations)
    cited_norm = re.sub(r"[^a-z0-9]", "", cited_filename.lower())
    for file_path in structures:
        fname_norm = re.sub(r"[^a-z0-9]", "", os.path.basename(file_path).lower())
        if cited_norm == fname_norm or cited_norm in fname_norm or fname_norm in cited_norm:
            return file_path

    return None


def _find_section_text(
    structure, section_ref: str
) -> str | None:
    """Find the text of a referenced section in a document structure."""
    if not section_ref:
        return None

    # Extract section number from reference like "Section 2.1" or "Article IV"
    section_match = re.search(
        r"(?:Section|Article|§)\s*(\d+(?:\.\d+)*(?:\([a-z]\))?|[IVXLCDMivxlcdm]+)",
        section_ref,
        re.IGNORECASE,
    )
    if not section_match:
        return None

    section_id = section_match.group(1)

    # Direct lookup
    if section_id in structure.sections:
        return structure.sections[section_id]

    # Try with "Article" prefix
    article_key = f"Article {section_id}"
    if article_key in structure.sections:
        return structure.sections[article_key]

    # Fuzzy match
    section_lower = section_id.lower()
    for key, content in structure.sections.items():
        if section_lower in key.lower() or key.lower() in section_lower:
            return content

    return None


def _text_overlap_score(claim: str, source_text: str) -> float:
    """
    Compute a simple overlap score between a claim and source text.

    Checks how many significant words from the claim appear in the source.
    """
    if not claim or not source_text:
        return 0.0

    # Extract significant words (skip common words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "than", "too", "very", "just", "that", "this", "these",
        "those", "it", "its",
    }

    claim_words = set(
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]+\b", claim)
        if len(w) > 2 and w.lower() not in stop_words
    )

    if not claim_words:
        return 0.5  # Can't assess, give neutral score

    source_lower = source_text.lower()
    matches = sum(1 for w in claim_words if w in source_lower)

    return matches / len(claim_words)


def _extract_compound_files(section_str: str, structures: dict) -> list[str]:
    """
    Extract additional file references from a compound citation's section field.

    When the LLM writes [Source: file1.pdf, Section X; file2.pdf, Section Y],
    the parser captures file1 as the primary and the rest as the "section".
    This helper extracts the additional filenames from that section string.

    Returns a list of resolved file paths (only those found in structures).
    """
    if not section_str:
        return []

    # Find all filename-like patterns (.pdf, .docx, .txt, .md, .xlsx)
    filename_pattern = re.compile(r"[\w\-\.]+\.(?:pdf|docx|txt|md|xlsx)", re.IGNORECASE)
    extra_files: list[str] = []

    for match in filename_pattern.finditer(section_str):
        fname = match.group(0)
        resolved = _find_file_in_structures(fname, structures)
        if resolved and resolved not in extra_files:
            extra_files.append(resolved)

    return extra_files


def verify_answer(answer_text: str, workspace_context) -> str:
    """Synchronous wrapper — use verify_answer_async for LLM-based verification."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Can't call asyncio.run inside a running loop — return sync-only result
        return _verify_sync(answer_text, workspace_context)
    return asyncio.run(verify_answer_async(answer_text, workspace_context))


async def verify_answer_async(
    answer_text: str,
    workspace_context,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Verify all citations in the agent's answer (hybrid: word-overlap + LLM).

    For citations in the unsure zone (overlap 0.2–0.5), an LLM call decides
    whether the source text semantically supports the claim.

    Args:
        answer_text: The agent's final answer text.
        workspace_context: The WorkspaceContext with parsed document structures.
        provider: LLM provider for semantic checks (uses current agent provider if None).
        model: Model override for verification calls.
        api_key: API key override for verification calls.

    Returns:
        A formatted verification report string, or empty string if no citations.
    """
    from .llm import generate_text
    from .workflow import _AGENT_PROVIDER, _AGENT_MODEL, _AGENT_API_KEY

    # Resolve provider settings — default to same as the agent
    v_provider = provider or _AGENT_PROVIDER
    v_model = model or _AGENT_MODEL
    v_api_key = api_key or _AGENT_API_KEY

    citations = _parse_citations(answer_text)

    if not citations:
        return ""

    report = VerificationReport(total_citations=len(citations))
    # Collect checks that need LLM escalation
    llm_pending: list[tuple[CitationCheck, str]] = []  # (check, source_text)

    for cit in citations:
        # Skip synthetic citations that reference context, not real files
        if re.search(r"pre.?analyzed|document context|context", cit["file"], re.IGNORECASE):
            continue

        # Skip citations that reference tool outputs (inherently unverifiable)
        combined = f"{cit['file']} {cit.get('section', '')}"
        if re.search(r"glob result|scan_folder result|grep result|tool result|workspace file list", combined, re.IGNORECASE):
            report.total_citations -= 1
            continue

        check = CitationCheck(
            claim=cit["claim"],
            cited_file=cit["file"],
            cited_section=cit.get("section"),
        )

        # Detect compound citations (multiple files in one [Source: ...] bracket)
        # by looking for additional .pdf/.docx/.txt filenames in the section field
        all_source_texts: list[str] = []
        extra_files = _extract_compound_files(cit.get("section", ""), workspace_context.structures)

        # Find the primary file
        file_path = _find_file_in_structures(cit["file"], workspace_context.structures)

        if file_path is None:
            check.status = "file_not_found"
            check.confidence = 0.0
            report.file_not_found += 1
            report.checks.append(check)
            continue

        structure = workspace_context.structures[file_path]

        if cit.get("section"):
            section_text = _find_section_text(structure, cit["section"])
            if section_text is None:
                primary_text = structure.full_text[:2000] if hasattr(structure, "full_text") else ""
            else:
                primary_text = section_text[:2000]
        else:
            primary_text = structure.full_text[:2000] if hasattr(structure, "full_text") else ""

        all_source_texts.append(f"[{os.path.basename(file_path)}]\n{primary_text}")

        # Gather text from additional referenced files in compound citations
        for extra_path in extra_files:
            extra_struct = workspace_context.structures.get(extra_path)
            if extra_struct:
                extra_text = extra_struct.full_text[:1500] if hasattr(extra_struct, "full_text") else ""
                all_source_texts.append(f"[{os.path.basename(extra_path)}]\n{extra_text}")

        combined_source = "\n\n".join(all_source_texts)
        score = _text_overlap_score(cit["claim"], combined_source)
        check.confidence = score
        check.source_text = combined_source[:300]

        if score >= 0.5:
            check.status = "verified"
            report.verified += 1
        else:
            # Escalate to LLM with all referenced source texts
            llm_pending.append((check, combined_source[:3000]))

        report.checks.append(check)

    # === LLM semantic verification for uncertain citations ===
    if llm_pending:
        async def _llm_verify(check: CitationCheck, source: str) -> None:
            prompt = (
                "You are a citation verifier. Determine if the SOURCE TEXT supports the CLAIM.\n\n"
                f"CLAIM: {check.claim}\n\n"
                f"SOURCE TEXT (from {check.cited_file}"
                f"{', ' + check.cited_section if check.cited_section else ''}):\n"
                f"{source}\n\n"
                "Does the source text support this claim? Answer ONLY one of:\n"
                "- SUPPORTED — the source clearly supports this claim (directly or by reasonable inference)\n"
                "- PARTIAL — the source partially supports it but key details are missing\n"
                "- NOT_SUPPORTED — the source does not contain information supporting this claim\n\n"
                "Answer:"
            )
            try:
                result, _ = await generate_text(
                    prompt=prompt,
                    provider=v_provider,
                    model=v_model,
                    api_key=v_api_key,
                    system_prompt="You are a precise citation verification assistant. Respond with exactly one word: SUPPORTED, PARTIAL, or NOT_SUPPORTED.",
                )
                result_upper = result.strip().upper()
                if "SUPPORTED" in result_upper and "NOT_SUPPORTED" not in result_upper:
                    check.status = "verified"
                    check.confidence = 0.8
                    report.verified += 1
                elif "PARTIAL" in result_upper:
                    check.status = "verified"
                    check.confidence = 0.6
                    report.verified += 1
                else:
                    check.status = "unverified"
                    check.confidence = check.confidence  # keep original low score
                    report.unverified += 1
            except Exception:
                # LLM call failed — fall back to original word-overlap verdict
                if check.confidence >= 0.3:
                    check.status = "verified"
                    report.verified += 1
                else:
                    check.status = "unverified"
                    report.unverified += 1

        # Run all LLM checks in parallel
        await asyncio.gather(*[_llm_verify(chk, src) for chk, src in llm_pending])

    # Calculate overall confidence
    if report.total_citations > 0:
        report.overall_confidence = sum(c.confidence for c in report.checks) / report.total_citations

    return _format_report(report)


def _verify_sync(answer_text: str, workspace_context) -> str:
    """Pure word-overlap verification (no LLM calls). Used as fallback."""
    citations = _parse_citations(answer_text)
    if not citations:
        return ""

    report = VerificationReport(total_citations=len(citations))

    for cit in citations:
        if re.search(r"pre.?analyzed|document context|context", cit["file"], re.IGNORECASE):
            continue

        check = CitationCheck(
            claim=cit["claim"],
            cited_file=cit["file"],
            cited_section=cit.get("section"),
        )

        file_path = _find_file_in_structures(cit["file"], workspace_context.structures)

        if file_path is None:
            check.status = "file_not_found"
            check.confidence = 0.0
            report.file_not_found += 1
        else:
            structure = workspace_context.structures[file_path]
            if cit.get("section"):
                section_text = _find_section_text(structure, cit["section"])
                if section_text is None:
                    score = _text_overlap_score(cit["claim"], structure.full_text if hasattr(structure, "full_text") else "")
                    check.confidence = score
                    if score >= 0.4:
                        check.status = "verified"
                        report.verified += 1
                    else:
                        check.status = "section_not_found"
                        report.section_not_found += 1
                else:
                    score = _text_overlap_score(cit["claim"], section_text)
                    check.source_text = section_text[:200]
                    check.confidence = score
                    if score >= 0.4:
                        check.status = "verified"
                        report.verified += 1
                    else:
                        check.status = "unverified"
                        report.unverified += 1
            else:
                score = _text_overlap_score(cit["claim"], structure.full_text if hasattr(structure, "full_text") else "")
                check.confidence = score
                if score >= 0.4:
                    check.status = "verified"
                    report.verified += 1
                else:
                    check.status = "unverified"
                    report.unverified += 1

        report.checks.append(check)

    if report.total_citations > 0:
        report.overall_confidence = sum(c.confidence for c in report.checks) / report.total_citations

    return _format_report(report)


def _format_report(report: VerificationReport) -> str:
    """Format the verification report for display."""
    if report.overall_confidence >= 0.7:
        status = "High confidence"
    elif report.overall_confidence >= 0.4:
        status = "Moderate confidence"
    else:
        status = "Low confidence"

    lines = [
        "---",
        f"Citations — {report.verified}/{report.total_citations} verified · {status} ({report.overall_confidence:.0%})",
    ]

    issues = [c for c in report.checks if c.status != "verified"]
    if issues:
        parts = []
        for check in issues:
            label = f"`{check.cited_file}`"
            if check.cited_section:
                label += f" § {check.cited_section}"
            parts.append(label)
        lines.append(f"*Could not verify*: {', '.join(parts)}")

    return "\n".join(lines)
