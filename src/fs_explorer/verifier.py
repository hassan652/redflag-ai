"""
Post-answer citation verification.

After the AI agent produces an answer, this module parses out every citation,
checks it against the actual source documents, and flags any mismatches.
This prevents the #1 risk in legal AI: hallucinated citations.
"""

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
    status: str  # "verified", "unverified", "file_not_found", "section_not_found"
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

        # Try to find the claim this citation supports
        claim = ""
        # Look backwards from the citation for the claim text
        before_text = answer_text[: match.start()]
        claim_match = CLAIM_PATTERN.search(before_text[-300:] + " [Source:")
        if claim_match:
            claim = claim_match.group(1).strip()
        else:
            # Fallback: take the sentence before the citation
            sentences = before_text.rsplit(".", 2)
            if len(sentences) >= 2:
                claim = sentences[-1].strip()

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


def verify_answer(answer_text: str, workspace_context) -> str:
    """
    Verify all citations in the agent's answer.

    Parses citations, checks them against source documents,
    and produces a verification summary.

    Args:
        answer_text: The agent's final answer text.
        workspace_context: The WorkspaceContext with parsed document structures.

    Returns:
        A formatted verification report string, or empty string if no citations.
    """
    citations = _parse_citations(answer_text)

    if not citations:
        return ""

    report = VerificationReport(total_citations=len(citations))

    for cit in citations:
        check = CitationCheck(
            claim=cit["claim"],
            cited_file=cit["file"],
            cited_section=cit.get("section"),
        )

        # Find the file
        file_path = _find_file_in_structures(cit["file"], workspace_context.structures)

        if file_path is None:
            check.status = "file_not_found"
            check.confidence = 0.0
            report.file_not_found += 1
        else:
            structure = workspace_context.structures[file_path]

            if cit.get("section"):
                # Try to find the specific section
                section_text = _find_section_text(structure, cit["section"])

                if section_text is None:
                    check.status = "section_not_found"
                    check.confidence = 0.3  # File exists but section not found
                    report.section_not_found += 1
                else:
                    # Check if the claim is supported by the section text
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
                # No section specified — check against full document
                score = _text_overlap_score(cit["claim"], structure.full_text)
                check.confidence = score

                if score >= 0.4:
                    check.status = "verified"
                    report.verified += 1
                else:
                    check.status = "unverified"
                    report.unverified += 1

        report.checks.append(check)

    # Calculate overall confidence
    if report.total_citations > 0:
        report.overall_confidence = sum(c.confidence for c in report.checks) / report.total_citations

    return _format_report(report)


def _format_report(report: VerificationReport) -> str:
    """Format the verification report for display."""
    lines: list[str] = []

    # Header
    if report.overall_confidence >= 0.7:
        icon = "✅"
        status = "HIGH CONFIDENCE"
    elif report.overall_confidence >= 0.4:
        icon = "⚠️"
        status = "MEDIUM CONFIDENCE"
    else:
        icon = "❌"
        status = "LOW CONFIDENCE"

    lines.append(f"## Citation Verification {icon}")
    lines.append(f"**Overall: {status}** (confidence: {report.overall_confidence:.0%})")
    lines.append("")
    lines.append(
        f"| Metric | Count |\n|--------|-------|\n"
        f"| Total Citations | {report.total_citations} |\n"
        f"| Verified | {report.verified} |\n"
        f"| Unverified | {report.unverified} |\n"
        f"| File Not Found | {report.file_not_found} |\n"
        f"| Section Not Found | {report.section_not_found} |"
    )

    # Show issues
    issues = [c for c in report.checks if c.status != "verified"]
    if issues:
        lines.append("")
        lines.append("### ⚠️ Issues Found")
        for check in issues:
            lines.append(f"- **{check.status.replace('_', ' ').title()}**: "
                        f"`{check.cited_file}`"
                        f"{f', {check.cited_section}' if check.cited_section else ''}")
            if check.claim:
                lines.append(f"  Claim: \"{check.claim[:100]}...\"")

    return "\n".join(lines)
