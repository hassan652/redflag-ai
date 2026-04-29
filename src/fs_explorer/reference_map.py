"""
Cross-reference extraction and resolution for legal documents.

Extracts references like "See Exhibit A", "as defined in the Master Agreement",
"refer to Schedule 3" from document text, and resolves them to actual files.
"""

import re
import os
from dataclasses import dataclass, field


# =============================================================================
# Reference Patterns
# =============================================================================

# Patterns for extracting cross-references from legal text
REFERENCE_PATTERNS: list[tuple[str, str]] = [
    # "See Exhibit A", "per Exhibit B-1", "attached as Exhibit C"
    (
        r'(?:see|per|in|under|to|attached\s+(?:hereto\s+)?as)\s+'
        r'(?:the\s+)?'
        r'((?:Exhibit|Schedule|Appendix|Annex|Attachment)\s+[A-Z0-9](?:-\d+)?)',
        "exhibit_reference",
    ),
    # "as defined in the Master Agreement", "pursuant to the Escrow Agreement"
    (
        r'(?:as\s+(?:defined|set\s+forth|described|provided|specified)\s+in|'
        r'pursuant\s+to|under\s+the|in\s+accordance\s+with|'
        r'referenced\s+in|referred\s+to\s+in)\s+'
        r'(?:the\s+)?'
        r'([A-Z][A-Za-z\s]{3,60}(?:Agreement|Letter|Report|Memo|Schedule|'
        r'Certificate|Plan|Checklist|Opinion|Statement|Assignment|Registration))',
        "document_reference",
    ),
    # "Document: Risk Assessment Memo", "Document: IP Certification Letter"
    (
        r'(?:Document:\s*)([A-Z][A-Za-z\s]{3,60}(?:Agreement|Letter|Report|Memo|'
        r'Schedule|Certificate|Plan|Checklist|Opinion|Statement|Assignment|Registration))',
        "explicit_document_reference",
    ),
    # "Section 2.1", "Article IV", "Section 4.2(b)"
    (
        r'(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)',
        "section_reference",
    ),
    # "Exhibit A - Financial Terms" (with title)
    (
        r'(Exhibit\s+[A-Z0-9](?:-\d+)?\s*[-–—]\s*[A-Z][A-Za-z\s]{3,50})',
        "titled_exhibit_reference",
    ),
    # "Schedule 1 - IP Assets", "Schedule 3 - Employee Transition Plan"
    (
        r'(Schedule\s+\d+\s*[-–—]\s*[A-Z][A-Za-z\s]{3,50})',
        "titled_schedule_reference",
    ),
]


@dataclass
class CrossReference:
    """A single cross-reference found in a document."""

    source_file: str
    reference_text: str
    reference_type: str  # "exhibit_reference", "document_reference", etc.
    context: str  # Surrounding text for disambiguation
    position: int  # Character position in source document
    resolved_file: str | None = None  # Resolved target file path


@dataclass
class ReferenceMap:
    """Complete map of cross-references across all documents."""

    references: list[CrossReference] = field(default_factory=list)
    # file_path -> list of references FROM that file
    outgoing: dict[str, list[CrossReference]] = field(default_factory=dict)
    # file_path -> list of references TO that file
    incoming: dict[str, list[CrossReference]] = field(default_factory=dict)
    # Unresolved references (couldn't match to a file)
    unresolved: list[CrossReference] = field(default_factory=list)


# =============================================================================
# Extraction
# =============================================================================


def extract_references(document_text: str) -> list[dict]:
    """
    Extract all cross-references from a document's text.

    This is called at ingest time for each document. Returns a list
    of reference dictionaries.

    Args:
        document_text: The full markdown text of the document.

    Returns:
        List of reference dicts with text, type, context, and position.
    """
    references: list[dict] = []
    seen: set[str] = set()

    for pattern_str, ref_type in REFERENCE_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)

        for match in pattern.finditer(document_text):
            ref_text = match.group(1).strip() if match.lastindex else match.group(0).strip()

            # Deduplicate
            dedup_key = f"{ref_type}:{ref_text.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Extract surrounding context (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(document_text), match.end() + 100)
            context = document_text[start:end].replace("\n", " ").strip()

            references.append(
                {
                    "text": ref_text,
                    "type": ref_type,
                    "context": context,
                    "position": match.start(),
                }
            )

    return references


# =============================================================================
# Resolution
# =============================================================================


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _compute_match_score(reference_text: str, filename: str) -> float:
    """
    Compute a match score between a reference and a filename.

    Returns a score between 0.0 (no match) and 1.0 (perfect match).
    """
    ref_norm = _normalize_for_matching(reference_text)
    file_norm = _normalize_for_matching(os.path.splitext(filename)[0])

    if not ref_norm or not file_norm:
        return 0.0

    # Exact containment
    if ref_norm in file_norm or file_norm in ref_norm:
        return 0.9

    # Word overlap
    ref_words = set(re.findall(r"[a-z]+", ref_norm))
    file_words = set(re.findall(r"[a-z]+", file_norm))

    if not ref_words or not file_words:
        return 0.0

    overlap = ref_words & file_words
    if not overlap:
        return 0.0

    # Jaccard similarity
    score = len(overlap) / len(ref_words | file_words)

    # Boost if key legal terms match
    legal_terms = {
        "agreement",
        "exhibit",
        "schedule",
        "escrow",
        "opinion",
        "nda",
        "amendment",
        "checklist",
        "disclosure",
        "certification",
        "assignment",
        "registration",
    }
    legal_overlap = overlap & legal_terms
    if legal_overlap:
        score += 0.1 * len(legal_overlap)

    return min(score, 1.0)


def resolve_references(
    all_references: dict[str, list[dict]],
    available_files: list[str],
    threshold: float = 0.3,
) -> ReferenceMap:
    """
    Resolve extracted cross-references to actual files.

    Takes all references from all documents and tries to match each
    reference text to one of the available files.

    Args:
        all_references: Dict mapping source file path -> list of reference dicts.
        available_files: List of all available file paths.
        threshold: Minimum match score to consider a resolution valid.

    Returns:
        A ReferenceMap with resolved and unresolved references.
    """
    ref_map = ReferenceMap()
    filenames = {os.path.basename(f): f for f in available_files}

    for source_file, refs in all_references.items():
        for ref_dict in refs:
            ref = CrossReference(
                source_file=source_file,
                reference_text=ref_dict["text"],
                reference_type=ref_dict["type"],
                context=ref_dict.get("context", ""),
                position=ref_dict.get("position", 0),
            )

            # Skip internal section references (they reference the same document)
            if ref.reference_type == "section_reference":
                ref.resolved_file = source_file
                ref_map.references.append(ref)
                continue

            # Try to resolve to a file
            best_score = 0.0
            best_file = None

            for fname, fpath in filenames.items():
                if fpath == source_file:
                    continue  # Don't match to self

                score = _compute_match_score(ref.reference_text, fname)
                if score > best_score:
                    best_score = score
                    best_file = fpath

            if best_score >= threshold and best_file is not None:
                ref.resolved_file = best_file
                ref_map.references.append(ref)

                # Track outgoing references
                if source_file not in ref_map.outgoing:
                    ref_map.outgoing[source_file] = []
                ref_map.outgoing[source_file].append(ref)

                # Track incoming references
                if best_file not in ref_map.incoming:
                    ref_map.incoming[best_file] = []
                ref_map.incoming[best_file].append(ref)
            else:
                ref.resolved_file = None
                ref_map.unresolved.append(ref)

    return ref_map


def format_reference_map(ref_map: ReferenceMap) -> str:
    """
    Format the reference map as a readable summary for the AI agent.

    Shows which documents reference which other documents, so the agent
    can plan its reading order without discovering references one at a time.
    """
    if not ref_map.references and not ref_map.unresolved:
        return "No cross-references detected between documents."

    lines: list[str] = []
    lines.append("📎 CROSS-REFERENCE MAP")
    lines.append("=" * 50)

    # Group by source file
    for source_file, refs in sorted(ref_map.outgoing.items()):
        source_name = os.path.basename(source_file)
        lines.append(f"\n  {source_name} references:")
        seen_targets: set[str] = set()
        for ref in refs:
            if ref.resolved_file and ref.resolved_file not in seen_targets:
                target_name = os.path.basename(ref.resolved_file)
                lines.append(f"    → {target_name}  (via: \"{ref.reference_text}\")")
                seen_targets.add(ref.resolved_file)

    # Show most-referenced documents
    if ref_map.incoming:
        lines.append(f"\n  Most Referenced Documents:")
        sorted_incoming = sorted(
            ref_map.incoming.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        for fpath, refs in sorted_incoming[:5]:
            fname = os.path.basename(fpath)
            sources = set(os.path.basename(r.source_file) for r in refs)
            lines.append(
                f"    ⭐ {fname} — referenced by {len(sources)} document(s)"
            )

    if ref_map.unresolved:
        lines.append(f"\n  ⚠️  {len(ref_map.unresolved)} unresolved reference(s)")

    return "\n".join(lines)
