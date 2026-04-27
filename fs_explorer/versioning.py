"""
Amendment chain tracking and document versioning.

Detects amendment relationships between documents and builds
"effective" versions of clauses — the current state after all
amendments are applied. This prevents the #1 junior lawyer mistake:
citing the original contract without checking if it was amended.
"""

import re
import os
from dataclasses import dataclass, field

from .document_structure import DocumentStructure


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class ClauseVersion:
    """A single clause's version history across amendments."""

    section: str  # e.g., "3.2"
    original_text: str  # Text from the original/master document
    current_text: str  # Text after all amendments applied
    is_amended: bool = False
    amendment_history: list[dict] = field(
        default_factory=list
    )  # [{"source": "Amendment 1", "date": "...", "change_summary": "..."}]


@dataclass
class AmendmentChain:
    """Tracks the full amendment chain for a master document."""

    master_file: str
    master_title: str
    amendments: list[dict] = field(
        default_factory=list
    )  # [{"file": "...", "title": "...", "date": "...", "sections_modified": [...]}]
    effective_clauses: dict[str, ClauseVersion] = field(
        default_factory=dict
    )  # section -> ClauseVersion
    has_amendments: bool = False


# =============================================================================
# Amendment Detection
# =============================================================================


def _detect_modified_sections(amendment_text: str) -> list[str]:
    """
    Detect which sections an amendment modifies.

    Looks for patterns like:
    - "Section 3.2 is hereby amended..."
    - "Article IV shall be deleted..."
    - "The following is added as Section 5.3..."
    """
    patterns = [
        r"Section\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)\s+(?:is|shall be)\s+(?:hereby\s+)?(?:amended|modified|replaced|deleted|restated)",
        r"(?:amend|modify|replace|restate|delete)\s+Section\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)",
        r"Article\s+([IVXLCDMivxlcdm]+|\d+)\s+(?:is|shall be)\s+(?:hereby\s+)?(?:amended|modified|replaced)",
        r"(?:new|additional)\s+Section\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)\s+(?:is|shall be)\s+(?:hereby\s+)?(?:added|inserted)",
    ]

    sections: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, amendment_text, re.IGNORECASE):
            section = match.group(1).strip()
            if section not in sections:
                sections.append(section)

    return sections


def detect_amendments(
    structures: dict[str, DocumentStructure],
    hierarchy_nodes: dict[str, object],
) -> dict[str, AmendmentChain]:
    """
    Detect amendment chains across all documents.

    For each master document, finds all amendments and tracks which
    sections they modify. Returns a dict of master_file -> AmendmentChain.

    Args:
        structures: All parsed document structures.
        hierarchy_nodes: Document hierarchy nodes (for type info).

    Returns:
        Dict mapping master file path -> AmendmentChain.
    """
    chains: dict[str, AmendmentChain] = {}

    # Find master documents and their amendments
    masters: list[str] = []
    amendments: list[str] = []

    for file_path, structure in structures.items():
        if structure.document_type == "master":
            masters.append(file_path)
        elif structure.document_type == "amendment":
            amendments.append(file_path)

    # If no explicit master, treat the first document as master
    if not masters and structures:
        first_key = next(iter(structures))
        masters.append(first_key)

    # Build chains for each master
    for master_path in masters:
        master = structures[master_path]
        chain = AmendmentChain(
            master_file=master_path,
            master_title=master.title,
        )

        # Add original sections as base
        for section_id, content in master.sections.items():
            chain.effective_clauses[section_id] = ClauseVersion(
                section=section_id,
                original_text=content,
                current_text=content,
                is_amended=False,
            )

        # Find amendments that reference this master
        for amend_path in amendments:
            amend = structures[amend_path]

            # Check if this amendment references the master
            references_master = False
            for ref in amend.cross_references:
                ref_text = ref.get("text", "").lower()
                master_keywords = master.title.lower().split()
                if any(kw in ref_text for kw in master_keywords if len(kw) > 3):
                    references_master = True
                    break

            # Also check by proximity (amendments in same folder likely relate)
            if not references_master:
                if os.path.dirname(master_path) == os.path.dirname(amend_path):
                    references_master = True

            if references_master:
                modified_sections = _detect_modified_sections(amend.full_text)

                chain.amendments.append(
                    {
                        "file": amend_path,
                        "title": amend.title,
                        "date": amend.date,
                        "sections_modified": modified_sections,
                    }
                )

                # Update effective clauses for modified sections
                for section_id in modified_sections:
                    if section_id in amend.sections:
                        new_text = amend.sections[section_id]

                        if section_id in chain.effective_clauses:
                            clause = chain.effective_clauses[section_id]
                            clause.current_text = new_text
                            clause.is_amended = True
                            clause.amendment_history.append(
                                {
                                    "source": os.path.basename(amend_path),
                                    "date": amend.date,
                                    "change_summary": f"Modified by {amend.title}",
                                }
                            )
                        else:
                            # New section added by amendment
                            chain.effective_clauses[section_id] = ClauseVersion(
                                section=section_id,
                                original_text="[Added by amendment]",
                                current_text=new_text,
                                is_amended=True,
                                amendment_history=[
                                    {
                                        "source": os.path.basename(amend_path),
                                        "date": amend.date,
                                        "change_summary": f"Added by {amend.title}",
                                    }
                                ],
                            )

                chain.has_amendments = True

        chains[master_path] = chain

    return chains


# =============================================================================
# Formatting
# =============================================================================


def format_amendment_chains(chains: dict[str, AmendmentChain]) -> str:
    """Format amendment chain information for the AI agent."""
    lines: list[str] = []

    has_any_amendments = any(chain.has_amendments for chain in chains.values())

    if not has_any_amendments:
        return "📝 No amendments detected. All documents appear to be original versions."

    lines.append("📝 AMENDMENT CHAINS")
    lines.append("=" * 50)

    for master_path, chain in chains.items():
        master_name = os.path.basename(master_path)
        lines.append(f"\n  📄 {master_name}: {chain.master_title}")

        if chain.amendments:
            lines.append(f"  ⚠️  {len(chain.amendments)} amendment(s) found:")
            for amend in chain.amendments:
                amend_name = os.path.basename(amend["file"])
                sections = ", ".join(amend["sections_modified"]) or "unspecified sections"
                date_str = f" ({amend['date']})" if amend.get("date") else ""
                lines.append(f"     → {amend_name}{date_str}")
                lines.append(f"       Modifies: {sections}")

            # Show amended clauses
            amended_clauses = [
                (sid, clause)
                for sid, clause in chain.effective_clauses.items()
                if clause.is_amended
            ]
            if amended_clauses:
                lines.append(f"  Modified Sections:")
                for section_id, clause in sorted(amended_clauses):
                    source = clause.amendment_history[-1]["source"] if clause.amendment_history else "unknown"
                    lines.append(f"     ⚡ Section {section_id} — last modified by {source}")
        else:
            lines.append("  ✅ No amendments found")

    lines.append("")
    lines.append(
        "  ⚠️  IMPORTANT: When citing amended sections, always cite the"
    )
    lines.append(
        "     amendment source, not the original document."
    )

    return "\n".join(lines)
