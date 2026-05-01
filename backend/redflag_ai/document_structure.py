"""
Document structure parsing for legal documents.

Parses Docling markdown output into structured components:
table of contents, definitions, sections, and cross-references.
This enables section-aware reading — the AI gets the TOC and definitions
first, then requests specific sections instead of reading entire documents.
"""

import re
from dataclasses import dataclass, field


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class DocumentStructure:
    """
    Represents a parsed legal document with its internal structure.

    Created once at ingest time. The agent sees the TOC and definitions
    upfront, then uses read_section() to access specific sections.
    """

    file_path: str
    document_type: str  # "master", "amendment", "exhibit", "schedule", etc.
    title: str
    parties: list[str] = field(default_factory=list)
    date: str | None = None
    table_of_contents: list[dict] = field(
        default_factory=list
    )  # [{"section": "2.1", "title": "Purchase Price"}]
    definitions: dict[str, str] = field(
        default_factory=dict
    )  # {"Closing Date": "means the date..."}
    sections: dict[str, str] = field(
        default_factory=dict
    )  # {"2.1": "The purchase price shall be..."}
    cross_references: list[dict] = field(
        default_factory=list
    )  # [{"text": "See Exhibit A", "type": "exhibit_reference"}]
    full_text: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Extraction Helpers
# =============================================================================


def _extract_title(markdown: str) -> str:
    """Extract document title from the first heading or bold text."""
    # Try markdown heading
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Try first bold text
    match = re.search(r"\*\*(.+?)\*\*", markdown)
    if match:
        return match.group(1).strip()

    # Try first non-empty line
    for line in markdown.split("\n"):
        line = line.strip()
        if line and len(line) > 3:
            return line[:100]

    return "Untitled Document"


def _extract_sections(markdown: str) -> tuple[dict[str, str], list[dict]]:
    """
    Extract numbered sections and build table of contents.

    Handles formats like:
    - ## Section 2.1 - Purchase Price
    - **ARTICLE II - PURCHASE AND SALE**
    - 2.1. The purchase price shall be...
    - **2.1 Purchase Price**

    Returns:
        Tuple of (sections dict, table_of_contents list).
    """
    sections: dict[str, str] = {}
    toc: list[dict] = []

    # Pattern for section headers with numbers
    header_pattern = re.compile(
        r"^(?:#{1,4}\s+)?"  # Optional markdown header
        r"(?:\*\*\s*)?"  # Optional bold start
        r"(?:ARTICLE\s+)?"  # Optional "ARTICLE" prefix
        r"(?:Section\s+)?"  # Optional "Section" prefix
        r"(\d+(?:\.\d+)*(?:\([a-z]\))?)"  # Section number: 2.1, 4.2(b)
        r"[.\s\-–—:]*"  # Separator
        r"(.+?)"  # Section title
        r"(?:\*\*\s*)?$",  # Optional bold end
        re.MULTILINE | re.IGNORECASE,
    )

    # Also match ARTICLE I, ARTICLE II, etc.
    article_pattern = re.compile(
        r"^(?:#{1,4}\s+)?(?:\*\*\s*)?"
        r"ARTICLE\s+([IVXLCDMivxlcdm]+|\d+)"
        r"[.\s\-–—:]*"
        r"(.+?)"
        r"(?:\*\*\s*)?$",
        re.MULTILINE | re.IGNORECASE,
    )

    matches: list[tuple[int, str, str]] = []

    for match in header_pattern.finditer(markdown):
        section_num = match.group(1)
        title = match.group(2).strip().rstrip("*").strip()
        if title and len(title) > 1:
            matches.append((match.start(), section_num, title))

    for match in article_pattern.finditer(markdown):
        article_num = match.group(1).strip()
        title = match.group(2).strip().rstrip("*").strip()
        if title and len(title) > 1:
            section_key = f"Article {article_num}"
            matches.append((match.start(), section_key, title))

    # Sort by position in document
    matches.sort(key=lambda x: x[0])

    # Extract content between sections
    for i, (pos, section_num, title) in enumerate(matches):
        # Content goes from end of this header to start of next header
        content_start = markdown.index("\n", pos) + 1 if "\n" in markdown[pos:] else pos
        if i + 1 < len(matches):
            content_end = matches[i + 1][0]
        else:
            content_end = len(markdown)

        content = markdown[content_start:content_end].strip()
        sections[section_num] = content[:5000]  # Limit per-section size
        toc.append({"section": section_num, "title": title})

    return sections, toc


def _extract_definitions(markdown: str) -> dict[str, str]:
    """
    Extract defined terms from a legal document.

    Common patterns:
    - "Term" means ...
    - "Term" shall mean ...
    - "Term" has the meaning set forth in ...
    """
    definitions: dict[str, str] = {}

    # Pattern: "Defined Term" means/shall mean ...
    def_pattern = re.compile(
        r'"([A-Z][^"]{1,80})"'  # Quoted term starting with capital
        r"\s+(?:means?|shall\s+mean|refers?\s+to|has\s+the\s+meaning)"  # Definition verb
        r"\s+(.+?)(?:\.\s|\.\n|;\s|;\n|\n\n)",  # Definition text
        re.DOTALL | re.IGNORECASE,
    )

    for match in def_pattern.finditer(markdown):
        term = match.group(1).strip()
        definition = match.group(2).strip()[:500]
        definitions[term] = definition

    return definitions


def _extract_parties(markdown: str) -> list[str]:
    """Extract party names from a legal document."""
    parties: list[str] = []
    first_section = markdown[:3000]  # Parties usually in first few pages

    # Pattern: ("Buyer"), ("Seller"), ("Company"), ("Licensor"), etc.
    role_pattern = re.compile(
        r"([A-Z][A-Za-z\s,\.&]+?)\s*"
        r'\(\s*(?:the\s+)?"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"\s*\)'
    )
    for match in role_pattern.finditer(first_section):
        name = match.group(1).strip().rstrip(",").strip()
        role = match.group(2).strip()
        if name and len(name) > 2 and len(name) < 100:
            parties.append(f"{name} (\"{role}\")")

    # Pattern: "between X and Y"
    if not parties:
        between_pattern = re.compile(
            r"(?:between|by and between)\s+(.+?)\s+and\s+(.+?)(?:\.|,|\()",
            re.IGNORECASE,
        )
        match = between_pattern.search(first_section)
        if match:
            for group in [match.group(1), match.group(2)]:
                name = group.strip().rstrip(",").strip()
                if name and len(name) > 2 and len(name) < 100:
                    parties.append(name)

    return parties


def _extract_date(markdown: str) -> str | None:
    """Extract document date."""
    first_section = markdown[:3000]

    date_patterns = [
        r"(?:dated?|as\s+of|entered\s+into\s+(?:as\s+of\s+|on\s+))"
        r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        r"(?:Date:\s*)([A-Z]?[a-z]*\s*\d{1,2},?\s+\d{4})",
        r"(?:dated?|as\s+of)\s+(\d{1,2}/\d{1,2}/\d{4})",
        r"(?:dated?|as\s+of)\s+(\d{4}-\d{2}-\d{2})",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, first_section, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


# =============================================================================
# Document Type Classification
# =============================================================================

# Maps document types to identifying keywords (checked against filename + title + first 2000 chars)
DOCUMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "master": [
        "master agreement",
        "stock purchase agreement",
        "asset purchase agreement",
        "acquisition agreement",
        "merger agreement",
        "purchase agreement",
    ],
    "amendment": ["amendment", "amended and restated", "modification", "supplement"],
    "schedule": ["schedule"],
    "exhibit": ["exhibit"],
    "disclosure": ["disclosure schedule", "disclosure letter"],
    "side_letter": ["side letter", "letter agreement"],
    "escrow": ["escrow agreement"],
    "nda": ["non-disclosure", "nda", "confidentiality agreement"],
    "employment": [
        "employment agreement",
        "retention agreement",
        "consulting agreement",
        "non-compete",
        "non-competition",
        "benefit plan",
    ],
    "ip": [
        "ip assignment",
        "patent assignment",
        "trademark",
        "intellectual property",
    ],
    "opinion": ["legal opinion", "opinion letter"],
    "report": [
        "due diligence",
        "audit report",
        "financial statement",
        "risk assessment",
    ],
    "checklist": ["closing checklist", "closing memorandum", "closing certificate"],
    "regulatory": ["regulatory approval", "hsr", "antitrust"],
    "consent": ["consent", "customer consent", "third party consent"],
    "certificate": ["certification", "certificate", "officer's certificate"],
    "financial": [
        "financial adjustment",
        "financial terms",
        "financial statement",
        "stock purchase",
    ],
}


def _classify_document_type(file_path: str, title: str, markdown: str) -> str:
    """Classify document type based on filename, title, and content."""
    combined = f"{file_path.lower()} {title.lower()} {markdown[:2000].lower()}"

    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                return doc_type

    return "unknown"


# =============================================================================
# Main Parsing Function
# =============================================================================


def parse_document_structure(file_path: str, markdown_content: str) -> DocumentStructure:
    """
    Parse a document's markdown content into a structured representation.

    Called once at ingest time. Extracts the document's internal structure
    so the agent can read strategically instead of sequentially.

    Args:
        file_path: Path to the original document file.
        markdown_content: Markdown content from Docling parsing.

    Returns:
        A DocumentStructure with all extracted components.
    """
    title = _extract_title(markdown_content)
    sections, toc = _extract_sections(markdown_content)
    definitions = _extract_definitions(markdown_content)
    parties = _extract_parties(markdown_content)
    date = _extract_date(markdown_content)
    doc_type = _classify_document_type(file_path, title, markdown_content)

    # Extract cross-references
    from .reference_map import extract_references

    cross_refs = extract_references(markdown_content)

    return DocumentStructure(
        file_path=file_path,
        document_type=doc_type,
        title=title,
        parties=parties,
        date=date,
        table_of_contents=toc,
        definitions=definitions,
        sections=sections,
        cross_references=cross_refs,
        full_text=markdown_content,
        metadata={
            "char_count": len(markdown_content),
            "section_count": len(sections),
            "definition_count": len(definitions),
            "cross_reference_count": len(cross_refs),
        },
    )


# =============================================================================
# Formatting for Agent
# =============================================================================


def format_structure_for_agent(structure: DocumentStructure, compact: bool = False) -> str:
    """
    Format a document structure as a concise summary for the AI agent.

    Returns the TOC, definitions, and metadata — NOT the full text.
    The agent can then request specific sections via read_section.

    Args:
        compact: When True, emit only title/type/TOC (no definitions or cross-refs).
                 Used when many documents are present to keep the prompt small.
    """
    lines: list[str] = []
    lines.append(f"📄 {structure.title}")
    lines.append(f"   Type: {structure.document_type}")

    if structure.parties:
        lines.append(f"   Parties: {', '.join(structure.parties[:5])}")

    if structure.date:
        lines.append(f"   Date: {structure.date}")

    lines.append(
        f"   Size: {structure.metadata.get('char_count', 0):,} chars, "
        f"{len(structure.sections)} sections"
    )

    toc_limit = 15 if compact else 30
    if structure.table_of_contents:
        lines.append("   Table of Contents:")
        for item in structure.table_of_contents[:toc_limit]:
            lines.append(f"     § {item['section']} — {item['title']}")

    if not compact:
        if structure.definitions:
            lines.append(f"   Defined Terms ({len(structure.definitions)}):")
            for term in sorted(structure.definitions.keys())[:20]:
                defn = structure.definitions[term][:80]
                lines.append(f"     • \"{term}\" — {defn}...")

        if structure.cross_references:
            lines.append(f"   Cross-References ({len(structure.cross_references)}):")
            seen: set[str] = set()
            for ref in structure.cross_references[:15]:
                ref_text = ref.get("text", "")
                if ref_text and ref_text not in seen:
                    seen.add(ref_text)
                    lines.append(f"     → {ref_text}")

    return "\n".join(lines)
