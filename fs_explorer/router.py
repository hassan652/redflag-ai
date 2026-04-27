"""
Query-type classification and strategy selection.

Different legal questions require completely different reading strategies.
A point lookup ("What is the purchase price?") needs 2-3 documents.
A risk analysis ("What are all the risks?") needs every document.
This router classifies the query and returns the right strategy.
"""

from enum import Enum
from dataclasses import dataclass, field
import re


# =============================================================================
# Query Types
# =============================================================================


class QueryType(Enum):
    """Types of legal document queries, each with a different optimal strategy."""

    POINT_LOOKUP = "point_lookup"  # "What is X?" → find one specific fact
    RISK_ANALYSIS = "risk_analysis"  # "What are the risks?" → comprehensive scan
    COMPARISON = "comparison"  # "Compare X across documents" → multi-doc extraction
    TIMELINE = "timeline"  # "What happened when?" → date extraction
    COMPLIANCE_CHECK = "compliance_check"  # "Does this comply?" → checklist
    SUMMARY = "summary"  # "Summarize this deal" → full read + synthesize
    RELATIONSHIP = "relationship"  # "How does X relate to Y?" → cross-reference tracing
    GENERAL = "general"  # Catch-all for unclassified queries


# =============================================================================
# Classification
# =============================================================================

# Keyword patterns for each query type
_QUERY_PATTERNS: list[tuple[QueryType, list[str]]] = [
    (
        QueryType.RISK_ANALYSIS,
        [
            r"\brisk",
            r"\bliabilit",
            r"\bexposure",
            r"\bindemnif",
            r"\bwarrant",
            r"\brepresentation",
            r"\bdefault",
            r"\bbreach",
            r"\btermination\s+(?:right|clause|provision)",
            r"\bpending\s+(?:litigation|claim|dispute)",
            r"\bmaterial\s+adverse",
            r"\bconcern",
            r"\bproblem",
            r"\bissue",
            r"\bred\s*flag",
        ],
    ),
    (
        QueryType.COMPARISON,
        [
            r"\bcompar",
            r"\bdiff(?:er|erence)",
            r"\bsame\b.*\bacross\b",
            r"\bconsisten",
            r"\bdeviat",
            r"\bside.by.side",
            r"\beach\s+(?:contract|agreement|document)",
            r"\ball\s+(?:contracts|agreements)\b",
            r"\bvs\b",
            r"\bversus\b",
        ],
    ),
    (
        QueryType.TIMELINE,
        [
            r"\btimeline",
            r"\bchronolog",
            r"\bwhen\s+(?:did|was|were|is|are)",
            r"\bsequence\s+of\s+events",
            r"\bmilestone",
            r"\bdeadline",
            r"\bschedule\s+of\s+(?:events|dates)",
            r"\bkey\s+dates",
            r"\bbefore\s+closing",
            r"\bafter\s+closing",
            r"\bpost.closing",
            r"\bpre.closing",
        ],
    ),
    (
        QueryType.COMPLIANCE_CHECK,
        [
            r"\bcompl(?:y|iance|iant)",
            r"\bsatisf(?:y|ied)",
            r"\bmeet\s+(?:the\s+)?(?:requirement|condition|obligation)",
            r"\bin\s+accordance\s+with",
            r"\bcondition\s+precedent",
            r"\bready\s+to\s+close",
            r"\bclosing\s+(?:condition|requirement)",
            r"\bchecklist",
        ],
    ),
    (
        QueryType.SUMMARY,
        [
            r"\bsummar",
            r"\boverview",
            r"\bcomplete\s+picture",
            r"\bfull\s+(?:picture|analysis|review)",
            r"\btell\s+me\s+everything",
            r"\bwhat\s+(?:does|is)\s+this\s+deal",
            r"\bkey\s+terms",
            r"\bmain\s+(?:terms|provisions|points)",
            r"\bhigh.level",
        ],
    ),
    (
        QueryType.RELATIONSHIP,
        [
            r"\brelat(?:e|ion|ionship)",
            r"\bhow\s+does\s+.+\s+(?:relate|connect|link)",
            r"\bconnect(?:ion|ed)",
            r"\breference(?:s|d)?\s+(?:to|by|in|from)",
            r"\bcross.reference",
            r"\btrace\b",
            r"\bfollow\s+the\s+reference",
        ],
    ),
    (
        QueryType.POINT_LOOKUP,
        [
            r"\bwhat\s+is\s+the\b",
            r"\bwhat\s+are\s+the\b",
            r"\bhow\s+much\b",
            r"\bwho\s+is\b",
            r"\bwho\s+are\b",
            r"\bwhen\s+is\b",
            r"\bwhere\s+is\b",
            r"\bpurchase\s+price\b",
            r"\btotal\s+(?:amount|value|consideration)",
            r"\bname\s+(?:of|the)\b",
            r"\bspecific\b",
        ],
    ),
]


def classify_query(question: str) -> QueryType:
    """
    Classify a user's question into a QueryType.

    Uses keyword pattern matching. Each pattern that matches adds a vote
    for that query type. The type with the most votes wins.

    Args:
        question: The user's natural language question.

    Returns:
        The classified QueryType.
    """
    scores: dict[QueryType, int] = {qt: 0 for qt in QueryType}
    question_lower = question.lower()

    for query_type, patterns in _QUERY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, question_lower):
                scores[query_type] += 1

    # Get the type with the highest score
    best_type = max(scores, key=lambda x: scores[x])

    # If no patterns matched, default to GENERAL
    if scores[best_type] == 0:
        return QueryType.GENERAL

    return best_type


# =============================================================================
# Strategy Configuration
# =============================================================================


@dataclass
class QueryStrategy:
    """Configuration for how to handle a specific query type."""

    query_type: QueryType
    max_docs_to_read: int | None  # None = read all
    read_strategy: str  # "section_targeted", "comprehensive", "extract_matching"
    output_format: str  # "single_answer", "risk_matrix", "comparison_table", "timeline"
    verify_citations: bool = True
    sections_to_focus: list[str] = field(default_factory=list)
    agent_instructions: str = ""  # Extra instructions appended to the prompt


_STRATEGIES: dict[QueryType, QueryStrategy] = {
    QueryType.POINT_LOOKUP: QueryStrategy(
        query_type=QueryType.POINT_LOOKUP,
        max_docs_to_read=5,
        read_strategy="section_targeted",
        output_format="single_answer",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Point Lookup\n"
            "- This is a specific factual question. Find the answer efficiently.\n"
            "- Start with the master agreement. Check the table of contents for relevant sections.\n"
            "- Use read_section to read only the specific sections you need.\n"
            "- Check for amendments that might modify the relevant sections.\n"
            "- You should NOT need to read more than 3-5 documents."
        ),
    ),
    QueryType.RISK_ANALYSIS: QueryStrategy(
        query_type=QueryType.RISK_ANALYSIS,
        max_docs_to_read=None,
        read_strategy="comprehensive",
        output_format="risk_matrix",
        verify_citations=True,
        sections_to_focus=[
            "indemnification",
            "representations",
            "warranties",
            "liability",
            "termination",
            "default",
            "remedies",
            "limitation",
            "risk",
            "material adverse",
        ],
        agent_instructions=(
            "STRATEGY: Risk Analysis\n"
            "- Read ALL documents systematically — risks can hide anywhere.\n"
            "- Focus on: indemnification, reps & warranties, termination, default,\n"
            "  liability caps, material adverse change, and pending litigation.\n"
            "- For each risk found, note: severity (High/Medium/Low), source, and\n"
            "  any mitigation measures mentioned.\n"
            "- Check disclosure schedules — they contain exceptions to representations.\n"
            "- Structure your answer as a risk matrix with categories."
        ),
    ),
    QueryType.COMPARISON: QueryStrategy(
        query_type=QueryType.COMPARISON,
        max_docs_to_read=None,
        read_strategy="extract_matching",
        output_format="comparison_table",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Document Comparison\n"
            "- Identify which documents contain the clause/term being compared.\n"
            "- Extract the SAME section/clause from each relevant document.\n"
            "- Present findings in a comparison table format.\n"
            "- Highlight deviations from standard or majority terms.\n"
            "- Note any documents that are MISSING the clause entirely."
        ),
    ),
    QueryType.TIMELINE: QueryStrategy(
        query_type=QueryType.TIMELINE,
        max_docs_to_read=None,
        read_strategy="comprehensive",
        output_format="timeline",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Timeline Construction\n"
            "- Extract ALL dates, deadlines, and milestones from every document.\n"
            "- Pay special attention to: signing dates, effective dates, closing dates,\n"
            "  condition deadlines, post-closing obligations, and survival periods.\n"
            "- Present findings in chronological order.\n"
            "- Note any conflicting dates between documents.\n"
            "- Flag any upcoming or past-due deadlines."
        ),
    ),
    QueryType.COMPLIANCE_CHECK: QueryStrategy(
        query_type=QueryType.COMPLIANCE_CHECK,
        max_docs_to_read=None,
        read_strategy="comprehensive",
        output_format="checklist",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Compliance Check\n"
            "- Identify ALL conditions, requirements, or obligations.\n"
            "- For each condition, determine: status (satisfied/pending/not met),\n"
            "  responsible party, deadline, and supporting evidence.\n"
            "- Check the closing checklist if one exists.\n"
            "- Present as a checklist with status indicators."
        ),
    ),
    QueryType.SUMMARY: QueryStrategy(
        query_type=QueryType.SUMMARY,
        max_docs_to_read=None,
        read_strategy="comprehensive",
        output_format="structured_summary",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Comprehensive Summary\n"
            "- Start with the master agreement to understand the overall deal.\n"
            "- Read all supporting documents to build a complete picture.\n"
            "- Structure your summary with clear sections: Deal Overview, Key Terms,\n"
            "  Financial Terms, Conditions, Risks, and Open Items.\n"
            "- Include all key numbers, dates, and party names with citations."
        ),
    ),
    QueryType.RELATIONSHIP: QueryStrategy(
        query_type=QueryType.RELATIONSHIP,
        max_docs_to_read=None,
        read_strategy="cross_reference_tracing",
        output_format="relationship_map",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: Relationship Tracing\n"
            "- Use the pre-built reference map to identify connections.\n"
            "- Follow all cross-references from and to the target document(s).\n"
            "- Map out how documents depend on each other.\n"
            "- Present findings showing the chain of references."
        ),
    ),
    QueryType.GENERAL: QueryStrategy(
        query_type=QueryType.GENERAL,
        max_docs_to_read=10,
        read_strategy="adaptive",
        output_format="single_answer",
        verify_citations=True,
        agent_instructions=(
            "STRATEGY: General Query\n"
            "- Scan documents to assess relevance, then deep-read the most relevant ones.\n"
            "- Follow cross-references as needed.\n"
            "- Provide a clear answer with citations."
        ),
    ),
}


def get_strategy(query_type: QueryType) -> QueryStrategy:
    """
    Get the strategy configuration for a query type.

    Args:
        query_type: The classified query type.

    Returns:
        A QueryStrategy with reading and output configuration.
    """
    return _STRATEGIES.get(query_type, _STRATEGIES[QueryType.GENERAL])


def format_strategy(query_type: QueryType, strategy: QueryStrategy) -> str:
    """Format the strategy as context for the AI agent."""
    lines: list[str] = []
    lines.append(f"🎯 QUERY CLASSIFICATION: {query_type.value.upper()}")
    lines.append(f"   Reading Strategy: {strategy.read_strategy}")
    lines.append(f"   Output Format: {strategy.output_format}")

    if strategy.max_docs_to_read:
        lines.append(f"   Max Documents: {strategy.max_docs_to_read}")
    else:
        lines.append("   Max Documents: ALL (comprehensive scan)")

    if strategy.sections_to_focus:
        lines.append(
            f"   Focus Sections: {', '.join(strategy.sections_to_focus)}"
        )

    lines.append(f"\n{strategy.agent_instructions}")

    return "\n".join(lines)
