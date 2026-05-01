"""
Document hierarchy detection for legal document sets.

Classifies documents by their role (master agreement, amendment, schedule,
exhibit, etc.) and builds a tree showing how they relate. This tree is
shown to the AI agent BEFORE it starts reading, so it knows the most
important documents to read first.
"""

import os
from dataclasses import dataclass, field

from .document_structure import DocumentStructure


# =============================================================================
# Data Model
# =============================================================================


@dataclass
class DocumentNode:
    """A single document's position in the hierarchy."""

    file_path: str
    doc_type: str  # "master", "amendment", "schedule", "exhibit", etc.
    title: str
    priority: int  # 1=master (read first), 2=amendment, 3=schedule/exhibit, 4=supporting
    parent: str | None = None  # File path of the document this modifies/supplements
    children: list[str] = field(default_factory=list)  # File paths that reference this
    superseded_by: str | None = None  # If an amendment replaces this document's terms


@dataclass
class DocumentHierarchy:
    """Complete hierarchy of documents in a workspace."""

    nodes: dict[str, DocumentNode] = field(default_factory=dict)  # file_path -> node
    root_documents: list[str] = field(default_factory=list)  # Master agreements
    reading_order: list[str] = field(default_factory=list)  # Suggested reading order


# =============================================================================
# Priority Mapping
# =============================================================================

# Lower number = higher priority (read first)
TYPE_PRIORITIES: dict[str, int] = {
    "master": 1,
    "amendment": 2,
    "disclosure": 2,
    "schedule": 3,
    "exhibit": 3,
    "escrow": 3,
    "financial": 3,
    "report": 3,
    "checklist": 3,
    "opinion": 4,
    "certificate": 4,
    "regulatory": 4,
    "consent": 4,
    "nda": 4,
    "employment": 5,
    "ip": 5,
    "side_letter": 5,
    "unknown": 6,
}


# =============================================================================
# Hierarchy Building
# =============================================================================


def classify_document(structure: DocumentStructure) -> DocumentNode:
    """
    Create a DocumentNode from a parsed DocumentStructure.

    Uses the document type already classified in the structure,
    and assigns a priority for reading order.
    """
    priority = TYPE_PRIORITIES.get(structure.document_type, 6)

    return DocumentNode(
        file_path=structure.file_path,
        doc_type=structure.document_type,
        title=structure.title,
        priority=priority,
    )


def build_hierarchy(
    structures: dict[str, DocumentStructure],
) -> DocumentHierarchy:
    """
    Build a document hierarchy from all parsed structures.

    1. Classifies each document by type
    2. Identifies master/root documents
    3. Links amendments, schedules, and exhibits to their parent
    4. Produces a suggested reading order

    Args:
        structures: Dict mapping file_path -> DocumentStructure.

    Returns:
        A DocumentHierarchy with all relationships mapped.
    """
    hierarchy = DocumentHierarchy()

    # Step 1: Create nodes for all documents
    for file_path, structure in structures.items():
        node = classify_document(structure)
        hierarchy.nodes[file_path] = node

    # Step 2: Identify root documents (master agreements)
    for file_path, node in hierarchy.nodes.items():
        if node.doc_type == "master":
            hierarchy.root_documents.append(file_path)

    # If no master agreement found, treat the highest-priority document as root
    if not hierarchy.root_documents:
        sorted_nodes = sorted(hierarchy.nodes.items(), key=lambda x: x[1].priority)
        if sorted_nodes:
            hierarchy.root_documents.append(sorted_nodes[0][0])

    # Step 3: Link children to parents using cross-references
    for file_path, structure in structures.items():
        node = hierarchy.nodes[file_path]

        # Amendments, schedules, exhibits are children of the master
        if node.doc_type in ("amendment", "schedule", "exhibit", "disclosure"):
            # Try to find which master document they reference
            for ref in structure.cross_references:
                ref_text = ref.get("text", "").lower()
                for root_path in hierarchy.root_documents:
                    root_title = structures[root_path].title.lower()
                    if any(
                        word in ref_text
                        for word in root_title.split()
                        if len(word) > 3
                    ):
                        node.parent = root_path
                        hierarchy.nodes[root_path].children.append(file_path)
                        break
                if node.parent:
                    break

            # If no parent found via cross-reference, default to first master
            if node.parent is None and hierarchy.root_documents:
                node.parent = hierarchy.root_documents[0]
                hierarchy.nodes[hierarchy.root_documents[0]].children.append(file_path)

    # Step 4: Build reading order (BFS from roots, sorted by priority)
    visited: set[str] = set()
    queue: list[str] = []

    # Start with root documents
    for root in sorted(hierarchy.root_documents):
        if root not in visited:
            queue.append(root)
            visited.add(root)

    # BFS: root -> amendments -> schedules/exhibits -> supporting docs
    reading_order: list[str] = []
    while queue:
        current = queue.pop(0)
        reading_order.append(current)

        # Add children sorted by priority
        children = hierarchy.nodes[current].children
        children_sorted = sorted(
            children,
            key=lambda x: hierarchy.nodes[x].priority if x in hierarchy.nodes else 99,
        )
        for child in children_sorted:
            if child not in visited:
                queue.append(child)
                visited.add(child)

    # Add any remaining documents not yet in the reading order
    remaining = [
        fp
        for fp in sorted(
            hierarchy.nodes.keys(),
            key=lambda x: hierarchy.nodes[x].priority,
        )
        if fp not in visited
    ]
    reading_order.extend(remaining)

    hierarchy.reading_order = reading_order
    return hierarchy


# =============================================================================
# Formatting
# =============================================================================


def format_hierarchy(
    hierarchy: DocumentHierarchy,
    structures: dict[str, DocumentStructure],
) -> str:
    """
    Format the document hierarchy as a readable tree for the AI agent.

    Shows the master agreement at the top, with amendments, schedules,
    and exhibits nested underneath.
    """
    lines: list[str] = []
    lines.append("📁 DOCUMENT HIERARCHY")
    lines.append("=" * 50)

    # Show tree from each root
    for root_path in hierarchy.root_documents:
        root_node = hierarchy.nodes[root_path]
        root_name = os.path.basename(root_path)
        root_title = structures[root_path].title if root_path in structures else ""

        lines.append(f"\n  🔵 {root_name}")
        lines.append(f"     [{root_node.doc_type.upper()}] {root_title}")

        # Group children by type
        children_by_type: dict[str, list[str]] = {}
        for child_path in root_node.children:
            if child_path in hierarchy.nodes:
                child_type = hierarchy.nodes[child_path].doc_type
                if child_type not in children_by_type:
                    children_by_type[child_type] = []
                children_by_type[child_type].append(child_path)

        for doc_type, children in sorted(children_by_type.items()):
            for child_path in children:
                child_name = os.path.basename(child_path)
                child_title = (
                    structures[child_path].title if child_path in structures else ""
                )
                lines.append(f"     ├── {child_name}")
                lines.append(
                    f"     │   [{doc_type.upper()}] {child_title[:60]}"
                )

    # Show orphan documents (not connected to any root)
    orphans = [
        fp
        for fp in hierarchy.nodes
        if fp not in hierarchy.root_documents
        and hierarchy.nodes[fp].parent is None
    ]
    if orphans:
        lines.append(f"\n  📋 Other Documents ({len(orphans)}):")
        for fp in orphans:
            node = hierarchy.nodes[fp]
            fname = os.path.basename(fp)
            lines.append(f"     • {fname} [{node.doc_type.upper()}]")

    # Show suggested reading order
    lines.append(f"\n  📖 Suggested Reading Order:")
    for i, fp in enumerate(hierarchy.reading_order[:15], 1):
        fname = os.path.basename(fp)
        node = hierarchy.nodes[fp]
        lines.append(f"     {i}. {fname} (priority: {node.priority})")

    return "\n".join(lines)
