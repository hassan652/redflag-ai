# Agentic File Search

> **Based on**: [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer) — The original CLI agent for filesystem exploration.

An AI-powered document intelligence agent that explores files like a senior lawyer would — scanning strategically, reasoning about document hierarchies, following cross-references, and verifying its own citations. Unlike traditional RAG systems that rely on pre-computed embeddings, this agent dynamically navigates documents to find answers.

## Why Agentic Search?

Traditional RAG (Retrieval-Augmented Generation) has limitations:
- **Chunks lose context** — Splitting documents destroys relationships between sections
- **Cross-references are invisible** — "See Exhibit B" means nothing to embeddings
- **Similarity ≠ Relevance** — Semantic matching misses logical connections
- **No document hierarchy** — RAG treats a master agreement the same as an exhibit

This system uses a **multi-phase strategy**:
1. **Ingest & Analyze** — Build document hierarchy, cross-reference map, and amendment chains
2. **Strategic Reading** — Query router selects the optimal reading strategy per question type
3. **Section-Aware Navigation** — Read specific sections, not entire documents
4. **Cross-Reference Resolution** — Follow references to related documents automatically
5. **Citation Verification** — Every claim is checked against source documents before output



## Features

- 🔍 **7 Tools**: `scan_folder`, `preview_file`, `parse_file`, `read_section`, `read`, `grep`, `glob`
- 📄 **Document Support**: PDF, DOCX, PPTX, XLSX, HTML, Markdown (via Docling)
- 🤖 **Multi-Provider LLM**: Groq (Llama 4 Scout, Llama 3.3 70B) + Google Gemini 3 Flash
- 🧠 **Query Router**: Classifies questions into 8 types (point lookup, risk analysis, timeline, comparison, compliance, summary, relationship, general) with optimized strategies per type
- 📐 **Document Hierarchy**: Automatically detects master agreements → amendments → schedules → exhibits
- 🔗 **Cross-Reference Map**: Extracts and resolves "See Exhibit A", "as defined in...", etc.
- 📝 **Amendment Tracking**: Detects superseded clauses and amendment chains
- ✅ **Citation Verification**: Post-answer verification against actual source text
- 💰 **Cost Efficient**: ~$0.001 per query with token tracking
- 🌐 **Web UI**: Real-time WebSocket streaming interface with playground mode
- 🐳 **Docker Ready**: Production Dockerfile with health checks

## Installation

```bash
# Clone the repository
git clone https://github.com/PromtEngineer/agentic-file-search.git
cd agentic-file-search

# Install with uv (recommended)
uv sync

# Or with pip
pip install .
```

## Configuration

Create a `.env` file in the project root:

```bash
# At least one provider required
GROQ_API_KEY=your_groq_key_here        # Free at https://console.groq.com
GOOGLE_API_KEY=your_google_key_here     # From https://aistudio.google.com/apikey
```

## Usage

### CLI

```bash
# Basic query (uses Groq by default)
uv run explore --task "What is the purchase price in data/test_acquisition/?"

# Use a specific provider
uv run explore --task "What are the risks?" --provider gemini

# Multi-document query
uv run explore --task "Look in data/large_acquisition/. What are all the financial terms including adjustments and escrow?"
```

### Web UI

```bash
# Start the server
uv run explore-ui

# Open http://localhost:8000 in your browser
```

The web UI provides:
- Folder browser to select target directory
- Real-time step-by-step execution log with agent reasoning
- Document ingestion progress (hierarchy, cross-references)
- Final answer with verified citations
- Token usage and cost statistics

### Docker

```bash
# Build and run
docker compose up

# Or manually
docker build -t casepilot .
docker run -p 8000:8000 --env-file .env casepilot
```

## Architecture

```
User Query
    ↓
┌──────────────────┐
│  Query Router    │ → Classifies into 8 strategy types
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Workspace Ingest │ → Hierarchy + Cross-refs + Amendments
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Workflow Engine   │ ←→ LlamaIndex Workflows (event-driven)
└────────┬─────────┘
         ↓
┌──────────────────┐
│     Agent        │ ←→ Groq/Gemini (structured JSON)
└────────┬─────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ scan_folder │ preview │ parse │ read_section │ read │ grep │ glob │
└─────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐
│ Citation Verifier│ → Checks claims against source docs
└──────────────────┘
         ↓
   Document Parser (Docling - local, no cloud upload)
```

See [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md) for detailed diagrams.

## Test Documents

The repo includes test document sets for evaluation:

- `data/test_acquisition/` — 10 interconnected legal documents
- `data/large_acquisition/` — 25 documents with extensive cross-references

Example queries:
```bash
# Simple (single doc)
uv run explore --task "Look in data/test_acquisition/. Who is the CTO?"

# Cross-reference required
uv run explore --task "Look in data/test_acquisition/. What is the adjusted purchase price?"

# Multi-document synthesis
uv run explore --task "Look in data/large_acquisition/. What happens to employees after the acquisition?"
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (Llama 4 Scout / 3.3 70B) + Google Gemini 3 Flash |
| Document Parsing | Docling (local, open-source) |
| Orchestration | LlamaIndex Workflows |
| Query Routing | Pattern-based classifier (8 strategy types) |
| CLI | Typer + Rich |
| Web Server | FastAPI + WebSocket |
| Validation | Pydantic |
| Package Manager | uv |

## Project Structure

```
src/fs_explorer/
├── agent.py               # LLM agent, tool registry, system prompt
├── workflow.py            # Event-driven workflow orchestration
├── fs.py                  # File tools: scan, parse, grep, glob
├── models.py              # Pydantic action schemas
├── llm.py                 # Multi-provider LLM abstraction (Gemini, Groq)
├── router.py              # Query classification + strategy selection
├── workspace.py           # Document ingestion pipeline
├── document_structure.py  # TOC, definitions, section parsing
├── hierarchy.py           # Document hierarchy detection
├── reference_map.py       # Cross-reference extraction + resolution
├── versioning.py          # Amendment chain detection
├── verifier.py            # Post-answer citation verification
├── playground.py          # Session management + rate limiting
├── server.py              # FastAPI + WebSocket + playground API
├── main.py                # CLI entry point
└── ui.html                # Single-file web interface
```

## Deployment

```bash
# Production (VPS)
# See deploy/vps-setup.sh for full guide

# Docker
docker compose up -d
```

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
make test

# Lint
make lint

# Type check
make typecheck

# Format
make format
```
uv run ruff check .
```

## License

MIT

## Acknowledgments

- Original concept from [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer)
- Document parsing by [Docling](https://github.com/DS4SD/docling)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/agentic-file-search&type=Date)](https://star-history.com/#PromtEngineer/agentic-file-search&Date)
