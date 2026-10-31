# RagApp

A minimal **Retrieval-Augmented Generation (RAG)** pipeline built from scratch in about 50 lines of Python, with no LangChain or LlamaIndex. It answers questions about a document by finding the most relevant passages and passing them to a local LLM.

I built it to learn how RAG works under the hood: chunking, embeddings, vector search and grounded prompting.

## How it works

```
docs/*.md ──► chunk by paragraph ──► embed (nomic-embed-text) ──► vectors in memory
                                                                         │
question ──► embed ──► cosine similarity vs. every chunk ──► top 3 chunks
                                                                         │
                         prompt = instructions + top 3 chunks + question ┘
                                                                         │
                                                  qwen3:4b-instruct ──► answer
```

1. **Chunk.** Every Markdown file in `docs/` is split into paragraphs. Each paragraph is one chunk.
2. **Embed.** Each chunk becomes a vector using `nomic-embed-text`.
3. **Retrieve.** The question is embedded the same way and compared with every chunk using cosine similarity (1 = most similar, 0 = least). The 3 closest chunks are kept.
4. **Generate.** The chunks and the question go to `qwen3:4b-instruct` with a grounding prompt. It must answer only from the context and say *"I don't know"* when the answer isn't there, so it can't make things up.

## Tech stack

| Part | Choice | Why |
|---|---|---|
| Runtime | [Ollama](https://ollama.com) | Runs models locally: free, no API key, documents never leave your machine |
| Embedding model | `nomic-embed-text` | Small, fast, strong open embedding model |
| Chat model | `qwen3:4b-instruct` | Lightweight instruction model that runs on a laptop |
| Vector search | NumPy | Cosine similarity written by hand, no vector database |
| Packaging | [uv](https://docs.astral.sh/uv/) | Reproducible installs via `pyproject.toml` + `uv.lock` |

## Getting started

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) running locally (default `http://localhost:11434`).

```bash
# 1. Pull the models
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct

# 2. Install dependencies
uv sync            # or: pip install -r requirements.txt

# 3. Run (from the project root, so docs/ is found)
uv run rag.py
```

Then ask questions at the prompt:

```
Ask: What time does the shop open on Saturdays?

The shop opens at 8:00 AM on Saturdays.

Ask: What is the store's Wi-Fi password?

I don't know.
```

To use your own documents, put Markdown files in `docs/`. Paragraphs (separated by blank lines) become chunks.

## The prompt

```
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
<top 3 chunks retrieved for the question>

Question:
<user question>
```

## Project structure

```
RagApp/
├── rag.py            # The whole pipeline: load, chunk, embed, retrieve, prompt
├── docs/
│   └── handbook.md   # Sample data: a fictional coffee shop employee handbook
├── pyproject.toml    # Dependencies (numpy, requests)
├── uv.lock           # Locked versions
└── requirements.txt  # pip alternative (generated with uv export)
```

## Limitations

- Embeddings are recomputed on every run (nothing is cached).
- Chunking is by paragraph only; no overlap or size control.
- Vectors live in memory, which is fine for a handbook but not for thousands of documents.
- No measurement yet of how often retrieval finds the right chunk.

## Roadmap

- [ ] **Retrieval eval:** a set of test questions, each with its correct paragraph, to measure top-3 hit rate
- [ ] **Citations:** show which chunks each answer used
- [ ] **Cache embeddings** to disk (`.npy`), then move them into PostgreSQL + pgvector
- [ ] **Chunking experiments:** sentence vs. paragraph vs. overlapping windows, compared with the eval
