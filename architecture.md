# Smart Support Triage Agent Architecture

This document describes the design and components of the LangGraph-based Smart Support Triage Agent.

## Overview
The system is built as a multi-agent workflow using **LangGraph**. It utilizes an LLM (Qwen 32B via Groq) to analyze incoming user queries and decide on the best course of action. If a query requires external knowledge, it uses Retrieval-Augmented Generation (RAG) backed by a **ChromaDB** vector store.

## Components

### 1. State Management (`src/state.py`)
A `TypedDict` structure acts as the source of truth passed between nodes. It contains:
- `messages`: The list of conversational messages (History).
- `triage_route`: The routing decision string ("researcher" or "responder").
- `retrieved_context`: Document content retrieved by the RAG system.
- `final_response`: The finalized text to be delivered to the user.

### 2. Triage Node (`src/nodes/triage.py`)
This is the entry point. It uses the LLM to inspect the user's latest message and context.
- **Goal**: Decide if the question requires looking up documentation ("researcher") or if it can be answered directly ("responder").
- **Output**: Updates the `triage_route` state.

### 3. Researcher Node (`src/nodes/researcher.py`)
This node is triggered if the Triage Node decides external knowledge is needed.
- **Goal**: Extract search terms from the query, embed them, and query ChromaDB for the top-k most relevant documents.
- **Output**: Updates the `retrieved_context` state.

### 4. Responder Node (`src/nodes/responder.py`)
This is the final generative step.
- **Goal**: Use the user's query and the `retrieved_context` (if any) to generate a helpful, polite, and accurate support response using the LLM.
- **Output**: Updates the `final_response` and appends an `AIMessage` to `messages`.

### 5. Vector Store (`src/vectorstore.py`)
Handles the connection to ChromaDB. We use an embedding model to convert text documents into vectors. (Initially set up with mock data/scripts for testing).

## Workflow (Graph Routing)
1. **User Query** -> Triage Node
2. **Triage Node** sets `triage_route`:
   - If `researcher` -> proceeds to Researcher Node
   - If `responder` -> proceeds to Responder Node
3. **Researcher Node** -> proceeds to Responder Node
4. **Responder Node** -> END.

## Vector Store Management (`update_db.py`)
To incrementally update the ChromaDB vector store without duplicating existing documents, use the `update_db.py` script. It automatically scans the `chroma_db` directory for new `.txt` files and computes a SHA-256 hash of the document content to use as deterministic chunk IDs.
- Usage: `python update_db.py`

## Observability (LangSmith)
The application utilizes **LangSmith** for native tracing and observability of the LangGraph workflow. To enable, the following environment variables must be set in `.env`:
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your_key>`
- `LANGCHAIN_PROJECT=<project_name>`

## Evaluations (DeepEval + Pytest)
To ensure the LLM outputs meet quality standards, we employ **DeepEval** alongside **Pytest**. 
- Evaluation tests are located in `tests/evals/`.
- We use a custom LLM wrapper (`CustomGroqLLM` in `tests/evals/custom_llm.py`) so DeepEval utilizes the configured Groq model (`llama-3.3-70b-versatile`) to score outputs natively.
- Tests are parametrized against the **Golden Dataset** (`tests/evals/golden_dataset.json`). Each golden entry is run through the full agent pipeline and scored.
- Key metrics measured:
  - **Answer Relevancy** — Does the response directly address the user's question?
  - **Faithfulness** — Is the response factually grounded in the retrieved context?
  - **Contextual Relevancy** — Is the retrieved context relevant to the query?
  - **Hallucination** — Does the response contain fabricated information not in the context?
- Run evaluations via: `.venv/bin/pytest tests/evals/test_rag.py -v`
- Note: Due to Groq free-tier rate limits (12K TPM), a 65-second pause is applied between test cases.

### Golden Dataset Generation (`tests/evals/generate_goldens.py`)
To create a synthetic "Golden Dataset" of Q&A pairs from your documentation, run:
- Usage: `.venv/bin/python tests/evals/generate_goldens.py`
- The script scans `chroma_db/` for `.txt` files and uses DeepEval's `Synthesizer` (powered by `llama-3.3-70b-versatile` via Groq) to generate question-answer pairs.
- Per-file generation counts are configurable in the `custom_counts` dict inside the script. Current settings:
  - `dct-version.txt`: 10 pairs
  - `Mango_analytics.txt`: 20 pairs
  - `email_tracking_setup_detailed.txt`: 5 pairs
  - All other files: 2 pairs (default)
- Output: `tests/evals/golden_dataset.json`

## 6. Web Interface & Lead Capture
The project now includes a web interface and lead capture capabilities:
- **Backend**: A FastAPI application (`api.py`) serves the static frontend and provides a `POST /chat` API endpoint. It maintains an in-memory session history for context continuity.
- **Frontend**: A modern UI (`static/index.html`, `static/style.css`, `static/app.js`) featuring a responsive landing page and a floating chat widget. Navigation links point to several mock pages (`features.html`, `about.html`, `contact.html`).
- **Lead Capture & Escalation**: 
  - The `responder` node has been upgraded to support tool calling.
  - A `leads.db` SQLite database (`src/database.py`) securely stores captured leads in the `leads` table and support tickets in the `tickets` table.
  - When users show intent to buy, the agent explicitly asks for their details and consent before using the `save_lead_tool` to insert the data into the database.
  - When the agent cannot resolve a customer's issue, it asks for their details and uses the `escalate_to_human_tool` to log a support ticket, providing the user with a tracking number (e.g., TKT-1) and promising human follow-up.
