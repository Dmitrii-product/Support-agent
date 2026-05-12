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
