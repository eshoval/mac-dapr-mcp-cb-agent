# mac-dapr-mcp-cb-agent

**A Dapr Agents & Chainlit Chatbot for Dynamic Couchbase DB Interrogation via MCP**

## Overview

This project provides a chat-based user interface (Chainlit) that allows users to interrogate a Couchbase database using natural language. The system leverages Dapr Agents and the Model Context Protocol (MCP) to convert user queries into N1QL (SQL++) and execute them against the database.

For Data context, the system uses a robust, **modular discovery engine** that dynamically maps the bucket structure. This schema is then processed by a compressor utility to create a compact, LLM-optimized context.

The main application (`app.py`) uses this compressed context to provide accurate, schema-aware answers to user questions.

## Architecture and Components

The system is built on a modular architecture designed for resilience and separation of concerns.

### 1. Main Application (`app.py`)
* **Role:** Runs the Chainlit UI server and manages user interaction.
* **Caching Logic:** Manages a 3-step initialization:
    1.  Load compressed cache (`_compressed.json`).
    2.  Load full cache (`_complete.json`) & compress.
    3.  Run full discovery (`discovery.py`) if no cache exists.
* **Agents:** Initializes `LLM Agent` (Planner) and `Tools Agent` (Executor) with specific timeouts.
* **Observability:** Uses `EnhancedPerformanceLogger` for non-blocking, async logging of session metrics.

### 2. Schema Discovery Engine (Modular)
The discovery process has been refactored into a clean, multi-file architecture to ensure stability and ease of maintenance:

* **`discovery.py` (Orchestrator):**
    * The entry point for schema discovery.
    * Manages the business logic flow: Phase 1 (Scopes) → Phase 2 (Counts) → Phase 3 (Inference) → Phase 4 (Indexes).
    * Implements **Adaptive Inference**: Dynamically adjusts `INFER` parameters (sample size, similarity) based on collection size to prevent LLM timeouts on small/large collections.

* **`mcp_executor.py` (Execution Layer):**
    * Handles all interactions with the Dapr Agent.
    * Implements **Robust Retry Logic**: Automatic handling of `429` (Rate Limit) and `503` errors with exponential backoff.
    * Manages prompt construction for tool execution.

* **`response_parser.py` (Parsing Layer):**
    * A specialized parser for handling unpredictable LLM outputs.
    * **Multi-layer Strategy**: Tries standard JSON -> Fenced JSON -> `ast.literal_eval` -> Tool wrapper unwrapping.
    * **Auto-Repair**: Detects and fixes truncated JSON responses (e.g., unclosed brackets/strings).

* **`file_manager.py` (I/O Layer):**
    * Centralizes all file operations (JSON load/save, Text save).
    * Ensures directory structures exist and logs all I/O activities.

### 3. Schema Compressor (`schema_compressor.py`)
* A utility that runs after discovery.
* Compresses the massive JSON schema into a lightweight format token-optimized for LLMs.
* Features smart string truncation and multi-schema support for collections with mixed document types.

---

## Key Features

* **Modular Architecture:** Separation of concerns (Execution, Parsing, I/O, Logic) makes the system highly maintainable.
* **Resilient Discovery:**
    * **Auto-Retry:** Handles API rate limits and temporary failures automatically.
    * **Smart Parsing:** Can recover data even if the LLM returns malformed or truncated JSON.
    * **Adaptive Config:** Uses small sample sizes for small collections and larger ones for big datasets.
* **3-Step Caching:** Instant startup if cache exists; automatic recovery if it doesn't.
* **Async Logging:** Continuous, non-blocking logging of all events and metrics to `logs/` directory.
* **User Experience:** Real-time progress updates in the chat UI ("Executing query...", "Retrying...", etc.).

---

## Prerequisites

This project requires a WSL2 environment on Windows 11 for full networking compatibility.

### System and Environment
* **OS:** Windows 11 (WSL2).
* **Docker Desktop:** Configured with WSL2 backend.
* **Python:** 3.10+.

### Dapr & Services
* **Dapr:** v1.16.x+ (compatible with `dapr-agents`).
* **Couchbase DB:** Running instance.
* **MCP Server:** Running Couchbase MCP server instance.
* **Redis:** Running Redis instance (for Dapr State Store).

---

## Installation & Configuration

1.  **Clone & Setup:**
    ```bash
    git clone <repo_url>
    cd mac-dapr-mcp-cb-agent
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Dapr Init:**
    ```bash
    dapr init
    ```

3.  **Configure Environment:**
    * Copy `.env.sample` to `.env`.
    * Update `MCP_SERVER_URL`, `COUCHBASE_BUCKET`, and Dapr settings.
    * Ensure `components/openai.yaml` is configured with your LLM credentials (Azure OpenAI or Standard OpenAI).

---

## Running the Application

### 1. Start MCP Server
(In a separate terminal)
```bash
python3 -m uv run src/mcp_server.py --transport=sse ...

2. Run the Agent
(In main terminal)
dapr run --app-id couchbase-chat --dapr-grpc-port 50002 --resources-path ./components -- chainlit run app.py --port 8001

3. Access UI
Open http://localhost:8001 in your browser.

Project Structure

├── app.py                    # [Main] Chainlit UI & Dapr Agent application
├── discovery.py              # [Orchestrator] Schema Discovery Business Logic
├── mcp_executor.py           # [Module] Tool Execution & Retry Logic
├── response_parser.py        # [Module] JSON Parsing & Repair Logic
├── file_manager.py           # [Module] File I/O Operations
├── performance_logger.py     # [Module] Logging System
├── schema_compressor.py      # [Utility] Schema Compression

├── components/               # Dapr component definitions
│   ├── conversationstore.yaml
│   └── openai.yaml

├── logs/                     # Session and Discovery logs
├── prompts/                  # LLM System Prompts
└── schema_context/           # Generated Cache Files
    ├── travel-sample_schema_final_complete.json
    └── travel-sample_schema_final_compressed.json

