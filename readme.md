# mac-dapr-mcp-cb-agent

**A Dapr Agents & Chainlit Chatbot for Dynamic Couchbase DB Interrogation via MCP**

## Overview

This project provides a chat-based user interface (Chainlit) that allows users to interrogate a Couchbase database using natural language. The system leverages Dapr Agents and the Model Context Protocol (MCP) to convert user queries into N1QL (SQL++) and execute them against the database.

For Data context we use three-phase schema discovery script (`discovery.py`) that dynamically maps the bucket structure. This schema is then processed by a `schema_compressor.py` utility to create a compact, LLM-optimized context.

The main application (`app.py`) uses this compressed context to provide accurate, schema-aware answers to user questions.

## Architecture and Components

The system consists of three primary Python files and a file-based caching system.

1.  **`app.py` (Main Application)**
    * Runs the Chainlit UI server and manages user interaction.
    * Manages a **3-step cache logic** for schema initialization:
        1.  Loads the compressed schema (`_compressed.json`) if it exists.
        2.  If missing, loads the full schema (`_complete.json`) and runs compression.
        3.  If both are missing, runs the full discovery and compression process from scratch.
    * Initializes two Dapr Agents:
        * **LLM Agent:** Converts natural language to N1QL using the compressed schema context.
        * **Tools Agent:** Receives the N1QL and executes it against Couchbase using MCP tools.
    * Includes a user-driven `retry` mechanism and session-based metrics logging (`EnhancedPerformanceLogger`).

2.  **`discovery.py` (Schema Discovery Engine)**
    * A helper application run by `app.py` when no cache is found.
    * Executes a **three-phase process (P1-P2-P3)** to create a full schema map:
        * **Phase 1:** Discovers all Scopes and Collections.
        * **Phase 2:** Counts documents in each collection.
        * **Phase 3:** Runs `INFER` on relevant collections to extract the schema.
    * Features critical resilience mechanisms: automatic `retry` for `429 Rate Limit` errors and smart parsing for malformed JSON (using `ast.literal_eval`).

3.  **`schema_compressor.py` (Context Compressor)**
    * A helper utility run after `discovery.py` or on an existing full cache.
    * Compresses the large, complete schema file (`travel-sample_schema_final_complete.json`) into a compact, lightweight context file.
    * Includes logic for smart truncation of long samples (`_truncate_samples`) and full support for multi-schema collections (like `_default`).

---

## Features

* **Dynamic Schema Discovery:** No manual schema configuration needed; the system maps any bucket automatically.
* **3-Step Caching Logic:** Enables fast startups by checking for a compressed file > a full file > running a full discovery.
* **Built-in Resilience:** Automatically handles LLM rate-limiting errors.
* **Smart Parsing:** Capable of parsing non-JSON LLM responses (like Python-list-as-string).
* **Context Compression:** Dramatically reduces the schema size to fit LLM context windows and save costs.
* **Observability:** Every chat session is logged to a separate JSON file in the `logs/` directory, including performance metrics.
* **Enhanced User Experience:** Features real-time progress indicators, in-session query history, and a user-driven retry mechanism.

---

## Prerequisites

This project requires a WSL2 environment on Windows 11 for full networking compatibility.

### System and Environment

* **Operating System**: **Windows 11**.
* **WSL2**: Windows Subsystem for Linux 2.
* **Docker Desktop**: Configured to use the WSL2 backend.

### Software and Services

* **Python**: Version 3.10+ (based on `requirements.txt`).
* **Git**: For cloning the repository.
* **Dapr**: **Dapr v1.16.x** (or newer, compatible with `dapr-agents==0.9.3`).
* **Couchbase DB**: A running instance of Couchbase DB.
* **Couchbase MCP Server**: A running instance of the MCP Server.
* **Redis**: A running Redis instance (for Dapr State Store).

---

## Installation

All commands are intended to be run from within a **WSL2 terminal**.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/eshoval/dapr_cb_mcp_agent.git](https://github.com/eshoval/dapr_cb_mcp_agent.git)
    cd mac-dapr-mcp-cb-agent
    ```
    *(Your path in WSL will look similar to: `/mnt/c/Users/YourUser/path/to/mac-dapr-mcp-cb-agent`)*

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize Dapr (if not done previously):**
    Ensure Dapr is installed and using Docker:
    ```bash
    dapr init
    ```

5.  **Run Dapr Components (Redis):**
    The agents require a Dapr state store. If you don't have Redis running, you can start one locally using Docker:
    ```bash
    docker run --name dapr-redis -d -p 6379:6379 redis
    ```
    * Ensure a `components/conversationstore.yaml` file exists (it is created by `dapr init`).

6.  **Configure the `openai.yaml` component:**
    * Copy the example file:
        ```bash
        cp components/openai.yaml.example components/openai.yaml
        ```
    * **Edit** the `components/openai.yaml` file and enter your Azure OpenAI credentials (key, endpoint, model).

7.  **Configure the Environment (`.env`):**
    * Copy the sample file:
        ```bash
        cp .env.sample .env
        ```
    * **Edit** the `.env` file and update the following required variables:
        * `MCP_SERVER_URL`: The full URL to your running MCP Server.
        * `COUCHBASE_BUCKET`: The name of the Couchbase bucket to query.
        * `DAPR_LLM_COMPONENT_DEFAULT`: Must be "openai" (to match `openai.yaml`).
        * `DAPR_LLM_PROVIDER`: Must be "openai" (for the `DaprChatClient`).

---

## Running the Application

1.  **Start the MCP Server:** (In a separate WSL2 terminal)
    Navigate to your MCP server directory and run it (adjust parameters as needed):
    ```bash
    python3 -m uv run src/mcp_server.py --connection-string='couchbase://localhost' --username='your_user' --password='your_pass' --bucket-name='travel-sample' --read-only-query-mode=true --transport=sse
    ```

2.  **Run the Chatbot Application:** (In your main WSL2 terminal, with the `.venv` activated)
    This command starts the Chainlit UI and the Dapr sidecar for `app.py`.
    ```bash
    dapr run --app-id couchbase-chat --dapr-grpc-port 50002 --resources-path ./components -- chainlit run app.py -w --port 8001      
    ```

3.  **Access the UI:**
    * Open your browser to: `http://localhost:8001`

On the first run, the application will display "Running full discovery process...". This may take several minutes. On subsequent runs, it will load the compressed cache (`_compressed.json`) and start in seconds.

---

## Project Structure
=================

├── .gitignore
├── .env                      # (Created from .env.sample) Environment variables and secrets
├── .env.sample
├── app.py                    # [Main] Chainlit UI & Dapr Agent application
├── discovery.py              # [Helper] P1-P2-P3 Schema Discovery Engine
├── schema_compressor.py      # [Helper] Schema Compression Engine
├── requirements.txt          # Python dependencies

├── components/
│   ├── conversationstore.yaml     # Dapr State Store (Redis)
│   ├── openai.yaml                # (Created from .example) Dapr LLM Component (Azure)
│   └── openai.yaml.example

├── logs/
│   ├── (Auto-generated) app_session_log_...json
│   └── (Auto-generated) discovery_complete_log_...json

├── prompts/
│   ├── llm_error_handling_prompt.txt
│   └── llm_router_prompt.txt

└── schema_context/
    ├── (Auto-generated) travel-sample_schema_final_complete.json
    ├── (Auto-generated) travel-sample_schema_final_compressed.json
    └── (Auto-generated) infer_results_full/
        └── (Optional raw .txt files)