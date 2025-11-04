# app.py version 2.4.0 (Async Logging)
# Integrates EnhancedPerformanceLogger for session-based logging and metrics.
# Adds Query History tracking per session.
# Improves UX with a single, updating progress message for queries.
# Adds Context Window size validation.
# MODIFIED: Implements non-blocking, continuous file logging using asyncio.to_thread.

import os
import time
import json
import chainlit as cl
import asyncio # <-- Ensured asyncio is imported
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from dapr_agents import Agent
from dapr_agents.tool.mcp.client import MCPClient
from dapr_agents.llm.dapr import DaprChatClient

# Import discovery script
from discovery import main_discovery as run_full_discovery
# Import the improved compressor
from schema_compressor import SchemaCompressor

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SCHEMA_CONTEXT_DIR = Path("./schema_context")
LOGS_DIR = Path("./logs") # Ensure logs directory is defined
# Max samples for the compressor (can be overridden by .env)
COMPRESSOR_MAX_SAMPLES = int(os.getenv("COMPRESSOR_MAX_SAMPLES", "2"))

# --- *** MODIFIED: PerformanceLogger Class (v2.4.0 - Async) *** ---
# Base class (from discovery.py)
class PerformanceLogger:
    """Simple in-memory logger for tracking performance with continuous, non-blocking flush."""
    
    def __init__(self, filepath: Path): # <-- MODIFIED: filepath is mandatory
        self.logs = []
        self.log_counter = 0
        self.event_timers = {}
        self.filepath = filepath # <-- NEW: Store the filepath
        # --- NEW: Ensure directory exists on init ---
        self.filepath.parent.mkdir(exist_ok=True)

    def _blocking_write_to_file(self):
        """(BLOCKING) Internal helper to write the entire log list to the file."""
        try:
            # 'w' (write/overwrite) to always have the latest full JSON
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Don't crash the app, just log to console
            print(f"\n❌ CRITICAL: Error flushing log to {self.filepath}: {e}")

    async def log(self, event_name: str, status: str, details: str = "") -> dict: # <-- MODIFIED: async def
        self.log_counter += 1
        timestamp = datetime.now()
        log_entry = {
            "log_id": self.log_counter,
            "timestamp": timestamp.isoformat(),
            "event": event_name,
            "status": status,
            "details": details
        }
        duration_seconds = None
        if status == "Ended" and event_name in self.event_timers:
            start_time = self.event_timers.pop(event_name)
            duration = (timestamp - start_time).total_seconds()
            duration_seconds = round(duration, 3)
            log_entry["duration_seconds"] = duration_seconds
        elif status == "Started":
            if event_name not in self.event_timers:
                self.event_timers[event_name] = timestamp

        self.logs.append(log_entry)
        duration_str = f" ({duration_seconds}s)" if duration_seconds is not None else ""
        # Print to console for real-time monitoring
        print(f"LOG [{log_entry['log_id']}] {timestamp.strftime('%H:%M:%S.%f')[:-3]} | {event_name} | {status}{duration_str} {f'| {details}' if details else ''}")
        
        # --- MODIFIED: Flush to disk in a separate thread ---
        try:
            # Run the blocking I/O operation in a separate thread
            await asyncio.to_thread(self._blocking_write_to_file)
        except Exception as e:
            # This might happen if the loop is shutting down
            print(f"\n❌ CRITICAL: asyncio.to_thread failed for logging: {e}")
            
        return log_entry

    def save_to_file(self) -> Path: # <-- MODIFIED: Removed filename argument
        """Prints the final log path (writing is handled by log())."""
        # This function just provides the final confirmation.
        print(f"\n✓ Session log saved to: {self.filepath}")
        return self.filepath

# NEW: Enhanced logger with query metrics
class EnhancedPerformanceLogger(PerformanceLogger):
    """Extended logger with query metrics"""
    
    def __init__(self, filepath: Path): # <-- MODIFIED: Pass filepath to super
        super().__init__(filepath=filepath) # <-- MODIFIED
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_execution_time": 0.0,
            "retry_count": 0
        }
    
    def log_query_result(self, success: bool, duration: float, retry: bool = False):
        """Track query-specific metrics"""
        self.query_stats["total_queries"] += 1
        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1
        if retry:
            self.query_stats["retry_count"] += 1
        
        # Update running average
        n = self.query_stats["total_queries"]
        current_avg = self.query_stats["avg_execution_time"]
        self.query_stats["avg_execution_time"] = (current_avg * (n - 1) + duration) / n
    
    def get_summary_stats(self) -> dict:
        """Get session summary for final report"""
        success_rate = (self.query_stats["successful_queries"] / 
                        max(self.query_stats["total_queries"], 1)) * 100
        stats = {
            **self.query_stats,
            "success_rate_percent": round(success_rate, 2),
            "avg_execution_time_ms": round(self.query_stats["avg_execution_time"] * 1000, 2)
        }
        del stats["avg_execution_time"] # Remove the seconds-based one
        return stats
# --- *** END EnhancedPerformanceLogger Class *** ---

# ---------- Prompt utilities ----------

def load_template(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Prompt file not found at '{path}'. Please ensure it exists.")

def create_prompt(
    mode: str,
    schema_context: str,
    user_content: str | None = None,
    failed_query: str | None = None,
    error: str | None = None,
) -> str:
    if mode == "retry":
        template_path = os.path.join("prompts", "llm_error_handling_prompt.txt")
        template = load_template(template_path)
        return template.format(
            schema_context=schema_context or "",
            sql_query=failed_query or "",
            final_answer=error or "",
        )
    else: # mode == "normal"
        template_path = os.path.join("prompts", "llm_router_prompt.txt")
        template = load_template(template_path)
        return template.format(
            user_question=user_content or "",
            schema_context=schema_context or "",
        )

# ---------- Conversation reset ----------

def reset_agent_conversation_state():
    llm_agent = cl.user_session.get("llm_agent")
    tools_agent = cl.user_session.get("tools_agent")
    if llm_agent and hasattr(llm_agent, "_conversation_history"):
        llm_agent._conversation_history = []
    if tools_agent and hasattr(tools_agent, "_conversation_history"):
        tools_agent._conversation_history = []

# --- *** NEW: Query History Helper *** ---
def add_to_history(query: str, result: str, success: bool):
    """Track query history for session context"""
    history = cl.user_session.get("query_history", [])
    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "success": success,
        "result_preview": result[:200] if result else None
    })
    cl.user_session.set("query_history", history[-10:])  # Keep last 10

# --- *** NEW: Context Validation Helper *** ---
async def validate_schema_size(schema_context: str, logger: EnhancedPerformanceLogger, max_tokens: int = 100000) -> str: # <-- MODIFIED: async
    """Validate and potentially truncate schema if too large"""
    estimated_tokens = len(schema_context) / 4  # Rough estimation
    if estimated_tokens > max_tokens:
        await logger.log("Schema Validation", "Warning",  # <-- MODIFIED: await
                   f"Schema too large ({estimated_tokens} tokens). Context may be truncated by LLM.")
    return schema_context

# ---------- Execution path (MODIFIED for UX) ----------

async def execute_and_correct(sql_query: str, logger: EnhancedPerformanceLogger, is_retry: bool):
    """
    Enhanced execution with:
    1. Real-time progress updates via a single cl.Message
    2. Metric logging (success/fail, duration)
    3. Query history tracking
    """
    tools_agent = cl.user_session.get("tools_agent")
    final_answer_raw = "" # Raw string/JSON response
    final_answer_formatted = "" # Formatted for UI
    duration = 0
    success = False

    # --- NEW: Create persistent progress message ---
    progress_msg = cl.Message(content="⏳ Preparing query execution...")
    await progress_msg.send()
    
    await logger.log("Query Execution", "Started", f"Executing query: {sql_query}") # <-- MODIFIED: await
    execution_prompt = f"execute the following sql++ query using the mcp tools : {sql_query}"
    
    try:
        progress_msg.content = f"🔍 **Executing query:**\n```sql\n{sql_query}\n```\n\n⚙️ Running query on Couchbase..."
        await progress_msg.update()
        
        await logger.log("Dapr Tool Call", "Started", "Sending query to Dapr tools agent...") # <-- MODIFIED: await
        start_time = time.time()
        result_set = await tools_agent.run(execution_prompt)
        duration = time.time() - start_time
        await logger.log("Dapr Tool Call", "Ended", f"Duration: {duration:.4f} seconds.") # <-- MODIFIED: await
        
        final_answer_raw = result_set.content.strip()
        final_answer_formatted = final_answer_raw
        
        # Format if JSON, but keep raw for history
        if final_answer_raw.startswith(("{", "[")) and final_answer_raw.endswith(("}", "]")):
            try:
                parsed_json = json.loads(final_answer_raw)
                final_answer_formatted = f"```json\n{json.dumps(parsed_json, indent=2)}\n```"
            except json.JSONDecodeError:
                pass # Keep as original string
        
        success = "ERROR" not in (final_answer_raw or "").upper()

    except Exception as e:
        final_answer_raw = f"ERROR: Tool execution failed with an exception: {str(e)}"
        final_answer_formatted = f"`{final_answer_raw}`"
        await logger.log("Dapr Tool Call", "Error", final_answer_raw) # <-- MODIFIED: await
        success = False

    # --- Log results and update history ---
    # log_query_result is NOT async, it's just in-memory math
    logger.log_query_result(success=success, duration=duration, retry=is_retry)
    add_to_history(sql_query, final_answer_raw, success=success)

    if not success:
        await logger.log("Query Execution", "Error", f"Query Failed. Error: {final_answer_raw}") # <-- MODIFIED: await
        
        # --- NEW: Update progress message with error ---
        progress_msg.content = f"⚠️ **Query Failed**\n```sql\n{sql_query}\n```\n\n**Error:** {final_answer_formatted}"
        await progress_msg.update()
        
        cl.user_session.set("last_failed_query", sql_query)
        cl.user_session.set("last_error_response", final_answer_raw)
        cl.user_session.set("conversation_state", "WAITING_FOR_RETRY_CONFIRMATION")
        await cl.Message(content="האם תרצה שאנסה שנית? השב 'כן'.").send()
    else:
        await logger.log("Query Execution", "Ended", "Query successful.") # <-- MODIFIED: await
        
        # --- NEW: Update progress message with success ---
        progress_msg.content = f"✅ **Query Successful**\n```sql\n{sql_query}\n```\n\n**Result:**\n{final_answer_formatted}"
        await progress_msg.update()
        
        cl.user_session.set("conversation_state", "WAITING_FOR_PROMPT")


# ---------- Schema Loading Logic (with logging) ----------

async def load_or_discover_schema_and_compress(status_message: cl.Message, logger: EnhancedPerformanceLogger) -> dict:
    """
    Implements the 3-step cache logic with logging.
    """
    bucket_name = os.getenv("COUCHBASE_BUCKET", "travel-sample")
    
    compressed_cache_file = SCHEMA_CONTEXT_DIR / f"{bucket_name}_schema_final_compressed.json"
    full_cache_file = SCHEMA_CONTEXT_DIR / f"{bucket_name}_schema_final_complete.json"

    SCHEMA_CONTEXT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True) # Ensure logs dir exists

    # --- Step 1: Check for existing COMPRESSED schema cache ---
    if compressed_cache_file.exists():
        await logger.log("Schema Cache", "Started", f"Loading COMPRESSED schema from: {compressed_cache_file}") # <-- MODIFIED: await
        status_message.content = f"✅ Loading cached compressed schema..."
        await status_message.update()
        try:
            with open(compressed_cache_file, 'r', encoding='utf-8') as f:
                compressed_schema = json.load(f)
            await logger.log("Schema Cache", "Ended", "COMPRESSED schema loaded successfully.") # <-- MODIFIED: await
            return compressed_schema
        except Exception as e:
            await logger.log("Schema Cache", "Error", f"Error loading compressed schema cache: {e}. Attempting recovery...") # <-- MODIFIED: await
            status_message.content = f"⚠️ Error loading compressed cache: {e}. Attempting recovery..."
            await status_message.update()

    await logger.log("Schema Cache", "Info", "Compressed schema cache not found or failed to load.") # <-- MODIFIED: await
    
    full_schema_object = None 

    # --- Step 2: Check for existing FULL schema cache ---
    if full_cache_file.exists():
        await logger.log("Schema Cache", "Started", f"Loading FULL schema cache from {full_cache_file} for compression...") # <-- MODIFIED: await
        status_message.content = f"⚙️ Loading full schema cache for compression..."
        await status_message.update()
        try:
            with open(full_cache_file, 'r', encoding='utf-8') as f:
                full_schema_object = json.load(f)
            await logger.log("Schema Cache", "Ended", "FULL schema loaded successfully.") # <-- MODIFIED: await
        except Exception as e:
            await logger.log("Schema Cache", "Error", f"Error loading full schema cache: {e}. Re-running discovery.") # <-- MODIFIED: await
            status_message.content = f"⚠️ Error loading full cache: {e}. Re-running discovery..."
            await status_message.update()
    
    # --- Step 3: Run FULL discovery if no full schema was loaded ---
    if full_schema_object is None:
        await logger.log("Schema Discovery", "Started", "Running full discovery process (discovery.py)...") # <-- MODIFIED: await
        status_message.content = "⚙️ Schema not found. Running full discovery... (This may take several minutes)"
        await status_message.update()
        try:
            # discovery.py will create its own log file
            full_schema_object = await run_full_discovery() 
            if not full_schema_object:
                raise RuntimeError("Schema discovery (discovery.py) ran but returned no data.")
            
            await logger.log("Schema Discovery", "Ended", "Discovery process finished.") # <-- MODIFIED: await
            status_message.content = "✅ Discovery complete. Compressing schema..."
            await status_message.update()
        except Exception as e:
            await logger.log("Schema Discovery", "Error", f"Critical error during schema discovery: {e}") # <-- MODIFIED: await
            status_message.content = f"🚨 Critical error during schema discovery: {e}. Aborting."
            await status_message.update()
            raise 

    # --- Step 4: Compress the FULL schema (which now exists in-memory) ---
    try:
        await logger.log("Schema Compression", "Started", f"Running compression with max_samples={COMPRESSOR_MAX_SAMPLES}...") # <-- MODIFIED: await
        status_message.content = "⚙️ Compressing schema..."
        await status_message.update()
        
        compressor = SchemaCompressor(
            max_samples=COMPRESSOR_MAX_SAMPLES, 
            include_statistics=False
        )
        compressed_schema = compressor.compress_schema(full_schema_object)

        await logger.log("Schema Compression", "Info", f"Saving compressed schema to cache: {compressed_cache_file}") # <-- MODIFIED: await
        with open(compressed_cache_file, 'w', encoding='utf-8') as f:
            json.dump(compressed_schema, f, indent=2, ensure_ascii=False)

        await logger.log("Schema Compression", "Ended", "Schema compressed and cached.") # <-- MODIFIED: await
        status_message.content = f"✅ Schema compressed and cached."
        await status_message.update()

        return compressed_schema

    except Exception as e:
        await logger.log("Schema Compression", "Error", f"Error during schema compression: {e}") # <-- MODIFIED: await
        status_message.content = f"🚨 Error during schema compression: {e}. Aborting."
        await status_message.update()
        raise


# ---------- Startup flow (with logging) ----------

@cl.on_chat_start
async def start():
    # --- *** MODIFIED: Use EnhancedPerformanceLogger with continuous, async logging *** ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = cl.user_session.get('id', 'unknown_session')
    filename = f"app_session_log_{session_id}_{timestamp}.json"
    log_filepath = LOGS_DIR / filename
        
    logger = EnhancedPerformanceLogger(filepath=log_filepath) # <-- MODIFIED: Pass filepath
    cl.user_session.set("logger", logger)
    await logger.log("Chat Session", "Started", f"Session ID: {cl.user_session.get('id')}") # <-- MODIFIED: await
    
    # --- *** NEW: Init query history *** ---
    cl.user_session.set("query_history", [])

    mcp_url = os.getenv("MCP_SERVER_URL")
    if not mcp_url:
        await logger.log("Config Validation", "Error", "MCP_SERVER_URL environment variable not set.") # <-- MODIFIED: await
        await cl.Message(content="Error: MCP_SERVER_URL environment variable not set.").send()
        return

    client = MCPClient(timeout=60.0)
    msg = None
    try:
        msg = cl.Message(content="🔌 Connecting to MCP server...")
        await msg.send()
        await logger.log("MCP Connection", "Started", f"Connecting to MCP server at {mcp_url}...") # <-- MODIFIED: await
        
        start_time = time.time()
        await client.connect_sse(server_name="couchbase_mcp", url=mcp_url, headers=None)
        duration = time.time() - start_time
        
        await logger.log("MCP Connection", "Ended", f"MCP connection took {duration:.4f} seconds.") # <-- MODIFIED: await
        msg.content = f"✅ MCP connection successful ({duration:.4f}s)."
        await msg.update()
    except Exception as e:
        await logger.log("MCP Connection", "Error", f"Failed to connect to MCP Server: {e}") # <-- MODIFIED: await
        error_content = f"Error: Failed to connect to MCP Server: {e}"
        if msg:
            msg.content = error_content
            await msg.update()
        else:
            await cl.Message(content=error_content).send()
        return

    tools = client.get_all_tools()
    if not tools:
        await logger.log("MCP Tools", "Error", "Tools not initialized.") # <-- MODIFIED: await
        await cl.Message(content="Error: Tools not initialized. Please restart the chat.").send()
        return
    await logger.log("MCP Tools", "Info", f"Found {len(tools)} tools.") # <-- MODIFIED: await

    # Step 2: Load/Discover AND Compress Schema
    try:
        if not msg:
            msg = cl.Message(content="⚙️ Loading schema...")
            await msg.send()
        else:
            msg.content = "⚙️ Loading schema..."
            await msg.update()
        
        await logger.log("Schema Loading", "Started", "Starting 3-step schema load/discovery process...") # <-- MODIFIED: await
        start_time = time.time()
        compressed_schema_object = await load_or_discover_schema_and_compress(msg, logger)
        duration = time.time() - start_time

        schema_context = json.dumps(compressed_schema_object, indent=2)
        
        # --- *** NEW: Validate schema size *** ---
        await logger.log("Schema Validation", "Started", "Validating compressed schema size...") # <-- MODIFIED: await
        # Pass logger to the now-async validation function
        schema_context = await validate_schema_size(schema_context, logger=logger) # <-- MODIFIED: await
        await logger.log("Schema Validation", "Ended", "Validation complete.") # <-- MODIFIED: await
        # --- *** END NEW *** ---
        
        cl.user_session.set("schema_context", schema_context)

        await logger.log("Schema Loading", "Ended", f"Schema processing complete in {duration:.4f} seconds.") # <-- MODIFIED: await
        msg.content = f"✅ Schema processed in {duration:.4f} seconds."
        await msg.update()

    except Exception as e:
        await logger.log("Schema Loading", "Error", f"Critical error during schema processing: {e}") # <-- MODIFIED: await
        error_content = f"🚨 Error during schema processing: {e}"
        if msg:
             msg.content = error_content
             await msg.update()
        else:
             await cl.Message(content=error_content).send()
        return # Cannot continue without schema

    # Step 3: Initialize agents
    component_name = os.getenv("DAPR_LLM_COMPONENT_DEFAULT", "openai")
    provider = os.getenv("DAPR_LLM_PROVIDER")

    await logger.log("Agent Init", "Started", f"Creating LLM Agent (Component: {component_name})...") # <-- MODIFIED: await
    llm_instructions = [
        "You are an expert N1QL (Couchbase) query specialist with 10+ years of experience.",
        "ROLE: Senior N1QL/SQL++ Database Engineer specializing in JSON document querying and optimization.",
        "SESSION BEHAVIOR: For query generation, rely on the already provided schema context.",
        "Your goal is ONLY to generate a valid SQL++ query based on the user's request and the schema context.",
        "If the user asks a question that cannot be answered with a SQL++ query based on the schema, respond with 'Cannot answer this question based on the provided schema.'",
        "If the user asks for data modification (INSERT, UPDATE, DELETE), respond with 'Data modification queries are not supported.'",
        "Output ONLY the SQL++ query itself, prefixed with 'Tool needed:'. For example: 'Tool needed: SELECT * FROM my_bucket LIMIT 1;'",
        "Do NOT add explanations, justifications, or any text other than the prefix and the query.",
        "Do NOT add a semicolon at the end of the query unless absolutely required by SQL++ syntax (rarely needed).",
    ]
    llm_agent = Agent(
        name="llm_agent",
        role="Senior SQL++ Couchbase Database Expert Engineer specializing in SQL++ Couchbase querying and optimization.",
        instructions=llm_instructions,
        llm=DaprChatClient(component_name=component_name, provider=provider),
    )
    cl.user_session.set("llm_agent", llm_agent)
    await logger.log("Agent Init", "Ended", "LLM Agent created.") # <-- MODIFIED: await

    await logger.log("Agent Init", "Started", "Creating Tools Agent...") # <-- MODIFIED: await
    tools_agent = Agent(
        name="tools_agent",
        role="Execution agent using couchbase mcp tools.",
        instructions=[
            "You are an Execution agent using mcp tools to query the Couchbase.",
            "if given sql++ query, use the mcp tools to execute it. Do not change the query.",
#            "Return ONLY the raw JSON result from the tool, or an error message if it fails.",
#            "Do not add explanations, summaries, or any extra text.",
        ],
        tools=tools,
    )
    cl.user_session.set("tools_agent", tools_agent)
    await logger.log("Agent Init", "Ended", "Tools Agent created.") # <-- MODIFIED: await
    
    cl.user_session.set("conversation_state", "WAITING_FOR_PROMPT")
    await cl.Message(content="✅ Couchbase Agent is ready. How can I help?").send()
    cl.user_session.set("status_message", msg)
    await logger.log("Chat Session", "Info", "Initialization complete. Waiting for user prompt.") # <-- MODIFIED: await


# ---------- Unified message handler (with logging) ----------

@cl.on_message
async def main(message: cl.Message):
    # --- *** MODIFIED: Get EnhancedPerformanceLogger *** ---
    logger = cl.user_session.get("logger")
    if not logger:
        # Failsafe in case session start failed
        # --- *** MODIFIED: Create failsafe filepath *** ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = cl.user_session.get('id', 'failsafe_session')
        filename = f"app_session_log_{session_id}_{timestamp}.json"
        log_filepath = LOGS_DIR / filename
        # --- *** END MODIFIED *** ---
            
        logger = EnhancedPerformanceLogger(filepath=log_filepath) # Use new class with path
        cl.user_session.set("logger", logger)
        await logger.log("Logger Error", "Warning", "Logger not found in session, created a new failsafe log file.") # <-- MODIFIED: await

    reset_agent_conversation_state()
    state = cl.user_session.get("conversation_state")
    msg = cl.user_session.get("status_message")
    llm_agent = cl.user_session.get("llm_agent")
    schema_context = cl.user_session.get("schema_context")

    if not all([llm_agent, schema_context]):
        await logger.log("Message Error", "Error", "Session not fully initialized.") # <-- MODIFIED: await
        await cl.Message(content="Error: Session not fully initialized. Please restart the chat.").send()
        return

    user_text = (message.content or "").strip()
    await logger.log("User Message", "Info", f"State: {state}, Content: '{user_text}'") # <-- MODIFIED: await
    
    is_retry_state = state == "WAITING_FOR_RETRY_CONFIRMATION"
    wants_retry = user_text.lower() == "כן"
    mode = "retry" if is_retry_state and wants_retry else "normal"

    prompt = "" 
    if mode == "retry":
        last_failed_query = cl.user_session.get("last_failed_query")
        last_error_response = cl.user_session.get("last_error_response")
        await cl.Message(content="🔄 Understood. Attempting another correction...").send()
        await logger.log("LLM Call (Retry)", "Started", f"Attempting correction for query: {last_failed_query}") # <-- MODIFIED: await
        prompt = create_prompt(mode="retry", schema_context=schema_context, failed_query=last_failed_query, error=last_error_response,)
    else: # mode == "normal"
        if is_retry_state and not wants_retry:
            cl.user_session.set("conversation_state", "WAITING_FOR_PROMPT")
            await logger.log("User Message", "Info", "Retry declined. Resetting state.") # <-- MODIFIED: await
            await cl.Message(content="Understood. Let's start over. How can I help?").send()
            return
        await logger.log("LLM Call (Normal)", "Started", "Generating SQL query from user prompt.") # <-- MODIFIED: await
        prompt = create_prompt(mode="normal", schema_context=schema_context, user_content=user_text,)

    if msg:
        msg.content = "⚙️ Sending prompt to LLM..."
        await msg.update()

    start_time = time.time()
    result = await llm_agent.run(prompt)
    duration = time.time() - start_time
    await logger.log(f"LLM Call ({mode.capitalize()})", "Ended", f"Duration: {duration:.4f}s") # <-- MODIFIED: await

    if msg:
        msg.content = f"✅ LLM call took: {duration:.4f} seconds."
        await msg.update()

    prefix = "Tool needed:"
    content = (result.content or "").strip()

    if content.startswith(prefix):
        sql_query = content[len(prefix):].strip()
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1].strip()
        
        # --- *** MODIFIED: Pass logger and retry state *** ---
        await execute_and_correct(sql_query, logger, is_retry=(mode == "retry"))
        return

    # No tool needed
    await logger.log("LLM Call", "Info", f"LLM responded directly (no tool): {content}") # <-- MODIFIED: await
    if mode == "retry":
        await cl.Message(content=f"🚨 Automated correction failed again. The LLM responded: {content}").send()
        cl.user_session.set("conversation_state", "WAITING_FOR_PROMPT")
    else:
        await cl.Message(content=f"✅ **Answer:** {content}").send()

# --- *** MODIFIED: Save log and stats on chat end *** ---
@cl.on_chat_end
async def end():
    """
    Saves the performance log and stats for the session when the user closes the chat.
    """
    logger = cl.user_session.get("logger")
    if logger and isinstance(logger, EnhancedPerformanceLogger):
        await logger.log("Chat Session", "Ended", "User session ended.") # <-- MODIFIED: await
        
        # --- NEW: Log summary stats ---
        summary_stats = logger.get_summary_stats()
        # This log call will automatically flush the final stats to the file
        await logger.log("Session Summary", "Info", json.dumps(summary_stats, indent=2)) # <-- MODIFIED: await
        
        # --- *** MODIFIED: Remove filename generation *** ---
        # The filename is set at the start. We just call save_to_file()
        # to print the final confirmation message.
        
        try:
            logger.save_to_file() # <-- MODIFIED: Removed filename argument
        except Exception as e:
            print(f"Error printing log path on chat end: {e}")