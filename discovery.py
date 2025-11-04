# discovery_complete.py version 3.0.4
#
# Description:
# - Merges discovery.py (P1, P2) and disc.py (P3) into a single, fully dynamic script.
# - Phase 1: Dynamically discovers all scopes and collections.
# - Phase 2: Dynamically counts documents in all discovered collections.
# - Phase 3: Runs INFER for all non-empty collections using robust,
#   retry-enabled logic (429 handling) and final JSON consolidation.
# - (v3.0.1) main_discovery() now returns the final schema object.
# - (v3.0.1) WRITE_RAW_FILES_TO_DISK switch now controls P1 and P2 checkpoint files as well.
# - (v3.0.4) Integrated user-provided advanced parsing and repair logic:
#   - Added detect_truncation() and smart/simple repair functions.
#   - Replaced parse_and_normalize_schema_content with the new 4-scenario handler
#     that includes truncation repair, ast.literal_eval fallback, and error/warning reporting.
#   - P3 now saves parsing warnings (e.g., truncation) to the final JSON 'error' field.

import os
import asyncio
import json
from datetime import datetime
import time
from pathlib import Path
from dotenv import load_dotenv
import re
import ast # For parsing the list-of-strings output
from typing import Dict, Any, Optional, List

# --- Dapr Agents Imports ---
try:
    from dapr_agents import Agent
    from dapr_agents.tool.mcp.client import MCPClient
    from dapr_agents.llm.dapr import DaprChatClient
    from langchain_core.messages import BaseMessage
except ImportError:
    print("🛑 Error: dapr_agents library not found. Please install it (`pip install dapr-agents`).")
    exit(1)

# Load environment variables
load_dotenv()

# --- Configuration ---
INFER_SAMPLE_SIZE = int(os.getenv("INFER_SAMPLE_SIZE", "1000"))
INFER_NUM_SAMPLE_VALUES = int(os.getenv("INFER_NUM_SAMPLE_VALUES", "3"))
INFER_SIMILARITY_METRIC = float(os.getenv("INFER_SIMILARITY_METRIC", "0.6"))

# --- This switch now controls ALL non-final file writing ---
WRITE_RAW_FILES_TO_DISK = False # Set to True to write P1, P2, and P3 raw/checkpoint files

# Define directories
SCHEMA_CONTEXT_DIR = Path("./schema_context")
LOGS_DIR = Path("./logs")
# --- Directory for FULL run raw results (from disc.py) ---
RAW_RESPONSES_DIR = SCHEMA_CONTEXT_DIR / "infer_results_full"

# --- HELPER FUNCTIONS (Shared & from discovery - Copy.py) ---

def log_message(message: str):
    """Prints a message with a timestamp."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} - {message}")

class PerformanceLogger:
    """Simple in-memory logger for tracking performance (based on disc.py)"""
    def __init__(self):
        self.logs = []
        self.log_counter = 0
        self.event_timers = {}

    def log(self, event_name: str, status: str, details: str = "") -> dict:
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
        print(f"[{log_entry['log_id']}] {timestamp.strftime('%H:%M:%S.%f')[:-3]} | {event_name} | {status}{duration_str} {f'| {details}' if details else ''}")
        return log_entry

    def save_to_file(self, filename: str = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovery_complete_log_{timestamp}.json"
        
        logs_dir = LOGS_DIR
        logs_dir.mkdir(exist_ok=True) 
        filepath = logs_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Log saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Error saving log file: {e}")
        return filepath

# --- Agent Interaction Helpers (Merged) ---

# Pre-compiled regex (needed for P1/P2)
JSON_FENCE_RE = re.compile(
    r'```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```',
    re.DOTALL | re.IGNORECASE
)

def normalize_llm_content(resp: object) -> str:
    """(Helper for P1/P2/P3) Normalize any tool/LLM response to a clean string."""
    try:
        # This logic works for both Agent (P1/P2) and DaprChatClient (P3) responses
        if hasattr(resp, 'content'):
            raw = resp.content
        else:
            raw = resp
    except Exception:
        raw = resp
    return (str(raw) if raw is not None else "").strip().lstrip("\ufeff")

def extract_json(text: str, logger: Optional[PerformanceLogger] = None) -> Optional[Any]:
    """(Helper for P1/P2/P3) Extract top-level JSON (object or array) from text with fenced+fallback."""
    if not text: return None
    t = text.strip().lstrip("\ufeff")
    m = JSON_FENCE_RE.search(t)
    json_str = next((g for g in m.groups() if g), None) if m else None
    if not json_str:
        first_obj = t.find("{")
        first_arr = t.find("[")
        if t.startswith('[') and t.endswith(']'): json_str = t
        elif t.startswith('{') and t.endswith('}'): json_str = t
        else:
            idxs = [i for i in (first_obj, first_arr) if i != -1]
            if idxs: json_str = t[min(idxs):].strip()
    if not json_str:
        if logger: logger.log("Extract JSON", "Warning", "No JSON structure found.")
        return None
    try: return json.loads(json_str)
    except json.JSONDecodeError as e:
        if logger: logger.log("Extract JSON", "Error", f"Decode failed: {e} | Text: {json_str[:200]}...")
        return None

async def agent_run_tool(agent: Agent, tool_name: str, args: dict, logger: PerformanceLogger) -> object:
    """(Helper for P1/P2) Execute a tool via the agent and log performance."""
    event_name = f"Call {tool_name}"
    args_summary_dict = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 100: args_summary_dict[k] = f"{v[:50]}...{v[-50:]}"
        else: args_summary_dict[k] = v
    args_summary = json.dumps(args_summary_dict)
    log_entry = logger.log(event_name, "Started", args_summary)
    try:
        prompt = (
            f"Run the exact MCP tool `{tool_name}` with the following JSON arguments:\n"
            f"```json\n{json.dumps(args, indent=2)}\n```\n"
            "Return only the raw result from the tool (JSON or text), with no extra explanation or formatting."
        )
        result = await agent.run(prompt)
        logger.log(event_name, "Ended", f"Type: {type(result).__name__}")
        return result
    except Exception as e:
        logger.log(event_name, "Error", f"{str(e)} - Args: {args_summary}")
        raise

async def agent_run_tool_minimal(agent: Agent, tool_name: str, args: dict, logger: PerformanceLogger) -> object:
    """(Helper for P3) Executes a tool via the agent, returns raw response object."""
    event_name = f"Call {tool_name}"
    args_summary_log = {"query": args.get("query", "N/A")[:100] + "..."} if "query" in args else args
    log_entry = logger.log(event_name, "Started", json.dumps(args_summary_log))

    print(f"--- Running Tool: {tool_name} ---")
    print(f"Arguments:\n{json.dumps(args, indent=2)}")

    try:
        prompt = (
            f"Run the exact MCP tool `{tool_name}` with the following JSON arguments:\n"
            f"```json\n{json.dumps(args, indent=2)}\n```\n"
            "Return ONLY the raw output from the tool. Do not add any explanation or formatting."
        )
        result = await agent.run(prompt)
        logger.log(event_name, "Ended", f"Type: {type(result).__name__}")
        return result
    except Exception as e:
        print(f"🛑 Error running tool {tool_name}: {e}")
        logger.log(event_name, "Error", str(e))
        raise

# --- P3 Helper Functions (from user's discovery.py) ---

def detect_truncation(raw_str: str) -> bool:
    """
    (Helper for P3) Detects if the content was truncated by checking if it ends with 
    ...' role='assistant' without proper JSON closing brackets.
    """
    truncation_pattern = r"\.\.\.'\s*role='assistant'$"
    if re.search(truncation_pattern, raw_str):
        # Verify it's NOT properly closed JSON by counting brackets
        before_truncation = re.sub(r"\.\.\.'\s*role='assistant'$", "", raw_str)
        open_curly = before_truncation.count('{')
        close_curly = before_truncation.count('}')
        open_square = before_truncation.count('[')
        close_square = before_truncation.count(']')
        
        if open_curly > close_curly or open_square > close_square:
            return True
    return False


def repair_truncated_json_smart(raw_str: str, logger: PerformanceLogger) -> Optional[str]:
    """
    (Helper for P3) Smart JSON repair using stateful parsing.
    Tracks context (inside string, inside object/array) to close structures correctly.
    """
    logger.log("JSON Repair (Smart)", "Started", "Context-aware repair...")
    
    # Remove truncation suffix
    clean_str = re.sub(r"\.\.\.'\s*role='assistant'$", "", raw_str).strip()
    
    # State tracking
    stack = []  # Stack of open structures: '}' or ']' (what we need to close)
    in_string = False
    escape_next = False
    
    for i, char in enumerate(clean_str):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        # Only track brackets outside of strings
        if not in_string:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
    
    # If we ended inside a string, close it
    repaired = clean_str
    if in_string:
        repaired += '"'
        logger.log("JSON Repair (Smart)", "Info", "Closed unclosed string.")
    
    # Close remaining structures (in reverse order)
    closing_sequence = ''.join(reversed(stack))
    repaired += closing_sequence
    
    logger.log("JSON Repair (Smart)", "Info", 
               f"Added closing sequence: '{closing_sequence}' ({len(stack)} structures)")
    
    # Validation
    try:
        json.loads(repaired)
        logger.log("JSON Repair (Smart)", "Ended", "Smart repair successful and validated.")
        return repaired
    except json.JSONDecodeError as e:
        logger.log("JSON Repair (Smart)", "Error", f"Validation failed: {e}")
        # Fall back to simple repair
        logger.log("JSON Repair (Smart)", "Warning", "Falling back to simple bracket counting...")
        return repair_truncated_json_simple(clean_str, logger)


def repair_truncated_json_simple(clean_str: str, logger: PerformanceLogger) -> Optional[str]:
    """
    (Helper for P3) Fallback: simple bracket counting repair.
    """
    logger.log("JSON Repair (Simple)", "Started", "Simple bracket counting...")
    
    open_curly = clean_str.count('{')
    close_curly = clean_str.count('}')
    open_square = clean_str.count('[')
    close_square = clean_str.count(']')
    
    missing_curly = open_curly - close_curly
    missing_square = open_square - close_square
    
    # Safety threshold: don't close more than 100 brackets
    MAX_BRACKETS_TO_CLOSE = 100
    if missing_curly > MAX_BRACKETS_TO_CLOSE or missing_square > MAX_BRACKETS_TO_CLOSE:
        logger.log("JSON Repair (Simple)", "Error", 
                   f"Unrepairable: {missing_curly} '{{' and {missing_square} '[' unclosed. " +
                   f"Threshold: {MAX_BRACKETS_TO_CLOSE}")
        return None
    
    repaired = clean_str + ('}' * missing_curly) + (']' * missing_square)
    
    logger.log("JSON Repair (Simple)", "Info", 
               f"Added {missing_curly} '}}' and {missing_square} ']'")
    
    try:
        json.loads(repaired)
        logger.log("JSON Repair (Simple)", "Ended", "Simple repair successful.")
        return repaired
    except json.JSONDecodeError as e:
        logger.log("JSON Repair (Simple)", "Error", f"Validation failed: {e}")
        return None

# --- *** REPLACED with user-provided function (v3.0.4) *** ---
def parse_and_normalize_schema_content(resp: object, logger: PerformanceLogger) -> (List[Dict], str, Optional[str]):
    """
    (Helper for P3) Enhanced parser with 4-scenario handling:
    1. Array of objects → parse each
    2. Single object → wrap in array → parse
    3. Invalid content → error
    4. Truncated JSON → repair → parse
    
    Returns:
        (schemas_list, raw_content_str, error_message)
        - schemas_list: List[Dict] - parsed schemas (empty on error)
        - raw_content_str: str - original/repaired content
        - error_message: Optional[str] - None if success, warning/error otherwise
    """
    # Step 1: Normalize to clean string (using the shared helper)
    raw_content_str = normalize_llm_content(resp)
    
    if not raw_content_str:
        logger.log("Schema Parsing (P3)", "Warning", "Agent returned empty content.")
        return [], "", "Empty response from agent"
    
    logger.log("Schema Parsing (P3)", "Started", f"Content length: {len(raw_content_str)}")
    
    # Step 2: Check for truncation
    is_truncated = detect_truncation(raw_content_str)
    truncation_error = None
    
    if is_truncated:
        logger.log("Schema Parsing (P3)", "Warning", "Detected truncated response. Attempting repair...")
        repaired = repair_truncated_json_smart(raw_content_str, logger)
        
        if repaired:
            original_len = len(raw_content_str)
            raw_content_str = repaired
            truncation_error = f"WARNING: Response truncated at ~{original_len} chars. Repaired to {len(repaired)} chars. Schema may be incomplete."
            logger.log("Schema Parsing (P3)", "Info", truncation_error)
        else:
            logger.log("Schema Parsing (P3)", "Error", "Failed to repair truncated JSON.")
            return [], raw_content_str, "ERROR: Response truncated and unrepairable"
    
    # Step 3: Extract JSON using proven logic from P1/P2
    logger.log("Schema Parsing (P3)", "Started", "Extracting JSON content...")
    parsed_json = extract_json(raw_content_str, logger)
    
    if parsed_json is None:
        logger.log("Schema Parsing (P3)", "Warning", "extract_json failed. Trying ast.literal_eval fallback...")
        
        # Fallback for array-of-strings format: ['{"..."}', '{"..."}']
        try:
            eval_result = ast.literal_eval(raw_content_str)
            if isinstance(eval_result, list):
                parsed_json = []
                for item_str in eval_result:
                    if isinstance(item_str, str):
                        inner_obj = json.loads(item_str)
                        parsed_json.append(inner_obj)
                logger.log("Schema Parsing (P3)", "Ended", 
                          f"Success via ast.literal_eval. Parsed {len(parsed_json)} items.")
            else:
                raise ValueError(f"ast.literal_eval returned non-list: {type(eval_result)}")
        
        except Exception as fallback_err:
            logger.log("Schema Parsing (P3)", "Error", 
                      f"All parsing methods failed: {fallback_err}")
            # Scenario 3: Invalid content
            error_msg = f"ERROR: Unable to parse content - {str(fallback_err)}"
            return [], raw_content_str, error_msg
    
    # Step 4: Normalize to list (Scenarios 1 & 2)
    if isinstance(parsed_json, dict):
        # Scenario 2: Single object
        logger.log("Schema Normalization (P3)", "Info", "Normalized single object to list.")
        return [parsed_json], raw_content_str, truncation_error
    
    elif isinstance(parsed_json, list):
        # Scenario 1: Array of objects
        logger.log("Schema Normalization (P3)", "Info", 
                  f"Result is list with {len(parsed_json)} items.")
        return parsed_json, raw_content_str, truncation_error
    
    else:
        # Scenario 3: Unexpected type
        logger.log("Schema Parsing (P3)", "Error", 
                  f"Unexpected JSON type after parsing: {type(parsed_json)}")
        error_msg = f"ERROR: Unexpected JSON type - {type(parsed_json).__name__}"
        return [], raw_content_str, error_msg


def create_infer_agent(llm_client: DaprChatClient, tools: List[Dict], instructions: List[str], logger: PerformanceLogger) -> Agent:
    """
    (Helper for P3) Creates and returns a new instance of the InferHelperAgent.
    This ensures a clean history for each call.
    """
    logger.log("Create Agent", "Started", "Creating new InferHelperAgent instance...")
    agent = Agent(
        name="InferHelperAgent", role="Tool Execution Agent",
        instructions=instructions, tools=tools, llm=llm_client
    )
    logger.log("Create Agent", "Ended", "Instance created.")
    return agent

# --- PHASE 1 FUNCTION (from discovery - Copy.py) ---

async def run_phase1_discovery(agent: Agent, bucket_name: str, logger: PerformanceLogger) -> Optional[Dict[str, List[str]]]:
    """Phase 1: Discover scopes/collections, save checkpoint, return map."""
    logger.log("Phase 1", "Started", "Discover Scopes & Collections")
    
    output_path = SCHEMA_CONTEXT_DIR / f"{bucket_name}_schema_phase1.json"
    
    structured_data = None
    try:
        print(f"\nDiscovering scopes/collections for bucket: {bucket_name}...\n")
        raw_resp = await agent_run_tool(agent, "CouchbaseMcpGetScopesAndCollectionsInBucket", {}, logger)
        logger.log("Normalize Agent Response (P1)", "Started"); raw_content = normalize_llm_content(raw_resp)
        logger.log("Normalize Agent Response (P1)", "Ended", f"Length: {len(raw_content)}")
        print(f"\nRaw Content (P1):\n{raw_content[:500]}{'...' if len(raw_content) > 500 else ''}\n")
        
        logger.log("Parse Content to JSON (P1)", "Started"); structured_data = extract_json(raw_content, logger)
        if structured_data is None:
            logger.log("Parse Content to JSON (P1)", "Warning", "Robust extract failed, trying simple load.")
            try: structured_data = json.loads(raw_content); logger.log("Parse Content to JSON (P1)", "Ended", "Success: simple load.")
            except json.JSONDecodeError as err: logger.log("Parse Content to JSON (P1)", "Error", f"Simple load failed: {err}."); raise ValueError(f"CRITICAL: Failed Phase 1 parse: {raw_content}")
        else: logger.log("Parse Content to JSON (P1)", "Ended", "Success: robust extract.")
        
        if WRITE_RAW_FILES_TO_DISK:
            logger.log("Save Checkpoint File (P1)", "Started", str(output_path))
            with open(output_path, 'w', encoding='utf-8') as f: json.dump(structured_data, f, indent=2, ensure_ascii=False)
            logger.log("Save Checkpoint File (P1)", "Ended"); print(f"✓ Phase 1 Checkpoint saved: {output_path}")
        else:
            logger.log("Save Checkpoint File (P1)", "Skipped", "WRITE_RAW_FILES_TO_DISK is False.")
        
        logger.log("Phase 1", "Ended"); return structured_data
    except Exception as e: logger.log("Phase 1", "Error", str(e)); print(f"🛑 Error during Phase 1: {e}"); raise

# --- PHASE 2 FUNCTION (from discovery - Copy.py) ---

async def run_phase2_doc_count(agent: Agent, bucket_name: str, logger: PerformanceLogger, scope_collection_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Phase 2: Build & run UNION ALL COUNT, update counts, return updated structure."""
    logger.log("Phase 2", "Started", "Document Counting")
    
    phase1_path = SCHEMA_CONTEXT_DIR / f"{bucket_name}_schema_phase1.json"
    phase2_path = SCHEMA_CONTEXT_DIR / f"{bucket_name}_schema_phase2.json"
    
    intermediate_struct: Dict[str, List[Dict[str, Any]]] = {}
    
    if scope_collection_map is None:
        logger.log("Load Phase 1 Data (P2)", "Started", f"In-memory map not provided. Reading from disk: {phase1_path}")
        if not phase1_path.exists(): 
            logger.log("Load Phase 1 Data (P2)", "Error", "Not found"); 
            raise FileNotFoundError(f"{phase1_path} not found. Cannot proceed without P1 data.")
        try:
            with open(phase1_path, 'r', encoding='utf-8') as f: 
                scope_collection_map = json.load(f)
            logger.log("Load Phase 1 Data (P2)", "Ended")
        except Exception as e: 
            logger.log("Load Phase 1 Data (P2)", "Error", f"Load/parse failed: {e}"); raise
    else: 
        logger.log("Load Phase 1 Data (P2)", "Skipped", "Data provided in-memory from P1.")

    logger.log("Transform Data (P2)", "Started"); collections_to_query: List[Dict[str, str]] = []
    for scope, collections in scope_collection_map.items():
        if scope == "_system": continue
        if not isinstance(collections, list): logger.log("Transform Data (P2)", "Warning", f"Skipping scope '{scope}': expected list."); continue
        intermediate_struct[scope] = []
        for coll in collections:
            if not isinstance(coll, str): logger.log("Transform Data (P2)", "Warning", f"Skipping invalid collection name in '{scope}': {coll}"); continue
            intermediate_struct[scope].append({"name": coll, "documents": "UNDEFINED"})
            collections_to_query.append({"scope": scope, "collection": coll})
    logger.log("Transform Data (P2)", "Ended", f"Prepared {len(collections_to_query)} collections.")
    
    if not collections_to_query:
        logger.log("Generate Query (P2)", "Skipped", "No collections."); print("⚠️ No collections to count.")
        if WRITE_RAW_FILES_TO_DISK:
            logger.log("Save Output File (P2)", "Started", str(phase2_path))
            with open(phase2_path, 'w', encoding='utf-8') as f: json.dump(intermediate_struct, f, indent=2, ensure_ascii=False)
            logger.log("Save Output File (P2)", "Ended"); print(f"✓ Phase 2 (empty) saved: {phase2_path}")
        else:
            logger.log("Save Output File (P2)", "Skipped", "WRITE_RAW_FILES_TO_DISK is False.")
        logger.log("Phase 2", "Ended")
        return intermediate_struct

    logger.log("Generate Query (P2)", "Started"); union_parts = []
    for item in collections_to_query:
        scope, collection = item['scope'], item['collection']
        query_part = f'SELECT "{scope}.{collection}" AS scope_collection, COUNT(1) AS doc_count FROM `{bucket_name}`.`{scope}`.`{collection}`'
        union_parts.append(query_part)
    full_query = "\nUNION ALL\n".join(union_parts) + ";"; logger.log("Generate Query (P2)", "Ended", f"Length: {len(full_query)}")
    print("\nGenerated SQL++ Query (P2):\n---\n" + full_query + "\n---\n")
    
    doc_counts_result = None; print(f"Executing UNION ALL via Agent...\n")
    try:
        raw_resp = await agent_run_tool(agent, "CouchbaseMcpRunSqlPlusPlusQuery", {"query": full_query}, logger)
        logger.log("Normalize Count Response (P2)", "Started"); count_txt = normalize_llm_content(raw_resp)
        logger.log("Normalize Count Response (P2)", "Ended", f"Length: {len(count_txt)}")
        print(f"\nRaw Count Results (P2):\n{count_txt[:500]}{'...' if len(count_txt) > 500 else ''}\n")
        
        logger.log("Parse Count Results (P2)", "Started", "Expecting direct JSON array")
        doc_counts_result = extract_json(count_txt, logger)
        if doc_counts_result is None:
            logger.log("Parse Count Results (P2)", "Warning", "Direct JSON parse failed. Trying ast.literal_eval for list-of-strings.")
            try:
                eval_out = ast.literal_eval(count_txt)
                if isinstance(eval_out, list):
                    doc_counts_result = []
                    for item_str in eval_out:
                        if not isinstance(item_str, str): continue
                        inner_obj = json.loads(item_str)
                        if isinstance(inner_obj, dict) and "$1" in inner_obj: doc_counts_result.append(inner_obj["$1"])
                        elif isinstance(inner_obj, dict): doc_counts_result.append(inner_obj)
                    logger.log("Parse Count Results (P2)", "Ended", f"Success via ast + inner parse. Parsed {len(doc_counts_result)} entries.")
                else: raise ValueError(f"ast.literal_eval did not return a list: {count_txt}")
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as final_err:
                logger.log("Parse Count Results (P2)", "Error", f"All parsing failed: {final_err}"); raise
        if not isinstance(doc_counts_result, list):
            logger.log("Validate Count Results (P2)", "Error", f"Expected list, got {type(doc_counts_result)}"); raise
        
        logger.log("Parse Count Results (P2)", "Ended", f"Parsed {len(doc_counts_result)} entries.")
        count_map = {item.get("scope_collection"): item.get("doc_count", "ERROR") for item in doc_counts_result if isinstance(item, dict)}
        logger.log("Update Structure (P2)", "Started"); updated, not_found = 0, 0
        for scope, collections in intermediate_struct.items():
            for info in collections:
                full_name = f"{scope}.{info['name']}"
                if full_name in count_map: info["documents"] = count_map[full_name]; updated += 1
                else: info["documents"] = 0; not_found += 1; logger.log("Update Structure (P2)", "Warning", f"{full_name} not in results, assuming 0.")
        logger.log("Update Structure (P2)", "Ended", f"Updated: {updated}, Assumed 0: {not_found}")
    except Exception as e:
         logger.log("Phase 2 Execution", "Error", f"UNION ALL failed: {e}"); print(f"🛑 Error during Phase 2: {e}")
         logger.log("Update Structure (P2)", "Warning", "Marking counts as ERROR.");
         for collections in intermediate_struct.values():
             for info in collections: info["documents"] = "ERROR"
    
    if WRITE_RAW_FILES_TO_DISK:
        logger.log("Save Output File (P2)", "Started", str(phase2_path))
        try:
            with open(phase2_path, 'w', encoding='utf-8') as f: json.dump(intermediate_struct, f, indent=2, ensure_ascii=False)
            logger.log("Save Output File (P2)", "Ended"); print(f"\n✓ Phase 2 results saved: {phase2_path}")
        except Exception as e: logger.log("Save Output File (P2)", "Error", str(e)); print(f"🛑 Error saving Phase 2: {e}")
    else:
        logger.log("Save Output File (P2)", "Skipped", "WRITE_RAW_FILES_TO_DISK is False.")
        
    logger.log("Phase 2", "Ended")
    return intermediate_struct

# --- PHASE 3 FUNCTION (Adapted from disc.py) ---

async def run_phase3_schema_inference(
    llm_client: DaprChatClient,
    tools: list, 
    base_instructions: List[str],
    bucket_name: str, 
    logger: PerformanceLogger,
    scope_collection_counts: Dict[str, List[Dict[str, Any]]] # Input from Phase 2
) -> Dict[str, Any]:
    """
    Phase 3: Get schema for non-empty collections using the robust,
    retry-enabled, stateless agent loop from disc.py.
    """
    logger.log("Phase 3", "Started", "Schema Inference (Stateless Agent Loop)")
    
    if WRITE_RAW_FILES_TO_DISK:
        RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    final_schema_context = {
        "bucket_name": bucket_name,
        "generated_at": datetime.now().isoformat(),
        "scopes": {}
    }

    if not scope_collection_counts:
        logger.log("Phase 3", "Skipped", "Input from P2 empty."); print("⚠️ No counts from P2.")
        return final_schema_context

    for scope_name, collections in scope_collection_counts.items():
        if scope_name == "_system": continue
        logger.log("Processing Scope (P3)", "Started", scope_name)
        
        if scope_name not in final_schema_context["scopes"]:
            final_schema_context["scopes"][scope_name] = {"collections": {}}
        
        for info in collections:
            coll_name = info.get("name")
            count = info.get("documents")

            if coll_name is None:
                logger.log("Skipping Entry (P3)", "Warning", f"Invalid entry in {scope_name}: {info}")
                continue
            
            if not isinstance(count, int):
                logger.log("Skipping Collection (P3)", "Error", f"{scope_name}.{coll_name} (Count is ERROR: {count})")
                final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                    "document_count": 0,
                    "content": [],
                    "error": f"Skipped (P2 count was ERROR: {count})"
                }
                continue

            if count > 0:
                logger.log("Target Collection (P3)", "Info", f"{scope_name}.{coll_name} ({count} docs)")

                safe_bucket = f"`{bucket_name.replace('`', '``')}`"
                safe_scope = f"`{scope_name.replace('`', '``')}`"
                safe_collection = f"`{coll_name.replace('`', '``')}`"
                infer_options = {
                    "sample_size": INFER_SAMPLE_SIZE,
                    "num_sample_values": INFER_NUM_SAMPLE_VALUES,
                    "similarity_metric": INFER_SIMILARITY_METRIC
                }
                infer_query = (
                    f"INFER {safe_bucket}.{safe_scope}.{safe_collection} "
                    f"WITH {json.dumps(infer_options)};"
                )

                safe_scope_name = re.sub(r'[^\w\-]+', '_', scope_name)
                safe_coll_name = re.sub(r'[^\w\-]+', '_', coll_name)
                output_filename = f"{safe_scope_name}_{safe_coll_name}_infer_raw.txt"
                output_filepath = RAW_RESPONSES_DIR / output_filename

                max_attempts = 3
                wait_times = [15, 30] 
                tool_args = {"query": infer_query, "scope_name": scope_name}
                
                inferred_schemas = []
                error_message_for_json = None
                
                for attempt in range(max_attempts):
                    
                    agent = create_infer_agent(llm_client, tools, base_instructions, logger)
                    
                    try:
                        if attempt > 0:
                            wait_duration = wait_times[attempt - 1]
                            logger.log("Rate Limit Retry", "Started", f"Attempt {attempt + 1}/{max_attempts}. Waiting {wait_duration}s...")
                            time.sleep(wait_duration)
                            logger.log("Rate Limit Cooldown", "Ended")

                        raw_resp_obj = await agent_run_tool_minimal(
                            agent, "CouchbaseMcpRunSqlPlusPlusQuery", tool_args, logger
                        )

                        # --- *** MODIFIED: Use new 3-value return parser *** ---
                        schemas_list, raw_str_for_file, parse_warning_or_error = parse_and_normalize_schema_content(raw_resp_obj, logger)
                        
                        # Store any error/warning for the final JSON
                        if parse_warning_or_error:
                            error_message_for_json = parse_warning_or_error
                        
                        # If parsing failed, schemas_list will be empty.
                        if not schemas_list:
                            # If there was no error message, create one.
                            if not error_message_for_json:
                                error_message_for_json = "ERROR: Parsing returned no schemas and no error message."
                            # Raise exception to log this as a non-retryable failure for this attempt
                            raise ValueError(error_message_for_json)
                        
                        # Success
                        inferred_schemas = schemas_list
                        
                        if WRITE_RAW_FILES_TO_DISK:
                            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                                f_out.write(f"--- Query ---\n{infer_query}\n\n")
                                f_out.write(f"--- Scope Name Arg ---\n{scope_name}\n\n")
                                f_out.write(f"--- Agent Response Object Type: {type(raw_resp_obj).__name__} ---\n\n")
                                f_out.write(raw_str_for_file)
                            logger.log("Save Raw Response", "Info", f"Saved to {output_filepath.name}")
                        
                        break # Success

                    except Exception as e:
                        error_message = str(e)
                        is_rate_limit_error = "429" in error_message and "exceeded the token rate limit" in error_message
                        
                        if is_rate_limit_error:
                            logger.log(f"Rate Limit Error (Attempt {attempt + 1})", "Warning", f"Hit 429 error for {scope_name}.{coll_name}.")
                            error_message_for_json = f"ERROR (429 Exhausted after {attempt + 1} attempts): {error_message}"
                            
                            if attempt == max_attempts - 1:
                                logger.log("INFER Execution (P3)", "Error", f"Failed for {scope_name}.{coll_name} after {max_attempts} attempts (429 Rate Limit). Skipping.")
                                if WRITE_RAW_FILES_TO_DISK:
                                    try:
                                         with open(output_filepath, 'w', encoding='utf-8') as f_err:
                                             f_err.write(f"--- Query ---\n{infer_query}\n\n")
                                             f_err.write(f"--- Scope Name Arg ---\n{scope_name}\n\n")
                                             f_err.write(f"--- ERROR (429 Exhausted) ---\n{error_message}\n")
                                    except Exception as write_err:
                                         print(f"      Additionally failed to write error file: {write_err}")
                            else:
                                continue 
                        
                        else:
                            # This now also catches the ValueError from a failed parse
                            logger.log("INFER Execution (P3)", "Error", f"Failed for {scope_name}.{coll_name} with non-retryable error: {e}")
                            error_message_for_json = f"ERROR (Non-Retryable): {error_message}"
                            if WRITE_RAW_FILES_TO_DISK:
                                try:
                                     with open(output_filepath, 'w', encoding='utf-8') as f_err:
                                         f_err.write(f"--- Query ---\n{infer_query}\n\n")
                                         f_err.write(f"--- Scope Name Arg ---\n{scope_name}\n\n")
                                         f_err.write(f"--- ERROR (Non-Retryable) ---\n{error_message}\n")
                                         import traceback
                                         f_err.write("\n--- Traceback ---\n")
                                         traceback.print_exc(file=f_err)
                                except Exception as write_err:
                                     print(f"      Additionally failed to write error file: {write_err}")
                            break 

                final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                    "document_count": count,
                    "content": inferred_schemas, 
                    "error": error_message_for_json
                }

            else: # count == 0
                logger.log("Skipping Collection (P3)", "Info", f"{scope_name}.{coll_name} (0 docs)")
                final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                    "document_count": 0,
                    "content": [],
                    "error": "Skipped (0 documents)"
                }
        logger.log("Processing Scope (P3)", "Ended", scope_name)
    
    logger.log("Phase 3", "Ended")
    return final_schema_context


# --- MAIN ORCHESTRATOR (from discovery - Copy.py) ---

async def main_discovery() -> Optional[Dict[str, Any]]: # <-- MODIFIED: Added return type
    """Runs Phase 1, 2 and 3 sequentially."""
    logger = PerformanceLogger()
    logger.log("Main Application Start", "Started", "Running Full Discovery (P1, P2, P3)")
    
    SCHEMA_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    mcp_url = os.getenv("MCP_SERVER_URL")
    bucket_name = os.getenv("COUCHBASE_BUCKET", "travel-sample")
    temperature = float(os.getenv("AGENT_TEMPERATURE", 0.0))
    
    final_output_filename = f"{bucket_name}_schema_final_complete.json"
    final_output_path = SCHEMA_CONTEXT_DIR / final_output_filename
    
    final_schema_result = None # <-- NEW: Define to ensure it's in scope

    if not mcp_url:
        logger.log("Config Validation", "Error", "MCP_SERVER_URL not set.")
        print("🛑 Error: MCP_SERVER_URL not set.")
        logger.log("Main Application End", "Ended", "Config Error.")
        logger.save_to_file()
        return None # <-- MODIFIED: Return on error

    client = None; agent = None; tools = None
    try:
        logger.log("Setup", "Started", "Connecting to MCP and initializing components.")
        client = MCPClient(timeout=180.0)
        await client.connect_sse(server_name="couchbase_mcp", url=mcp_url, headers=None)
        logger.log("Connect to MCP Server", "Ended")
        
        tools = client.get_all_tools()
        if not tools: raise RuntimeError("MCP Client did not discover any tools.")
        logger.log("Get MCP Tools", "Ended", f"Found {len(tools)} tools")

        p1_p2_instructions = [
            "You are a Couchbase SQL++ and MCP tooling expert.",
            "Always run the exact named tool with the provided JSON args.",
            "Output only the raw JSON or raw text result from the tool with no extra explanation, prose or code fences.",
        ]
        agent = Agent( 
            name="CbDiscoveryAgent", 
            role="Couchbase Discovery Agent", 
            instructions=p1_p2_instructions, 
            tools=tools, 
            temperature=temperature
        )
        logger.log("Create Dapr Agent (for P1/P2)", "Ended")

        p3_instructions = [
            "You are an agent designed to execute MCP tools.",
            "Run the specified tool with the provided arguments.",
            "Return ONLY the raw result from the tool without modification, explanation, or formatting."
        ]
        logger.log("Init Dapr LLM Client (for P3)", "Started", f"Using component: {os.getenv('DAPR_LLM_COMPONENT_DEFAULT')}")
        llm_client = DaprChatClient(
            provider= os.getenv("DAPR_LLM_PROVIDER", "openai"),
            temperature=temperature
        )
        logger.log("Init Dapr LLM Client (for P3)", "Ended")
        logger.log("Setup", "Ended")

        # --- Run Phase 1 ---
        phase1_result = await run_phase1_discovery(agent, bucket_name, logger)
        
        if not isinstance(phase1_result, dict) or not all(isinstance(v, list) for v in phase1_result.values()):
             logger.log("Phase 1 Output", "Error", f"P1 result is not the expected Dict[str, List[str]]. Got: {type(phase1_result)}")
             raise TypeError("Phase 1 did not return the expected scope/collection map.")
        
        # --- Run Phase 2 ---
        phase2_result = await run_phase2_doc_count(agent, bucket_name, logger, scope_collection_map=phase1_result)
        
        # --- Run Phase 3 ---
        final_schema_result = await run_phase3_schema_inference(
            llm_client=llm_client,
            tools=tools,
            base_instructions=p3_instructions,
            bucket_name=bucket_name,
            logger=logger,
            scope_collection_counts=phase2_result
        )

        # --- Save Final Result (This always runs, regardless of switch) ---
        logger.log("Save Final Output", "Started", str(final_output_path))
        with open(final_output_path, 'w', encoding='utf-8') as f: json.dump(final_schema_result, f, indent=2, ensure_ascii=False)
        logger.log("Save Final Output", "Ended")
        print(f"\n✓✓✓ Final Schema saved to: {final_output_path} ✓✓✓")
        
        log_message("✅✅✅ Discovery Process (All Phases) Completed Successfully ✅✅✅")
        
    except Exception as e:
        log_entry = logger.log("Main Application", "CRITICAL ERROR", str(e))
        print(f"\n❌ A critical error stopped the discovery process (Log ID: {log_entry['log_id']}): {e}")
        import traceback; traceback.print_exc()
        return None # <-- MODIFIED: Return on error
    finally:
        logger.log("Main Application End", "Ended")
        logger.save_to_file()
        if client:
            try: await client.close(); log_message("🔌 MCP client connection closed.")
            except Exception as close_err: log_message(f"  ⚠️ Error closing MCP client: {close_err}")

    # --- *** MODIFIED: Return the final object *** ---
    return final_schema_result


if __name__ == "__main__":
    log_message("Script starting, ensure Dapr sidecar is running...")
    
    # --- *** MODIFIED: Capture and check return value *** ---
    final_result = asyncio.run(main_discovery())
    
    if final_result:
        print("\n" + "="*50)
        print("--- FINAL CONSOLIDATED SCHEMA CONTEXT (Summary) ---")
        print("="*50)
        print(json.dumps(final_result, indent=2, ensure_ascii=False))
    else:
        log_message("Script finished with errors. No final result was returned.")