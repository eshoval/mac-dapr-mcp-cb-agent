"""
discovery.py - Refactored Version with Separated Concerns
Complete schema discovery system with clean architecture

Architecture:
- MCPExecutor: Handles all MCP tool execution + retry
- ResponseParser: Handles all JSON parsing + validation
- FileManager: Handles all file I/O
- PerformanceLogger: Handles all logging
- discovery.py: Orchestration only (business logic)
"""

import os
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# Import our modules
from performance_logger import PerformanceLogger
from response_parser import ResponseParser
from mcp_executor import MCPExecutor
from file_manager import FileManager

# Dapr Agents Imports
try:
    from dapr_agents import Agent
    from dapr_agents.tool.mcp.client import MCPClient
    from dapr_agents.llm.dapr import DaprChatClient
except ImportError:
    print("🛑 Error: dapr_agents library not found. Please install it (`pip install dapr-agents`).")
    exit(1)

# Load environment variables
load_dotenv()

# --- Configuration ---
SCHEMA_CONTEXT_DIR = Path("./schema_context")
LOGS_DIR = Path("./logs")
RAW_RESPONSES_DIR = SCHEMA_CONTEXT_DIR / "infer_results_full"

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "max_retries": 3,  
    "initial_delay": 5.0,
    "backoff_multiplier": 2.0,
    "delay_between_infer": 4.0,
}

# Adaptive INFER configuration based on collection size
INFER_THRESHOLDS = {
    'very_small': {
        'max_docs': 10,
        'sample_size': 'ALL',
        'num_sample_values': 1,
        'similarity_metric': 0.3
    },
    'small': {
        'max_docs': 100,
        'sample_size': 50,
        'num_sample_values': 1,
        'similarity_metric': 0.4
    },
    'medium': {
        'max_docs': 1000,
        'sample_size': 200,
        'num_sample_values': 2,
        'similarity_metric': 0.5
    },
    'large': {
        'max_docs': float('inf'),
        'sample_size': 1000,
        'num_sample_values': 3,
        'similarity_metric': 0.6
    }
}


def get_adaptive_infer_params(doc_count: int) -> dict:
    """
    Calculate optimal INFER parameters based on collection size.
    Reduces complexity for small collections to prevent LLM errors.
    
    Args:
        doc_count: Number of documents in the collection
    
    Returns:
        Dictionary with sample_size, num_sample_values, similarity_metric
    """
    for tier, config in INFER_THRESHOLDS.items():
        if doc_count <= config['max_docs']:
            params = {
                'sample_size': doc_count if config['sample_size'] == 'ALL' else min(doc_count, config['sample_size']),
                'num_sample_values': config['num_sample_values'],
                'similarity_metric': config['similarity_metric']
            }
            return params
    
    # Fallback to default (should never reach here)
    return {
        'sample_size': 1000,
        'num_sample_values': 3,
        'similarity_metric': 0.6
    }


def log_message(message: str):
    """Print message with timestamp"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} - {message}")


# ============================================================
# PHASE 1: Discover Scopes & Collections
# ============================================================

async def run_phase1_discovery(
    mcp: MCPExecutor,
    parser: ResponseParser,
    files: FileManager,
    logger: PerformanceLogger,
    bucket_name: str
) -> Optional[Dict[str, List[str]]]:
    """
    Phase 1: Discover scopes/collections
    Clean separation: MCP → Parse → Save → Return
    """
    logger.log("Phase 1", "Started", "Discover Scopes & Collections")
    
    try:
        print(f"\n🔍 Discovering scopes/collections for bucket: {bucket_name}...\n")
        
        # Step 1: Execute MCP tool
        raw_resp = await mcp.execute(
            tool_name="CouchbaseMcpGetScopesAndCollectionsInBucket",
            args={}
        )
        
        # Step 2: Parse response
        structured_data, error = parser.parse(
            raw_response=raw_resp,
            expected_type="dict",
            phase_name="P1"
        )
        
        if structured_data is None:
            raise ValueError(f"Phase 1 parsing failed: {error}")
        
        if error:  # Warning (but data is valid)
            logger.log("Phase 1", "Warning", error)
        
        # Step 3: Validate structure
        logger.log("Validate Structure (P1)", "Started")
        if not all(isinstance(v, list) for v in structured_data.values()):
            raise TypeError(f"Phase 1: Expected Dict[str, List[str]], got invalid structure")
        logger.log("Validate Structure (P1)", "Ended", "Valid")
        
        # Step 4: Save checkpoint
        output_path = files.save_json(structured_data, f"{bucket_name}_schema_phase1.json")
        print(f"✅ Phase 1 Checkpoint saved: {output_path}")
        
        logger.log("Phase 1", "Ended")
        return structured_data
        
    except Exception as e:
        logger.log("Phase 1", "Error", str(e))
        print(f"🛑 Error during Phase 1: {e}")
        raise


# ============================================================
# PHASE 2: Document Counting
# ============================================================

async def run_phase2_doc_count(
    mcp: MCPExecutor,
    parser: ResponseParser,
    files: FileManager,
    logger: PerformanceLogger,
    bucket_name: str,
    scope_collection_map: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Phase 2: Count documents in all collections
    Clean separation: Load → Build Query → Execute → Parse → Update → Save
    """
    logger.log("Phase 2", "Started", "Document Counting")
    
    intermediate_struct: Dict[str, List[Dict[str, Any]]] = {}
    
    # Step 1: Load Phase 1 data if not provided
    if scope_collection_map is None:
        logger.log("Load Phase 1 Data (P2)", "Started")
        try:
            scope_collection_map = files.load_json(f"{bucket_name}_schema_phase1.json")
        except FileNotFoundError:
            logger.log("Load Phase 1 Data (P2)", "Error", "Phase 1 file not found")
            raise
        logger.log("Load Phase 1 Data (P2)", "Ended")
    else:
        logger.log("Load Phase 1 Data (P2)", "Skipped", "Data provided in-memory")
    
    # Step 2: Transform data
    logger.log("Transform Data (P2)", "Started")
    collections_to_query: List[Dict[str, str]] = []
    
    for scope, collections in scope_collection_map.items():
        if scope == "_system":
            continue
        if not isinstance(collections, list):
            logger.log("Transform Data (P2)", "Warning", f"Skipping scope '{scope}': expected list")
            continue
        
        intermediate_struct[scope] = []
        for coll in collections:
            if not isinstance(coll, str):
                logger.log("Transform Data (P2)", "Warning", f"Skipping invalid collection in '{scope}': {coll}")
                continue
            
            intermediate_struct[scope].append({"name": coll, "documents": "UNDEFINED"})
            collections_to_query.append({"scope": scope, "collection": coll})
    
    logger.log("Transform Data (P2)", "Ended", f"Prepared {len(collections_to_query)} collections")
    
    # Step 3: Handle empty case
    if not collections_to_query:
        logger.log("Phase 2", "Skipped", "No collections to count")
        print("⚠️ No collections to count.")
        files.save_json(intermediate_struct, f"{bucket_name}_schema_phase2.json")
        logger.log("Phase 2", "Ended")
        return intermediate_struct
    
    # Step 4: Generate UNION ALL query
    logger.log("Generate Query (P2)", "Started")
    union_parts = []
    for item in collections_to_query:
        scope, collection = item['scope'], item['collection']
        query_part = (
            f'SELECT "{scope}.{collection}" AS scope_collection, '
            f'COUNT(1) AS doc_count FROM `{bucket_name}`.`{scope}`.`{collection}`'
        )
        union_parts.append(query_part)
    
    full_query = "\nUNION ALL\n".join(union_parts) + ";"
    logger.log("Generate Query (P2)", "Ended", f"Length: {len(full_query)}")
    print("\n📝 Generated SQL++ Query (P2):\n---\n" + full_query + "\n---\n")
    
    # Step 5: Execute query
    print(f"⚙️ Executing UNION ALL via Agent...\n")
    
    try:
        raw_resp = await mcp.execute(
            tool_name="CouchbaseMcpRunSqlPlusPlusQuery",
            args={"query": full_query, "scope_name": ""}
        )
        
        # Step 6: Parse response
        doc_counts_result, error = parser.parse(
            raw_response=raw_resp,
            expected_type="list",
            phase_name="P2"
        )
        
        if doc_counts_result is None:
            raise ValueError(f"Phase 2 parsing failed: {error}")
        
        if error:  # Warning
            logger.log("Phase 2", "Warning", error)
        
        logger.log("Parse Count Results (P2)", "Ended", f"Parsed {len(doc_counts_result)} entries")
        
        # Step 7: Build count map
        count_map = {
            item.get("scope_collection"): item.get("doc_count", "ERROR")
            for item in doc_counts_result if isinstance(item, dict)
        }
        
        # Step 8: Update structure
        logger.log("Update Structure (P2)", "Started")
        updated, not_found = 0, 0
        
        for scope, collections in intermediate_struct.items():
            for info in collections:
                full_name = f"{scope}.{info['name']}"
                if full_name in count_map:
                    info["documents"] = count_map[full_name]
                    updated += 1
                else:
                    info["documents"] = 0
                    not_found += 1
                    logger.log("Update Structure (P2)", "Warning", f"{full_name} not in results, assuming 0")
        
        logger.log("Update Structure (P2)", "Ended", f"Updated: {updated}, Assumed 0: {not_found}")
        
    except Exception as e:
        logger.log("Phase 2 Execution", "Error", f"UNION ALL failed: {e}")
        print(f"🛑 Error during Phase 2: {e}")
        
        # Mark all as ERROR
        logger.log("Update Structure (P2)", "Warning", "Marking counts as ERROR")
        for collections in intermediate_struct.values():
            for info in collections:
                info["documents"] = "ERROR"
    
    # Step 9: Save output
    output_path = files.save_json(intermediate_struct, f"{bucket_name}_schema_phase2.json")
    print(f"\n✅ Phase 2 results saved: {output_path}")
    
    logger.log("Phase 2", "Ended")
    return intermediate_struct


# ============================================================
# PHASE 3: Schema Inference
# ============================================================

async def run_phase3_schema_inference(
    llm_client: DaprChatClient,
    tools: list,
    base_instructions: List[str],
    parser: ResponseParser,
    logger: PerformanceLogger,
    bucket_name: str,
    scope_collection_counts: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Phase 3: Get schema for non-empty collections
    Clean separation: Loop → Create Agent → Execute → Parse → Save Raw → Record
    """
    logger.log("Phase 3", "Started", "Schema Inference")
    
    # Prepare output directory
    RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    
    final_schema_context = {
        "bucket_name": bucket_name,
        "generated_at": datetime.now().isoformat(),
        "scopes": {}
    }
    
    if not scope_collection_counts:
        logger.log("Phase 3", "Skipped", "Input from P2 empty")
        print("⚠️ No counts from P2.")
        return final_schema_context
    
    # Process each scope
    for scope_name, collections in scope_collection_counts.items():
        if scope_name == "_system":
            continue
        
        logger.log("Processing Scope (P3)", "Started", scope_name)
        
        if scope_name not in final_schema_context["scopes"]:
            final_schema_context["scopes"][scope_name] = {"collections": {}}
        
        # Process each collection
        for info in collections:
            coll_name = info.get("name")
            count = info.get("documents")
            
            if coll_name is None:
                logger.log("Skipping Entry (P3)", "Warning", f"Invalid entry in {scope_name}: {info}")
                continue
            
            # Handle ERROR count
            if not isinstance(count, int):
                logger.log("Skipping Collection (P3)", "Error", f"{scope_name}.{coll_name} (Count is ERROR: {count})")
                final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                    "document_count": 0,
                    "content": [],
                    "error": f"Skipped (P2 count was ERROR: {count})"
                }
                continue
            
            # Handle empty collection
            if count == 0:
                logger.log("Skipping Collection (P3)", "Info", f"{scope_name}.{coll_name} (0 docs)")
                final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                    "document_count": 0,
                    "content": [],
                    "error": "Skipped (0 documents)"
                }
                continue
            
            # Process non-empty collection
            logger.log("Target Collection (P3)", "Info", f"{scope_name}.{coll_name} ({count} docs)")
            
            # Build INFER query with adaptive parameters
            safe_bucket = f"`{bucket_name.replace('`', '``')}`"
            safe_scope = f"`{scope_name.replace('`', '``')}`"
            safe_collection = f"`{coll_name.replace('`', '``')}`"
            
            adaptive_params = get_adaptive_infer_params(count)
            infer_options = adaptive_params
            
            logger.log(
                "Adaptive INFER Config (P3)",
                "Info",
                f"{scope_name}.{coll_name} ({count} docs) -> "
                f"sample={adaptive_params['sample_size']}, "
                f"values={adaptive_params['num_sample_values']}, "
                f"similarity={adaptive_params['similarity_metric']}"
            )
            
            infer_query = (
                f"INFER {safe_bucket}.{safe_scope}.{safe_collection} "
                f"WITH {json.dumps(infer_options)};"
            )
            
            # Prepare output file
            safe_scope_name = re.sub(r'[^\w\-]+', '_', scope_name)
            safe_coll_name = re.sub(r'[^\w\-]+', '_', coll_name)
            output_filename = f"{safe_scope_name}_{safe_coll_name}_infer_raw.txt"
            output_filepath = RAW_RESPONSES_DIR / output_filename
            
            # Create dedicated agent for this collection
            logger.log("Create Agent (P3)", "Started", f"For {scope_name}.{coll_name}")
            agent_for_collection = Agent(
                name="InferHelperAgent",
                role="Tool Execution Agent",
                instructions=base_instructions,
                tools=tools,
                llm=llm_client
            )
            logger.log("Create Agent (P3)", "Ended")
            
            # Create MCPExecutor for this agent
            mcp_for_collection = MCPExecutor(
                agent=agent_for_collection,
                retry_config=RATE_LIMIT_CONFIG,
                logger=logger
            )
            
            inferred_schemas = []
            error_message_for_json = None
            
            try:
                # Execute INFER
                raw_resp_obj = await mcp_for_collection.execute(
                    tool_name="CouchbaseMcpRunSqlPlusPlusQuery",
                    args={"query": infer_query, "scope_name": scope_name}
                )
                # --- 🛑 FIX START: Save raw response even on parse errors ---
                #try:
                #    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                #        f_out.write(f"--- Query ---\n{infer_query}\n\n")
                #        f_out.write(f"--- Scope Name Arg ---\n{scope_name}\n\n")
                #        f_out.write(f"--- Agent Response Object Type: {type(raw_resp_obj).__name__} ---\n\n")
                        
                        # שימוש בפונקציה הפנימית של הפארסר לנרמול הטקסט לכתיבה
                #        content_to_write = parser._normalize(raw_resp_obj)
                #        f_out.write(content_to_write)
                        
                #    logger.log("Save Raw Response (P3)", "Info", f"Saved raw output to {output_filepath.name}")
                #except Exception as write_err:
                #    logger.log("Save Raw Response (P3)", "Error", f"Failed to write raw file: {write_err}")
                # --- 🛑 FIX END ---

                
                # Fixed delay between INFER queries
                await asyncio.sleep(RATE_LIMIT_CONFIG["delay_between_infer"])
                
                # Parse response
                parsed_data, parse_warning_or_error = parser.parse(
                    raw_response=raw_resp_obj,
                    expected_type=None,  # Allow both dict and list
                    phase_name="P3"
                )
                
                if parse_warning_or_error:
                    error_message_for_json = parse_warning_or_error
                
                # Handle empty response
                if parsed_data is None:
                    if "Empty response" in str(parse_warning_or_error):
                        logger.log("INFER Execution (P3)", "Warning",
                                 f"INFER for {scope_name}.{coll_name} returned empty. Skipping.")
                        error_message_for_json = "WARNING: INFER returned an empty response."
                        schemas_list = []
                    else:
                        error_message_for_json = f"ERROR: Parsing returned None. {parse_warning_or_error}"
                        raise ValueError(error_message_for_json)
                
                # Normalize response to list
                elif isinstance(parsed_data, dict):
                    schemas_list = [parsed_data]
                    logger.log("Parse Response (P3)", "Info", "Normalized single dict to list")
                elif isinstance(parsed_data, list):
                    schemas_list = parsed_data
                else:
                    raise TypeError(f"INFER parsing returned unexpected type: {type(parsed_data)}")
                
                # Success
                inferred_schemas = schemas_list
                
                # Save raw response to disk
                #try:
                #    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                #        f_out.write(f"--- Query ---\n{infer_query}\n\n")
                #        f_out.write(f"--- Scope Name Arg ---\n{scope_name}\n\n")
                #        f_out.write(f"--- Agent Response Object Type: {type(raw_resp_obj).__name__} ---\n\n")
                        # Normalize content for file
                #        content = parser._normalize(raw_resp_obj)
                #        f_out.write(content)
                #    logger.log("Save Raw Response (P3)", "Info", f"Saved to {output_filepath.name}")
                #except Exception as write_err:
                #    logger.log("Save Raw Response (P3)", "Error", str(write_err))
            
            except Exception as e:
                # All retries exhausted or non-retryable error
                error_message = str(e)
                
                logger.log("INFER Execution (P3)", "Error",
                         f"Failed for {scope_name}.{coll_name}: {error_message[:200]}")
                
                error_message_for_json = f"ERROR: {error_message}"
                
                # Save error to file
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f_err:
                        f_err.write(f"--- ERROR (All Retries Exhausted) ---\n{error_message}\n")
                        import traceback
                        f_err.write("\n--- Traceback ---\n")
                        traceback.print_exc(file=f_err)
                except Exception as write_err:
                    print(f"      Additionally failed to write error file: {write_err}")
            
            # Record result
            final_schema_context["scopes"][scope_name]["collections"][coll_name] = {
                "document_count": count,
                "content": inferred_schemas,
                "error": error_message_for_json
            }
        
        logger.log("Processing Scope (P3)", "Ended", scope_name)
    
    logger.log("Phase 3", "Ended")
    return final_schema_context


# ============================================================
# PHASE 4: Index Discovery
# ============================================================

async def run_phase4_index_discovery(
    mcp: MCPExecutor,
    parser: ResponseParser,
    logger: PerformanceLogger,
    bucket_name: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Phase 4: Discover GSI indexes
    Clean separation: Build Query → Execute → Parse → Return
    """
    logger.log("Phase 4", "Started", "Discover GSI Indexes")
    
    index_query = (
        f"SELECT name, definition "
        f"FROM system:indexes "
        f"WHERE bucket_id = '{bucket_name}' "
        f"AND `using` = 'gsi' "
        f"AND scope_id != '_default';"
    )
    
    try:
        print(f"\n🔍 Discovering GSI indexes for bucket: {bucket_name}...\n")
        
        # Execute query
        raw_resp = await mcp.execute(
            tool_name="CouchbaseMcpRunSqlPlusPlusQuery",
            args={"query": index_query, "scope_name": ""}
        )
        
        # Parse response
        index_data, error = parser.parse(
            raw_response=raw_resp,
            expected_type="list",
            phase_name="P4"
        )
        
        if index_data is None:
            logger.log("Parse Index Content (P4)", "Warning", f"Parsing failed: {error}")
            index_list = []
        else:
            if error:  # Warning
                logger.log("Phase 4", "Warning", error)
            index_list = index_data
            logger.log("Parse Index Content (P4)", "Info", f"Found {len(index_list)} indexes")
        
        logger.log("Phase 4", "Ended")
        return index_list
        
    except Exception as e:
        logger.log("Phase 4", "Error", str(e))
        print(f"🛑 Error during Phase 4 (Index Discovery): {e}")
        return None


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

async def main_discovery() -> Optional[Dict[str, Any]]:
    """
    Main orchestrator - Runs all 4 phases sequentially
    Clean initialization → Phase execution → Cleanup
    """
    # Initialize logger
    logger = PerformanceLogger()
    logger.log("Main Application Start", "Started", "Running Full Discovery (P1, P2, P3, P4)")
    
    # Create directories
    SCHEMA_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    mcp_url = os.getenv("MCP_SERVER_URL")
    bucket_name = os.getenv("COUCHBASE_BUCKET", "travel-sample")
    temperature = float(os.getenv("AGENT_TEMPERATURE", 0.0))
    
    final_output_filename = f"{bucket_name}_schema_final_complete.json"
    final_schema_result = None
    
    if not mcp_url:
        logger.log("Config Validation", "Error", "MCP_SERVER_URL not set")
        print("🛑 Error: MCP_SERVER_URL not set.")
        logger.log("Main Application End", "Ended", "Config Error")
        logger.save_to_file(logs_dir=LOGS_DIR)
        return None
    
    client = None
    agent = None
    
    try:
        # ==================== SETUP ====================
        logger.log("Setup", "Started", "Connecting to MCP")
        
        # Connect to MCP
        client = MCPClient(timeout=180.0)
        await client.connect_sse(server_name="couchbase_mcp", url=mcp_url, headers=None)
        logger.log("Connect to MCP Server", "Ended")
        
        # Get tools
        tools = client.get_all_tools()
        if not tools:
            raise RuntimeError("MCP Client did not discover any tools")
        logger.log("Get MCP Tools", "Ended", f"Found {len(tools)} tools")
        
        # Initialize LLM client
        logger.log("Init Dapr LLM Client", "Started")
        llm_client = DaprChatClient(
            provider=os.getenv("DAPR_LLM_PROVIDER", "googleai"),
            temperature=temperature,
            timeout=180
        )
        logger.log("Init Dapr LLM Client", "Ended")
        
        # Create main agent (for P1, P2, P4)
        p_all_instructions = [
            "You are a tool execution agent. Your ONLY job is to execute ONE tool and return its output.",
            "",
            "MANDATORY WORKFLOW:",
            "1. User will request a tool execution",
            "2. Call that tool ONCE with the provided arguments",
            "3. You will receive the tool's output (as a 'tool' message)",
            "4. Your FINAL response must be ONLY the raw content from the tool - NOTHING ELSE",
            "",
            "CRITICAL RULES:",
            "- Do NOT analyze, interpret, or transform the tool's output",
            "- Do NOT add explanations, summaries, formatting, or commentary",
            "- Do NOT respond with empty messages, 'done', or 'finished'",
            "- DO NOT call any additional tools, even if they seem related",
            "- DO NOT chain multiple tool calls together",
            "- Your ENTIRE response = the tool's raw output",
        ]
        
        agent = Agent(
            name="CbDiscoveryAgent",
            role="Couchbase Discovery Agent",
            instructions=p_all_instructions,
            llm=llm_client,
            tools=tools,
            temperature=temperature
        )
        logger.log("Create Dapr Agent (for P1/P2/P4)", "Ended")
        
        # Initialize helper classes
        mcp_executor = MCPExecutor(agent=agent, retry_config=RATE_LIMIT_CONFIG, logger=logger)
        response_parser = ResponseParser(logger=logger)
        file_manager = FileManager(base_dir=SCHEMA_CONTEXT_DIR, logger=logger)
        
        logger.log("Setup", "Ended")
        
        # ==================== RUN PHASES ====================
        
        # Phase 1: Discover Scopes & Collections
        phase1_result = await run_phase1_discovery(
            mcp=mcp_executor,
            parser=response_parser,
            files=file_manager,
            logger=logger,
            bucket_name=bucket_name
        )
        
        # Phase 2: Document Counting
        phase2_result = await run_phase2_doc_count(
            mcp=mcp_executor,
            parser=response_parser,
            files=file_manager,
            logger=logger,
            bucket_name=bucket_name,
            scope_collection_map=phase1_result
        )
        
        # Phase 3: Schema Inference
        final_schema_result = await run_phase3_schema_inference(
            llm_client=llm_client,
            tools=tools,
            base_instructions=p_all_instructions,
            parser=response_parser,
            logger=logger,
            bucket_name=bucket_name,
            scope_collection_counts=phase2_result
        )
        
        # Phase 4: Index Discovery
        if final_schema_result:
            phase4_result = await run_phase4_index_discovery(
                mcp=mcp_executor,
                parser=response_parser,
                logger=logger,
                bucket_name=bucket_name
            )
            
            if phase4_result is not None:
                final_schema_result["available_indexes"] = phase4_result
                logger.log("Phase 4 Result", "Info", f"Added {len(phase4_result)} indexes to final schema")
            else:
                logger.log("Phase 4 Result", "Warning", "Index discovery failed. Schema will not include indexes")
        else:
            logger.log("Phase 4", "Skipped", "P3 did not return a valid schema")
        
        # ==================== SAVE FINAL RESULT ====================
        
        if final_schema_result:
            output_path = file_manager.save_json(final_schema_result, final_output_filename)
            print(f"\n✅✅✅ Final Schema saved to: {output_path} ✅✅✅")
            log_message("✅✅✅ Discovery Process (All Phases) Completed Successfully ✅✅✅")
        
    except Exception as e:
        log_entry = logger.log("Main Application", "CRITICAL ERROR", str(e))
        print(f"\n❌ A critical error stopped the discovery process (Log ID: {log_entry['log_id']}): {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        logger.log("Main Application End", "Ended")
        logger.save_to_file(logs_dir=LOGS_DIR)
        
        if client:
            try:
                await client.close()
                log_message("🔌 MCP client connection closed.")
            except Exception as close_err:
                log_message(f"  ⚠️ Error closing MCP client: {close_err}")
    
    return final_schema_result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    log_message("Script starting, ensure Dapr sidecar is running...")
    
    final_result = asyncio.run(main_discovery())
    
    if final_result:
        print("\n" + "="*50)
        print("--- FINAL CONSOLIDATED SCHEMA CONTEXT (Summary) ---")
        print("="*50)
        print(f"Bucket: {final_result.get('bucket_name')}")
        print(f"Generated At: {final_result.get('generated_at')}")
        print(f"Scopes Found: {len(final_result.get('scopes', {}))}")
        print(f"Indexes Found: {len(final_result.get('available_indexes', []))}")
        
        # Print summary per scope
        for scope_name, scope_data in final_result.get('scopes', {}).items():
            collections = scope_data.get('collections', {})
            print(f"\n  Scope: {scope_name}")
            print(f"    Collections: {len(collections)}")
            
            errors = sum(1 for c in collections.values() if (c.get('error') or '').startswith('ERROR'))
            
            inferred = 0
            empty = 0
            errors = 0
            # Count collections by status
            for c in collections.values():
                # Safely get error string, handling None and missing keys
                err_val = c.get('error')
                err_str = str(err_val) if err_val is not None else ""
                
                if c.get('document_count', 0) == 0:
                    empty += 1
                elif err_str.startswith('ERROR'):
                    errors += 1
                elif c.get('content'):
                    inferred += 1

            print(f"      ├─ Inferred: {inferred}")
            print(f"      ├─ Empty: {empty}")
            print(f"      └─ Errors: {errors}")
    else:
        log_message("Script finished with errors. No final result was returned.")