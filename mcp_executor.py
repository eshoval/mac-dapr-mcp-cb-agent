"""
MCP Executor - Tool execution with automatic retry on rate limits
Handles: Agent execution, rate limit detection, exponential backoff
"""

import asyncio
import json
from typing import Dict, Any, Optional

# Rate limit detection patterns (case-insensitive)
RATE_LIMIT_PATTERNS = [
    # HTTP codes
    "429", "503", "408",
    # Generic patterns
    "rate limit", "rate_limit", "ratelimit",
    "quota exceeded", "quota_exceeded",
    "too many requests", "too_many_requests",
    "throttled", "throttling",
    # Provider-specific
    "resource_exhausted",  # Google/Gemini
    "tokens_per_min",      # OpenAI
    "requests_per_minute", # OpenAI
    "insufficient_quota",  # OpenAI
    "rate_limit_exceeded", # Anthropic
]


class MCPExecutor:
    """
    MCP tool executor with retry logic.
    
    Features:
    - Automatic rate limit detection across different LLM providers
    - Exponential backoff retry strategy
    - Detailed logging of execution and retries
    
    Responsibilities:
    - Execute MCP tools via agent
    - Detect and handle rate limit errors
    - Retry with exponential backoff
    - Log execution details
    
    Does NOT handle:
    - Response parsing (delegated to ResponseParser)
    - File I/O (delegated to FileManager)
    """
    
    def __init__(self, agent, retry_config: Dict[str, Any], logger=None):
        """
        Initialize MCP executor.
        
        Args:
            agent: Dapr Agent instance
            retry_config: Retry configuration dict with keys:
                - max_retries: Maximum retry attempts
                - initial_delay: Initial delay in seconds
                - backoff_multiplier: Exponential backoff multiplier
            logger: PerformanceLogger instance (optional)
        """
        self.agent = agent
        self.config = retry_config
        self.logger = logger
    
    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None
    ) -> object:
        """
        Execute MCP tool with automatic retry on rate limits.
        
        Args:
            tool_name: Name of the MCP tool to execute
            args: Tool arguments (dict)
            max_retries: Override max retries (optional)
            initial_delay: Override initial delay (optional)
        
        Returns:
            Raw response object from agent
        
        Raises:
            Exception: If all retries exhausted or non-retryable error occurs
        """
        # Use config defaults if not specified
        if max_retries is None:
            max_retries = self.config.get("max_retries", 1)
        if initial_delay is None:
            initial_delay = self.config.get("initial_delay", 5.0)
        
        backoff_multiplier = self.config.get("backoff_multiplier", 2.0)
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                # Build execution prompt
                prompt = self._build_prompt(tool_name, args)
                
                # Log execution start
                if self.logger:
                    args_summary = self._format_args_for_log(args)
                    self.logger.log(
                        f"MCP Execute ({tool_name})",
                        "Started",
                        f"Attempt {attempt + 1}/{max_retries} | Args: {args_summary}"
                    )
                
                # Execute via agent
                result = await self.agent.run(prompt)
                
                # Log success
                if self.logger:
                    self.logger.log(
                        f"MCP Execute ({tool_name})",
                        "Ended",
                        f"Success on attempt {attempt + 1}"
                    )
                
                return result
            
            except Exception as e:
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = initial_delay * (backoff_multiplier ** attempt)
                        
                        if self.logger:
                            self.logger.log(
                                f"MCP Retry ({tool_name})",
                                "Warning",
                                f"Rate limit detected (attempt {attempt + 1}/{max_retries}). "
                                f"Retrying in {delay:.1f}s... Error: {str(e)[:100]}"
                            )
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Max retries reached
                        error_msg = f"Rate limit error after {max_retries} retries: {str(e)}"
                        if self.logger:
                            self.logger.log(
                                f"MCP Execute ({tool_name})",
                                "Error",
                                error_msg
                            )
                        raise Exception(error_msg) from e
                else:
                    # Non-retryable error - raise immediately
                    if self.logger:
                        self.logger.log(
                            f"MCP Execute ({tool_name})",
                            "Error",
                            f"Non-retryable error: {str(e)[:200]}"
                        )
                    raise
        
        # Should never reach here, but just in case
        raise Exception(f"Failed to execute {tool_name} after {max_retries} attempts")
    
    def _build_prompt(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Build execution prompt for agent.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
        
        Returns:
            Formatted prompt string
        """
        return (
            f"EXECUTE EXACTLY ONE TOOL: `{tool_name}`\n\n"
            f"Tool Arguments (JSON):\n```json\n{json.dumps(args, indent=2)}\n```\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            f"1. Call ONLY the tool `{tool_name}` with the EXACT arguments shown above.\n"
            "2. After the tool executes, you will receive its output as a tool message.\n"
            "3. Your FINAL response must be ONLY the raw output from that tool.\n"
            "4. DO NOT call any additional tools after this one.\n"
            "5. DO NOT analyze, summarize, transform, or interpret the tool's output.\n"
            "6. DO NOT add explanations, commentary, or formatting.\n\n"
            "Return the tool's exact output and STOP."
        )
    
    def _format_args_for_log(self, args: Dict[str, Any]) -> str:
        """
        Format arguments for logging (truncate long values).
        
        Args:
            args: Tool arguments
        
        Returns:
            Formatted string
        """
        formatted = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 100:
                formatted[k] = f"{v[:50]}...{v[-50:]}"
            else:
                formatted[k] = v
        return json.dumps(formatted)
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Detect rate limit errors across different LLM providers.
        
        Checks for:
        - HTTP status codes (429, 503, 408)
        - Common rate limit keywords
        - Provider-specific error messages
        
        Args:
            error: The exception object
        
        Returns:
            True if error is likely a rate limit error
        """
        error_str = str(error).lower()
        error_repr = repr(error).lower()
        
        # Check if any rate limit pattern matches
        for pattern in RATE_LIMIT_PATTERNS:
            if pattern.lower() in error_str or pattern.lower() in error_repr:
                return True
        
        # Check exception type name
        error_type = type(error).__name__.lower()
        if "ratelimit" in error_type or "quota" in error_type or "throttle" in error_type:
            return True
        
        return False