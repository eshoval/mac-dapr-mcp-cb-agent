"""
Response Parser - Multi-layer JSON parsing with fallback logic
Handles: JSON extraction, truncation repair, type validation
"""

import json
import re
import ast
from typing import Tuple, Optional, Any, List

# JSON fence pattern
JSON_FENCE_RE = re.compile(
    r'```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```',
    re.DOTALL | re.IGNORECASE
)


class ResponseParser:
    """
    Multi-layer parser with structured logging.
    
    Parsing Strategy:
    - Layer 1: JSON extraction (fenced or bare)
    - Layer 2: ast.literal_eval (for stringified JSON)
    - Layer 3+: Unwrapping, type conversion, validation
    
    Each layer failure = WARNING (will try next layer)
    All layers fail = ERROR
    """
    
    def __init__(self, logger=None):
        """
        Initialize parser.
        
        Args:
            logger: PerformanceLogger instance (optional)
        """
        self.logger = logger
    
    def parse(
        self,
        raw_response: object,
        expected_type: Optional[str] = None,
        phase_name: str = "Generic"
    ) -> Tuple[Any, Optional[str]]:
        """
        Parse agent response with multi-layer fallback.
        
        Args:
            raw_response: Raw response object from agent
            expected_type: Expected type ("dict", "list", or None)
            phase_name: Phase name for logging (e.g., "P1", "P2")
        
        Returns:
            Tuple of (parsed_data, error_message)
            - parsed_data: The extracted and validated data
            - error_message: None (success), WARNING (soft failure), or ERROR (hard failure)
        """
        if self.logger:
            self.logger.log(f"Parse ({phase_name})", "Started", "Multi-layer parsing")
        
        # Step 1: Normalize to string
        raw_str = self._normalize(raw_response)
        if not raw_str:
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Error", "Empty response")
            return None, "ERROR: Empty response"
        
        if self.logger:
            self.logger.log(f"Parse ({phase_name})", "Info", f"Content length: {len(raw_str)}")
        
        # Step 2: Check for truncation
        truncation_warning = None
        if self._is_truncated(raw_str):
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Warning", "Truncation detected, attempting repair...")
            
            repaired = self._repair_truncated(raw_str)
            if repaired:
                truncation_warning = f"WARNING: Response was truncated. Repaired from {len(raw_str)} to {len(repaired)} chars."
                raw_str = repaired
                if self.logger:
                    self.logger.log(f"Parse ({phase_name})", "Info", "Truncation repair successful")
            else:
                if self.logger:
                    self.logger.log(f"Parse ({phase_name})", "Error", "Truncation repair failed")
                return None, "ERROR: Truncated and unrepairable"
        
        # Step 3: Try parsing layers
        parsed_data = None
        warnings: List[str] = []
        
        # Layer 1: JSON extraction
        parsed_data = self._try_json_extract(raw_str)
        if parsed_data is not None:
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Info", "Layer 1 (JSON extract) succeeded")
        else:
            warnings.append("Layer 1 (JSON extract) failed")
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Warning", warnings[-1])
            
            # Layer 2: ast.literal_eval
            parsed_data = self._try_ast_eval(raw_str)
            if parsed_data is not None:
                if self.logger:
                    self.logger.log(f"Parse ({phase_name})", "Info", "Layer 2 (ast.literal_eval) succeeded")
            else:
                warnings.append("Layer 2 (ast.literal_eval) failed")
                if self.logger:
                    self.logger.log(f"Parse ({phase_name})", "Warning", warnings[-1])
        
        # Step 4: If still None → ERROR
        if parsed_data is None:
            error = f"ERROR: All parsing layers failed. {'; '.join(warnings)}"
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Error", error)
            return None, error
        
        # Step 5: Unwrap tool response (Gemini-style wrapper)
        original_type = type(parsed_data).__name__
        parsed_data = self._unwrap_tool_response(parsed_data)
        if type(parsed_data).__name__ != original_type:
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Info", f"Unwrapped tool response: {original_type} → {type(parsed_data).__name__}")
        
        # Step 6: Handle list of JSON strings
        if isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], str):
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Info", "Detected list of JSON strings, parsing each...")
            parsed_data = self._parse_json_string_list(parsed_data)
            if self.logger:
                self.logger.log(f"Parse ({phase_name})", "Info", f"Parsed {len(parsed_data)} JSON strings")
        
        # Step 7: Validate type
        if expected_type:
            type_error = self._validate_type(parsed_data, expected_type)
            if type_error:
                if self.logger:
                    self.logger.log(f"Parse ({phase_name})", "Error", type_error)
                return None, f"ERROR: {type_error}"
        
        # Success
        if self.logger:
            self.logger.log(f"Parse ({phase_name})", "Ended", f"Success. Type: {type(parsed_data).__name__}")
        
        # Combine warnings
        final_message = None
        if truncation_warning:
            warnings.insert(0, truncation_warning)
        if warnings:
            final_message = "; ".join(warnings)
        
        return parsed_data, final_message
    
    # ========== Helper Methods ==========
    
    def _normalize(self, resp: object) -> str:
        """
        Normalize any response object to a clean string.
        
        Args:
            resp: Response object (can be BaseMessage, string, etc.)
        
        Returns:
            Clean string content
        """
        try:
            raw = resp.content if hasattr(resp, 'content') else resp
        except Exception:
            raw = resp
        
        result = str(raw).strip() if raw is not None else ""
        # Remove BOM if present
        return result.lstrip("\ufeff")
    
    def _try_json_extract(self, text: str) -> Optional[Any]:
        """
        Try to extract and parse JSON from text.
        Handles both fenced (```json) and bare JSON.
        
        Args:
            text: Input text
        
        Returns:
            Parsed JSON object or None
        """
        json_str = None
        
        # Try fenced JSON first
        match = JSON_FENCE_RE.search(text)
        if match:
            json_str = next((g for g in match.groups() if g), None)
        else:
            # Try bare JSON
            text = text.strip()
            if (text.startswith('{') and text.endswith('}')) or \
               (text.startswith('[') and text.endswith(']')):
                json_str = text
            else:
                # Find first { or [
                first_obj = text.find("{")
                first_arr = text.find("[")
                idxs = [i for i in (first_obj, first_arr) if i != -1]
                if idxs:
                    json_str = text[min(idxs):].strip()
        
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _try_ast_eval(self, text: str) -> Optional[Any]:
        """
        Try ast.literal_eval for stringified structures.
        
        Args:
            text: Input text
        
        Returns:
            Evaluated Python literal or None
        """
        try:
            return ast.literal_eval(text)
        except Exception:
            return None
    
    def _unwrap_tool_response(self, data: Any) -> Any:
        """
        Unwrap Gemini-style tool wrapper:
        {"ToolName_response": {"results": ["..."]}}
        
        Args:
            data: Parsed data
        
        Returns:
            Unwrapped data
        """
        if isinstance(data, dict) and len(data) == 1:
            first_key = next(iter(data))
            if first_key.endswith("_response") and "results" in data[first_key]:
                inner = data[first_key]["results"]
                if isinstance(inner, list) and inner:
                    first_item = inner[0]
                    if isinstance(first_item, str):
                        # Try parse as JSON
                        try:
                            return json.loads(first_item)
                        except json.JSONDecodeError:
                            try:
                                return ast.literal_eval(first_item)
                            except Exception:
                                return first_item
        return data
    
    def _parse_json_string_list(self, data: List[str]) -> List[Any]:
        """
        Parse list of JSON strings: ['"{...}"', '"{...}"']
        
        Args:
            data: List of JSON strings
        
        Returns:
            List of parsed objects
        """
        try:
            return [json.loads(item) for item in data]
        except Exception:
            return data  # Return as-is if parsing fails
    
    def _validate_type(self, data: Any, expected: str) -> Optional[str]:
        """
        Validate data type matches expected type.
        
        Args:
            data: Data to validate
            expected: Expected type ("dict" or "list")
        
        Returns:
            Error message or None
        """
        if expected == "list" and not isinstance(data, list):
            return f"Expected list, got {type(data).__name__}"
        if expected == "dict" and not isinstance(data, dict):
            return f"Expected dict, got {type(data).__name__}"
        return None
    
    def _is_truncated(self, text: str) -> bool:
        """
        Detect if content was truncated by checking for incomplete JSON.
        
        Args:
            text: Input text
        
        Returns:
            True if truncated
        """
        pattern = r"\.\.\.'\s*role='assistant'$"
        if re.search(pattern, text):
            clean = re.sub(pattern, "", text)
            open_curly = clean.count('{')
            close_curly = clean.count('}')
            open_square = clean.count('[')
            close_square = clean.count(']')
            return (open_curly > close_curly) or (open_square > close_square)
        return False
    
    def _repair_truncated(self, text: str) -> Optional[str]:
        """
        Smart repair of truncated JSON using state tracking.
        
        Args:
            text: Truncated text
        
        Returns:
            Repaired JSON string or None
        """
        clean = re.sub(r"\.\.\.'\s*role='assistant'$", "", text).strip()
        
        # State tracking for smart repair
        stack = []
        in_string = False
        escape_next = False
        
        for char in clean:
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}' and stack and stack[-1] == '}':
                    stack.pop()
                elif char == ']' and stack and stack[-1] == ']':
                    stack.pop()
        
        repaired = clean
        
        # Close unclosed string
        if in_string:
            repaired += '"'
        
        # Close unclosed structures
        repaired += ''.join(reversed(stack))
        
        # Validate repair
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            return None