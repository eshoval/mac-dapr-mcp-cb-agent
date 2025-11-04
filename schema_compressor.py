import json
from pathlib import Path
from typing import Any, Dict, List

class SchemaCompressor:
    """
    Compresses Couchbase schema JSON (from discovery.py output) to only essential
    information for LLM query generation. Removes redundant samples, statistics, 
    and overly detailed nested structures.
    
    Version 2.0 Enhancements:
    - Multi-schema support (handles collections with multiple document types)
    - Smart sample truncation (limits size of long strings/objects)
    - WARNING vs ERROR distinction (preserves partial schemas)
    - Enhanced validation and statistics
    """
    
    def __init__(self, max_samples: int = 2, max_sample_length: int = 100, include_statistics: bool = False):
        """
        Args:
            max_samples: Maximum number of sample values to keep per field (default: 2)
            max_sample_length: Maximum character length for string samples (default: 100)
            include_statistics: Whether to keep #docs and %docs statistics (default: False)
        """
        self.max_samples = max_samples
        self.max_sample_length = max_sample_length
        self.include_statistics = include_statistics
    
    def compress_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main compression function. Returns a lightweight version of the schema.
        """
        if "scopes" not in schema:
            raise ValueError("Invalid schema: missing 'scopes' key")
        
        compressed = {
            "bucket_name": schema.get("bucket_name", "unknown"),
            "generated_at": schema.get("generated_at", ""),
            "scopes": {}
        }
        
        for scope_name, scope_data in schema.get("scopes", {}).items():
            compressed["scopes"][scope_name] = {
                "collections": self._compress_collections(scope_data.get("collections", {}))
            }
        
        return compressed
    
    def _compress_collections(self, collections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress collection-level data with multi-schema support.
        Handles collections that contain multiple document types (e.g., _default._default).
        """
        compressed_collections = {}
        
        for coll_name, coll_data in collections.items():
            doc_count = coll_data.get("document_count", 0)
            error_msg = coll_data.get("error")
            content_list = coll_data.get("content", [])

            # Determine status with WARNING vs ERROR distinction
            status = "error"
            if error_msg is None:
                status = "populated" if doc_count > 0 else "empty"
            elif "WARNING" in str(error_msg):
                status = "populated_partial"  # Has content but may be incomplete
            elif "Skipped" in str(error_msg) or "0 documents" in str(error_msg):
                status = "empty"

            compressed_collections[coll_name] = {
                "document_count": doc_count,
                "status": status
            }
            
            # Include error/warning for transparency
            if error_msg:
                compressed_collections[coll_name]["note"] = error_msg

            # Process schemas (may be multiple in one collection)
            if status in ["populated", "populated_partial"] and isinstance(content_list, list):
                schemas = []
                for schema_obj in content_list:
                    if isinstance(schema_obj, dict) and "properties" in schema_obj:
                        compressed_schema = self._compress_properties(schema_obj)
                        # Add flavor/identifier if exists (useful for multi-type collections)
                        if "Flavor" in schema_obj:
                            compressed_schema["flavor"] = schema_obj["Flavor"]
                        if "#docs" in schema_obj:
                            compressed_schema["doc_count"] = schema_obj["#docs"]
                        schemas.append(compressed_schema)
                
                if schemas:
                    # If single schema, unwrap from array for simplicity
                    if len(schemas) == 1:
                        compressed_collections[coll_name]["schema"] = schemas[0]
                    else:
                        # Multiple schemas (e.g., route + landmark + hotel in _default)
                        compressed_collections[coll_name]["schemas"] = schemas
                        compressed_collections[coll_name]["note"] = f"Collection contains {len(schemas)} document types"
                else:
                    compressed_collections[coll_name]["note"] = "No valid schemas found in content"
            
            elif status == "empty":
                compressed_collections[coll_name]["note"] = "Collection is empty"
        
        return compressed_collections
    
    def _compress_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compress the properties section of a schema."""
        compressed = {
            "type": schema.get("type", "object"),
            "fields": {}
        }
        
        properties = schema.get("properties", {})
        
        for field_name, field_info in properties.items():
            compressed["fields"][field_name] = self._compress_field(field_info)
        
        return compressed
    
    def _compress_field(self, field_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compress individual field information with smart sample handling."""
        compressed_field = {}
        
        # Always keep type information
        if "type" in field_info:
            field_type = field_info["type"]
            if isinstance(field_type, list):
                # For nullable fields like ["null", "string"], simplify to "string (nullable)"
                non_null_types = [t for t in field_type if t != "null"]
                if len(non_null_types) == 1:
                    compressed_field["type"] = f"{non_null_types[0]} (nullable)"
                else:
                    # Keep complex multi-types (e.g., ["string", "number"])
                    compressed_field["type"] = field_type
            else:
                compressed_field["type"] = field_type
        
        # Keep limited samples with smart truncation
        if "samples" in field_info:
            samples = field_info["samples"]
            if isinstance(samples, list) and samples:
                # Handle nullable fields (nested arrays)
                if isinstance(samples[0], list):
                    for sample_list in samples:
                        if sample_list and sample_list[0] is not None:
                            compressed_field["sample_values"] = self._truncate_samples(
                                sample_list[:self.max_samples]
                            )
                            break
                else:
                    compressed_field["sample_values"] = self._truncate_samples(
                        samples[:self.max_samples]
                    )
        
        # Handle nested objects (like geo coordinates)
        if field_info.get("type") == "object" and "properties" in field_info:
            compressed_field["nested_fields"] = {}
            for nested_name, nested_info in field_info["properties"].items():
                nested_samples = nested_info.get("samples", [])[:self.max_samples]
                compressed_field["nested_fields"][nested_name] = {
                    "type": nested_info.get("type"),
                    "sample_values": self._truncate_samples(nested_samples)
                }
        
        # Handle arrays with item schemas
        if field_info.get("type") == "array" and "items" in field_info:
            items_info = field_info["items"]
            if isinstance(items_info, dict):
                compressed_field["array_item_type"] = items_info.get("type", "unknown")
                
                # If array contains objects, show structure
                if items_info.get("type") == "object" and "properties" in items_info:
                    compressed_field["array_item_structure"] = {
                        name: info.get("type")
                        for name, info in items_info["properties"].items()
                    }
                
                # Add min/max items if available (useful for understanding data structure)
                if "minItems" in field_info:
                    compressed_field["min_items"] = field_info["minItems"]
                if "maxItems" in field_info:
                    compressed_field["max_items"] = field_info["maxItems"]
        
        # Optionally include statistics (disabled by default to save space)
        if self.include_statistics and "#docs" in field_info:
            compressed_field["doc_count"] = field_info["#docs"]
        
        return compressed_field
    
    def _truncate_samples(self, samples: List[Any]) -> List[Any]:
        """
        Truncate sample values that are too long (e.g., long strings, large objects).
        This prevents massive reviews/descriptions from bloating the compressed schema.
        """
        if not samples:
            return []
        
        truncated = []
        for sample in samples:
            if isinstance(sample, str):
                if len(sample) > self.max_sample_length:
                    # Keep first part + ellipsis
                    truncated.append(sample[:self.max_sample_length] + "...")
                else:
                    truncated.append(sample)
            elif isinstance(sample, dict):
                # For complex objects (e.g., review objects), simplify to just structure
                # Keep only top-level keys to show what fields exist
                truncated.append({"_structure": list(sample.keys())[:5]})
            elif isinstance(sample, list):
                if len(sample) > 3:
                    # For large arrays, keep only first few items + indicator
                    truncated.append(sample[:3] + ["..."])
                else:
                    truncated.append(sample)
            else:
                # Numbers, booleans, etc. - keep as-is
                truncated.append(sample)
        
        return truncated
    
    def compress_and_save(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Load a schema from file, compress it, validate, and save the result.
        Returns detailed compression statistics.
        """
        # Load original schema
        print(f"📂 Loading schema from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            original_schema = json.load(f)
        
        # Validate input structure
        if "scopes" not in original_schema:
            raise ValueError("Invalid schema: missing 'scopes' key. Is this a discovery.py output?")
        
        # Compress
        print("🗜️  Compressing schema...")
        compressed_schema = self.compress_schema(original_schema)
        
        # Save compressed version
        print(f"💾 Saving compressed schema to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(compressed_schema, f, indent=2)
        
        # Calculate detailed statistics
        original_size = input_path.stat().st_size
        compressed_size = output_path.stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        # Count collections and schemas
        total_collections = 0
        total_schemas = 0
        multi_schema_collections = 0
        
        for scope in compressed_schema["scopes"].values():
            for coll_name, coll_data in scope["collections"].items():
                total_collections += 1
                if "schema" in coll_data:
                    total_schemas += 1
                elif "schemas" in coll_data:
                    schema_count = len(coll_data["schemas"])
                    total_schemas += schema_count
                    multi_schema_collections += 1
        
        stats = {
            "original_size_kb": round(original_size / 1024, 2),
            "compressed_size_kb": round(compressed_size / 1024, 2),
            "compression_ratio_percent": round(compression_ratio, 2),
            "total_collections": total_collections,
            "total_schemas_extracted": total_schemas,
            "multi_schema_collections": multi_schema_collections,
            "original_path": str(input_path),
            "compressed_path": str(output_path)
        }
        
        return stats


def main():
    """
    Example usage: compress the travel-sample schema.
    Run this after discovery.py has generated the full schema.
    """
    compressor = SchemaCompressor(
        max_samples=2,              # Keep only 2 sample values per field
        max_sample_length=100,      # Truncate strings longer than 100 chars
        include_statistics=False    # Don't include #docs/%docs (save space)
    )
    
    # Input is the file generated by discovery.py
    input_file = Path("./schema_context/travel-sample_schema_final_complete.json")
    
    # Output is the new compressed file
    output_file = Path("./schema_context/travel-sample_schema_final_compressed.json")
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found at {input_file}")
        print("Please run discovery.py first to generate the schema file.")
        return
    
    print(f"\n{'='*60}")
    print("Schema Compression Tool v2.0")
    print(f"{'='*60}\n")
    
    try:
        stats = compressor.compress_and_save(input_file, output_file)
        
        print(f"\n{'='*60}")
        print("✅ Compression Complete!")
        print(f"{'='*60}")
        print(f"📊 Original size:      {stats['original_size_kb']} KB")
        print(f"📊 Compressed size:    {stats['compressed_size_kb']} KB")
        print(f"📊 Size reduction:     {stats['compression_ratio_percent']}%")
        print(f"📊 Collections:        {stats['total_collections']}")
        print(f"📊 Schemas extracted:  {stats['total_schemas_extracted']}")
        if stats['multi_schema_collections'] > 0:
            print(f"📊 Multi-type colls:   {stats['multi_schema_collections']}")
        print(f"\n💾 Output saved to: {stats['compressed_path']}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error during compression: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()