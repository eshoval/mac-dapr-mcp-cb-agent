"""
File Manager - Simple file I/O operations
Handles: JSON save/load, text save, directory management
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class FileManager:
    """
    Simple file I/O manager.
    
    Responsibilities:
    - Save/load JSON files
    - Save text files
    - Create directories
    - Log file operations
    
    Does NOT handle:
    - Data validation (delegated to caller)
    - Data transformation (delegated to parser)
    """
    
    def __init__(self, base_dir: Path, logger=None):
        """
        Initialize file manager.
        
        Args:
            base_dir: Base directory for file operations
            logger: PerformanceLogger instance (optional)
        """
        self.base_dir = Path(base_dir)
        self.logger = logger
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Any, filename: str) -> Path:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save (must be JSON-serializable)
            filename: Output filename (relative to base_dir)
        
        Returns:
            Path to saved file
        
        Raises:
            Exception: If save fails
        """
        filepath = self.base_dir / filename
        
        if self.logger:
            self.logger.log("File Save (JSON)", "Started", str(filepath))
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if self.logger:
                size = filepath.stat().st_size
                self.logger.log("File Save (JSON)", "Ended", f"Size: {size} bytes")
            
            return filepath
        
        except Exception as e:
            if self.logger:
                self.logger.log("File Save (JSON)", "Error", str(e))
            raise
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load JSON from file.
        
        Args:
            filename: Input filename (relative to base_dir)
        
        Returns:
            Loaded data
        
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If load/parse fails
        """
        filepath = self.base_dir / filename
        
        if self.logger:
            self.logger.log("File Load (JSON)", "Started", str(filepath))
        
        if not filepath.exists():
            error_msg = f"File not found: {filepath}"
            if self.logger:
                self.logger.log("File Load (JSON)", "Error", error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.logger:
                self.logger.log("File Load (JSON)", "Ended", "Success")
            
            return data
        
        except Exception as e:
            if self.logger:
                self.logger.log("File Load (JSON)", "Error", str(e))
            raise
    
    def save_text(self, content: str, filename: str) -> Path:
        """
        Save raw text to file.
        
        Args:
            content: Text content to save
            filename: Output filename (relative to base_dir)
        
        Returns:
            Path to saved file
        
        Raises:
            Exception: If save fails
        """
        filepath = self.base_dir / filename
        
        if self.logger:
            self.logger.log("File Save (Text)", "Started", str(filepath))
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if self.logger:
                size = filepath.stat().st_size
                self.logger.log("File Save (Text)", "Ended", f"Size: {size} bytes")
            
            return filepath
        
        except Exception as e:
            if self.logger:
                self.logger.log("File Save (Text)", "Error", str(e))
            raise
    
    def ensure_directory(self, subdir: str) -> Path:
        """
        Ensure a subdirectory exists.
        
        Args:
            subdir: Subdirectory name (relative to base_dir)
        
        Returns:
            Path to directory
        """
        dirpath = self.base_dir / subdir
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath