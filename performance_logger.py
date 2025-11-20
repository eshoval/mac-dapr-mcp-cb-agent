"""
Performance Logger - Simple in-memory logger with file export
(נשאר כמו המקור - לא נוגעים!)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class PerformanceLogger:
    """Simple in-memory logger for tracking performance"""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.log_counter = 0
        self.event_timers: Dict[str, datetime] = {}
    
    def log(self, event_name: str, status: str, details: str = "") -> Dict[str, Any]:
        """
        Log an event with timestamp and optional duration tracking.
        
        Args:
            event_name: Name of the event
            status: Status (Started, Ended, Info, Warning, Error)
            details: Additional details
        
        Returns:
            The created log entry
        """
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
        
        # Handle duration tracking
        if status == "Ended" and event_name in self.event_timers:
            start_time = self.event_timers.pop(event_name)
            duration = (timestamp - start_time).total_seconds()
            duration_seconds = round(duration, 3)
            log_entry["duration_seconds"] = duration_seconds
        elif status == "Started":
            if event_name not in self.event_timers:
                self.event_timers[event_name] = timestamp
        
        self.logs.append(log_entry)
        
        # Print to console
        duration_str = f" ({duration_seconds}s)" if duration_seconds is not None else ""
        timestamp_str = timestamp.strftime('%H:%M:%S.%f')[:-3]
        details_str = f" | {details}" if details else ""
        
        print(f"[{log_entry['log_id']}] {timestamp_str} | {event_name} | {status}{duration_str}{details_str}")
        
        return log_entry
    
    def save_to_file(self, filename: str = None, logs_dir: Path = None) -> Path:
        """
        Save logs to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            logs_dir: Directory to save logs (default: ./logs)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discovery_complete_log_{timestamp}.json"
        
        if logs_dir is None:
            logs_dir = Path("./logs")
        
        logs_dir.mkdir(exist_ok=True, parents=True)
        filepath = logs_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Log saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Error saving log file: {e}")
        
        return filepath