import json
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

class TranslationLogger:
    def __init__(self, log_dir: str = "translation_logs"):
        """
        Initialize the translation logger.
        
        Args:
            log_dir (str): Directory where log files will be stored
        """
        self.log_dir = log_dir
        self._ensure_log_directory()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _ensure_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)

    def _generate_log_filename(self) -> str:
        """Generate log filename based on current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"translation_log_{current_date}.json")

    def _read_existing_logs(self, filename: str) -> list:
        """Read existing logs from file if it exists."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"Error reading log file {filename}")
            return []
        return []

    def _write_logs(self, filename: str, logs: list) -> None:
        """Write logs to file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing to log file: {str(e)}")

    def log_translation(self, 
                       source_text: str,
                       translated_text: str,
                       source_lang: str,
                       target_lang: str,
                       is_family: bool,
                       family_name: Optional[str] = None,
                       extracted_text: Optional[str] = None,
                       file_name: Optional[str] = None,
                       translation_type: str = "text",
                       success: bool = True,
                       error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a translation activity.
        
        Args:
            source_text (str): Original text to translate
            translated_text (str): Translated text
            source_lang (str): Source language
            target_lang (str): Target language
            is_family (bool): Whether translation used language family
            family_name (Optional[str]): Name of language family if applicable
            extracted_text (Optional[str]): Text extracted from document if applicable
            file_name (Optional[str]): Name of uploaded file if applicable
            translation_type (str): Type of translation ('text' or 'document' or 'html')
            success (bool): Whether translation was successful
            error_message (Optional[str]): Error message if translation failed
            
        Returns:
            Dict[str, Any]: The logged entry
        """
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": "document" if file_name else "text",
            "source_language": source_lang,
            "target_language": target_lang,
            "is_family_translation": is_family,
            "language_family": family_name if is_family else None,
            "source_text": source_text,
            "translated_text": translated_text,
            "translation_type": translation_type,
            "success": success,
            "error_message": error_message
        }

        # Add document-specific fields if applicable
        if file_name:
            log_entry.update({
                "file_name": file_name,
                "extracted_text": extracted_text
            })

        # Read existing logs, append new entry, and write back
        filename = self._generate_log_filename()
        logs = self._read_existing_logs(filename)
        logs.append(log_entry)
        self._write_logs(filename, logs)

        self.logger.info(f"Logged translation activity with ID: {log_entry['id']}")
        return log_entry

    def get_logs(self, date_str: Optional[str] = None) -> list:
        """
        Retrieve logs for a specific date or all logs if no date provided.
        
        Args:
            date_str (Optional[str]): Date in YYYY-MM-DD format
            
        Returns:
            list: List of log entries
        """
        if date_str:
            filename = os.path.join(self.log_dir, f"translation_log_{date_str}.json")
            return self._read_existing_logs(filename)
        
        # If no date provided, collect all logs
        all_logs = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("translation_log_") and filename.endswith(".json"):
                logs = self._read_existing_logs(os.path.join(self.log_dir, filename))
                all_logs.extend(logs)
        
        return sorted(all_logs, key=lambda x: x['timestamp'], reverse=True)

    def search_logs(self, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   translation_type: Optional[str] = None,
                   source_lang: Optional[str] = None,
                   target_lang: Optional[str] = None,
                   success_only: bool = False) -> list:
        """
        Search logs with various filters.
        
        Args:
            start_date (Optional[str]): Start date in YYYY-MM-DD format
            end_date (Optional[str]): End date in YYYY-MM-DD format
            translation_type (Optional[str]): 'text' or 'document'
            source_lang (Optional[str]): Source language
            target_lang (Optional[str]): Target language
            success_only (bool): Whether to return only successful translations
            
        Returns:
            list: Filtered log entries
        """
        all_logs = self.get_logs()
        filtered_logs = []

        for log in all_logs:
            # Apply date filters
            log_date = datetime.fromisoformat(log['timestamp']).date()
            if start_date and log_date < datetime.strptime(start_date, "%Y-%m-%d").date():
                continue
            if end_date and log_date > datetime.strptime(end_date, "%Y-%m-%d").date():
                continue

            # Apply other filters
            if translation_type and log['type'] != translation_type:
                continue
            if source_lang and log['source_language'] != source_lang:
                continue
            if target_lang and log['target_language'] != target_lang:
                continue
            if success_only and not log['success']:
                continue

            filtered_logs.append(log)

        return filtered_logs