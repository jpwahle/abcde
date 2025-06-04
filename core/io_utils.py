"""
Common I/O utilities for reading and writing data in various formats.
"""
import csv
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

logger = logging.getLogger("core.io_utils")


def write_results_to_csv(
    results: List[Dict[str, Any]], 
    output_csv: str, 
    output_tsv: bool, 
    data_source: str = "reddit", 
    split: str = None,
    flatten_fn=None
):
    """Helper function to write results to CSV/TSV with appropriate headers.
    
    Args:
        results: List of result dictionaries
        output_csv: Output file path
        output_tsv: Whether to output TSV instead of CSV
        data_source: Source of data ("reddit" or "tusc")
        split: Data split type (for TUSC)
        flatten_fn: Function to flatten results (defaults to core function)
    """
    if flatten_fn is None:
        from core.data_processing import flatten_result_to_csv_row, get_csv_fieldnames
        flatten_fn = flatten_result_to_csv_row
        fieldnames_fn = get_csv_fieldnames
    else:
        # For custom flatten functions, assume fieldnames are provided separately
        fieldnames_fn = None
    
    # Determine output format
    separator = '\t' if output_tsv else ','
    file_extension = 'tsv' if output_tsv else 'csv'
    output_file = output_csv.replace('.csv', f'.{file_extension}') if output_tsv else output_csv
    
    if results:
        # Flatten results for CSV/TSV
        csv_rows = [flatten_fn(result, data_source) for result in results]
        
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            if fieldnames_fn:
                fieldnames = fieldnames_fn(data_source, split)
            else:
                # Infer fieldnames from first row
                fieldnames = list(csv_rows[0].keys()) if csv_rows else []
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()
            
            for row in tqdm(csv_rows, desc=f"Writing {file_extension.upper()}"):
                writer.writerow(row)
    else:
        logger.warning("No results found. Creating empty CSV file.")
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            if fieldnames_fn:
                fieldnames = fieldnames_fn(data_source, split)
            else:
                fieldnames = []
            
            if fieldnames:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
                writer.writeheader()


def ensure_output_directory(output_path: str):
    """Ensure the output directory exists."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)


def detect_file_format(file_path: str) -> str:
    """Detect file format based on extension."""
    path = Path(file_path)
    if path.suffix.lower() in ['.tsv']:
        return 'tsv'
    elif path.suffix.lower() in ['.csv']:
        return 'csv'
    elif path.suffix.lower() in ['.parquet']:
        return 'parquet'
    elif path.suffix.lower() in ['.jsonl', '.json']:
        return 'jsonl'
    else:
        return 'unknown'


def auto_detect_delimiter(file_path: str) -> str:
    """Auto-detect delimiter for CSV/TSV files."""
    if file_path.lower().endswith('.tsv'):
        return '\t'
    elif file_path.lower().endswith('.csv'):
        return ','
    else:
        # Try to detect from first few lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' in first_line:
                    # More tabs or commas?
                    return '\t' if first_line.count('\t') > first_line.count(',') else ','
                elif '\t' in first_line:
                    return '\t'
                else:
                    return ','
        except Exception:
            return ','  # Default to comma