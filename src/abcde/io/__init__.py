"""I/O utilities for reading and writing data files."""

from abcde.io.csv_utils import (
    flatten_result_to_csv_row,
    get_csv_fieldnames,
    ensure_output_directory,
    write_results_to_csv,
    append_results_to_csv,
)
from abcde.io.jsonl_utils import (
    get_all_jsonl_files,
    filter_entry,
    extract_columns,
)
from abcde.io.demographics import (
    format_demographic_detections_for_output,
    aggregate_user_demographics,
)

__all__ = [
    "flatten_result_to_csv_row",
    "get_csv_fieldnames",
    "ensure_output_directory",
    "write_results_to_csv",
    "append_results_to_csv",
    "get_all_jsonl_files",
    "filter_entry",
    "extract_columns",
    "format_demographic_detections_for_output",
    "aggregate_user_demographics",
]
