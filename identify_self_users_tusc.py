"""
TUSC-specific self-identification detection pipeline.
Identifies users who self-identify demographic traits in TUSC tweets.
"""
import argparse
import logging
import pandas as pd

from self_identification import SelfIdentificationDetector
from core.io_utils import ensure_output_directory
from tusc.data_loader import load_tusc_file, determine_tusc_split

logger = logging.getLogger("identify_self_users_tusc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def main():
    parser = argparse.ArgumentParser(description="Detect self-identified users in TUSC parquet file")
    parser.add_argument("--input_file", required=True, help="Path to input parquet file")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for self-identification matches")
    parser.add_argument("--split", choices=["city", "country"], help="Data split type (auto-determined if not specified)")
    parser.add_argument("--min_words", type=int, default=5, help="Minimum word count filter")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum word count filter")
    parser.add_argument("--test_mode", action="store_true", help="Test mode with limited samples")
    parser.add_argument("--test_samples", type=int, default=10000, help="Number of samples in test mode")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    args = parser.parse_args()
    
    # Auto-determine split if not provided
    if args.split is None:
        args.split = determine_tusc_split(args.input_file)
    
    logger.info(f"Using split type: {args.split}")
    
    # Ensure output directory exists
    ensure_output_directory(args.output_csv)
    
    # Initialize detector
    detector = SelfIdentificationDetector()
    
    # Process TUSC file
    results_df = load_tusc_file(
        input_file=args.input_file,
        detector=detector,
        split=args.split,
        min_words=args.min_words,
        max_words=args.max_words,
        mode="self_identification",
        test_mode=args.test_mode,
        test_samples=args.test_samples,
        include_features=False
    )
    
    logger.info(f"Detected {len(results_df)} self-identification posts. Writing to {args.output_csv}")
    
    # Write results
    separator = '\t' if args.output_tsv else ','
    file_extension = 'tsv' if args.output_tsv else 'csv'
    output_file = args.output_csv.replace('.csv', f'.{file_extension}') if args.output_tsv else args.output_csv
    
    if len(results_df) > 0:
        # Convert DataFrame results to the format expected by CSV writer
        from core.data_processing import flatten_result_to_csv_row, get_csv_fieldnames
        
        csv_rows = [flatten_result_to_csv_row(row, "tusc") for _, row in results_df.iterrows()]
        fieldnames = get_csv_fieldnames("tusc", args.split)
        
        import csv
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
    else:
        logger.warning("No results found. Creating empty CSV file.")
        from core.data_processing import get_csv_fieldnames
        fieldnames = get_csv_fieldnames("tusc", args.split)
        
        import csv
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
            writer.writeheader()


if __name__ == "__main__":
    main()