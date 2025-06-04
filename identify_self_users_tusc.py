"""
TUSC-specific self-identification detection pipeline.
Identifies users who self-identify demographic traits in TUSC tweets.
"""
import argparse
import logging
import pandas as pd

from self_identification import SelfIdentificationDetector
from core.cluster import setup_dask_cluster, cleanup_cluster
from core.io_utils import ensure_output_directory, write_results_to_csv
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
    parser.add_argument("--n_workers", type=int, default=32, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Chunk size for parallel processing")

    args = parser.parse_args()
    
    # Auto-determine split if not provided
    if args.split is None:
        args.split = determine_tusc_split(args.input_file)
    
    logger.info(f"Using split type: {args.split}")
    
    # Ensure output directory exists
    ensure_output_directory(args.output_csv)
    
    # Initialize detector
    detector = SelfIdentificationDetector()
    
    # Setup Dask cluster (only for non-test mode)
    client = None
    if not args.test_mode:
        client = setup_dask_cluster(
            n_workers=args.n_workers,
            memory_per_worker=args.memory_per_worker,
            use_slurm=args.use_slurm
        )
    
    try:
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
            include_features=False,
            chunk_size=args.chunk_size,
            client=client
        )
    
        logger.info(f"Detected {len(results_df)} self-identification posts. Writing to {args.output_csv}")
        
        # Convert DataFrame to list of dictionaries for write_results_to_csv
        results = results_df.to_dict('records') if not results_df.empty else []
        
        # Write results using shared utility
        write_results_to_csv(
            results=results,
            output_csv=args.output_csv,
            output_tsv=args.output_tsv,
            data_source="tusc",
            split=args.split
        )
        
    finally:
        if client is not None:
            cleanup_cluster(client, args.use_slurm)


if __name__ == "__main__":
    main()