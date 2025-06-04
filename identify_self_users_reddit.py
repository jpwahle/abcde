"""
Reddit-specific self-identification detection pipeline.
Identifies users who self-identify demographic traits in Reddit posts.
"""
import argparse
import logging

from self_identification import SelfIdentificationDetector
from core.cluster import setup_dask_cluster, cleanup_cluster
from core.io_utils import write_results_to_csv, ensure_output_directory
from reddit.data_loader import load_reddit_files_for_self_identification

logger = logging.getLogger("identify_self_users_reddit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def main():
    parser = argparse.ArgumentParser(description="Detect self-identified users in Reddit JSONL files")
    parser.add_argument("--input_dir", required=True, help="Directory containing RS_*.jsonl files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for self-identification matches")
    parser.add_argument("--split", choices=["text", "multimodal"], default="text", help="Split type")
    parser.add_argument("--min_words", type=int, default=5, help="Minimum word count filter")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum word count filter")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of Dask workers")
    parser.add_argument("--memory_per_worker", type=str, default="4GB", help="Memory per worker")
    parser.add_argument("--use_slurm", action="store_true", help="Use SLURM cluster for Dask workers")
    parser.add_argument("--output_tsv", action="store_true", help="Output TSV instead of CSV")

    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_output_directory(args.output_csv)
    
    # Initialize detector
    detector = SelfIdentificationDetector()
    
    # Setup Dask cluster
    client = setup_dask_cluster(
        n_workers=args.n_workers,
        memory_per_worker=args.memory_per_worker,
        use_slurm=args.use_slurm
    )
    
    try:
        # Process Reddit files
        results = load_reddit_files_for_self_identification(
            input_dir=args.input_dir,
            detector=detector,
            split=args.split,
            min_words=args.min_words,
            max_words=args.max_words,
            client=client
        )
        
        logger.info(f"Detected {len(results)} self-identification posts. Writing to {args.output_csv}")
        
        # Write results
        write_results_to_csv(
            results=results,
            output_csv=args.output_csv,
            output_tsv=args.output_tsv,
            data_source="reddit"
        )
        
    finally:
        cleanup_cluster(client, args.use_slurm)


if __name__ == "__main__":
    main()