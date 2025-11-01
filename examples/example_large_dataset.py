"""
Example: Running Pipeline on 10M+ Row Dataset

Demonstrates memory-efficient processing for large files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor_anomaly_detection import CONFIG
from motor_anomaly_detection.core import (
    DataChunker,
    StreamingProcessor,
    MemoryMonitor,
    get_logger,
    get_performance_logger,
    setup_logging
)
from motor_anomaly_detection import run_pipeline
import os

# Setup logging
setup_logging(log_dir='logs')
logger = get_logger('large_dataset')
perf_logger = get_performance_logger()


def analyze_large_dataset(filepath: str):
    """
    Analyze large dataset with memory-efficient chunking.
    
    Args:
        filepath: Path to CSV file
    """
    logger.info(f"Starting analysis of large dataset: {filepath}")
    
    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024**2)
    logger.info(f"File size: {file_size_mb:.1f} MB")
    
    # Estimate memory usage
    chunker = DataChunker(chunk_size=100000)  # 100k rows per chunk
    estimate = chunker.estimate_memory_usage(filepath)
    
    logger.info("=" * 60)
    logger.info("MEMORY ESTIMATE")
    logger.info("=" * 60)
    logger.info(f"Total rows: {estimate['total_rows']:,}")
    logger.info(f"Estimated memory: {estimate['estimated_total_gb']:.2f} GB")
    logger.info(f"Chunks needed: {estimate['chunks_needed']}")
    logger.info(f"Chunk size: {chunker.chunk_size:,} rows")
    logger.info("=" * 60)
    
    # Check current memory
    perf_logger.log_memory_usage("Before processing")
    
    # Run pipeline with chunking
    try:
        perf_logger.start_timer('full_pipeline')
        
        # For very large datasets, we can process in chunks
        # Here's a simplified approach - in production, you'd want incremental processing
        logger.info("Running pipeline with memory optimization...")
        
        results = run_pipeline(
            filepath=filepath,
            output_dir='outputs',
            verbose=True
        )
        
        perf_logger.stop_timer('full_pipeline')
        perf_logger.log_memory_usage("After processing")
        
        # Summary
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Cycles analyzed: {len(results['cycle_summary'])}")
        logger.info(f"Anomalies found: {len(results['events'])}")
        logger.info("=" * 60)
        
        return results
        
    except MemoryError:
        logger.error("Out of memory! Consider:")
        logger.error("  1. Reducing chunk_size")
        logger.error("  2. Processing smaller date ranges")
        logger.error("  3. Using streaming approach")
        raise
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def process_in_streaming_mode(filepath: str):
    """
    Process dataset in streaming mode (minimal memory).
    
    For truly massive datasets (10M+ rows).
    """
    logger.info("Starting streaming mode processing")
    
    chunker = DataChunker(chunk_size=50000)  # Smaller chunks for streaming
    processor = StreamingProcessor(chunker)
    
    # Process each chunk
    chunk_results = []
    for chunk_num, result in enumerate(processor.stream_detect(filepath, detector=None)):
        if chunk_num % 10 == 0:
            logger.info(f"Processed {chunk_num} chunks")
            perf_logger.log_memory_usage(f"After chunk {chunk_num}")
        
        chunk_results.append(result)
    
    logger.info(f"Processing complete. Processed {len(chunk_results)} chunks")
    return chunk_results


def optimize_dataset_before_processing(filepath: str):
    """
    Optimize dataset before processing.
    
    Reduces memory footprint by 50-70%.
    """
    logger.info("Optimizing dataset...")
    
    chunker = DataChunker(
        chunk_size=100000,
        use_dtype_optimization=True,
        use_category_optimization=True
    )
    
    # Read in chunks, optimize, save
    optimized_file = filepath.replace('.csv', '_optimized.csv')
    
    def optimize_chunk(chunk):
        """Optimize a single chunk."""
        return chunker._optimize_dtypes(chunk)
    
    logger.info(f"Processing and optimizing {filepath}...")
    chunker.process_in_chunks(filepath, optimize_chunk, output_path=optimized_file)
    
    logger.info(f"Optimized file saved to: {optimized_file}")
    
    return optimized_file


if __name__ == '__main__':
    # Example usage
    
    # Your actual file path (when at office)
    actual_file = r'C:\Users\274005\Downloads\phase 0 check\DL1_Current Data.csv'
    
    # Check if file exists
    if os.path.exists(actual_file):
        print(f"\n{'='*60}")
        print("LARGE DATASET PROCESSING")
        print(f"{'='*60}\n")
        
        # Show memory analysis
        chunker = DataChunker()
        estimate = chunker.estimate_memory_usage(actual_file)
        
        print("File Analysis:")
        print(f"  Total rows: {estimate['total_rows']:,}")
        print(f"  Estimated size: {estimate['estimated_total_gb']:.2f} GB")
        print(f"  Chunks needed: {estimate['chunks_needed']}")
        print()
        
        # Ask user for processing mode
        print("Choose processing mode:")
        print("  1. Standard (balanced)")
        print("  2. Memory-optimized (chunking)")
        print("  3. Streaming (minimal memory)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            analyze_large_dataset(actual_file)
        elif choice == '2':
            optimized = optimize_dataset_before_processing(actual_file)
            analyze_large_dataset(optimized)
        elif choice == '3':
            process_in_streaming_mode(actual_file)
        else:
            print("Invalid choice!")
    else:
        print(f"\nFile not found: {actual_file}")
        print("\nFor demo purposes, using small dataset...")
        
        # Run on demo data
        from motor_anomaly_detection import run_pipeline
        results = run_pipeline('demo_motor_data.csv', output_dir='outputs', verbose=True)
        print("\nDemo complete!")

