"""
Memory-Efficient Chunking for Large Datasets

Optimized for 10M+ row datasets with minimal memory footprint.
"""

from typing import Iterator, Optional, Callable, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .base import BaseValidator


class DataChunker:
    """
    Memory-efficient data chunking for large files.
    
    Optimized for 10M+ row CSV files.
    """
    
    def __init__(
        self,
        chunk_size: int = 100000,
        use_dtype_optimization: bool = True,
        use_category_optimization: bool = True
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Number of rows per chunk (100k default)
            use_dtype_optimization: Optimize data types to reduce memory
            use_category_optimization: Convert low-cardinality columns to category
        """
        self.chunk_size = chunk_size
        self.use_dtype_optimization = use_dtype_optimization
        self.use_category_optimization = use_category_optimization
        self.validator = BaseValidator()
    
    def chunk_csv(
        self,
        filepath: str,
        **read_csv_kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over CSV file in chunks.
        
        Args:
            filepath: Path to CSV file
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Yields:
            DataFrame chunks
        """
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, **read_csv_kwargs):
            if self.use_dtype_optimization:
                chunk = self._optimize_dtypes(chunk)
            yield chunk
    
    def process_in_chunks(
        self,
        filepath: str,
        processor: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None,
        **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        Process file in chunks and combine results.
        
        Args:
            filepath: Input CSV path
            processor: Function to process each chunk
            output_path: Optional output path
            **read_csv_kwargs: Additional read_csv args
            
        Returns:
            Combined results DataFrame
        """
        chunk_list = []
        
        for chunk_num, chunk in enumerate(self.chunk_csv(filepath, **read_csv_kwargs)):
            processed = processor(chunk)
            chunk_list.append(processed)
            
            if (chunk_num + 1) % 10 == 0:
                print(f"Processed {chunk_num + 1} chunks ({len(chunk_list[-1])} rows each)")
        
        result = pd.concat(chunk_list, ignore_index=True)
        
        if output_path:
            result.to_csv(output_path, index=False)
            print(f"Saved results to {output_path}")
        
        return result
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        df = df.copy()
        
        # Integer optimization
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Float optimization
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Category optimization
        if self.use_category_optimization:
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < 100:  # Low cardinality
                    df[col] = df[col].astype('category')
        
        return df
    
    def estimate_memory_usage(self, filepath: str, sample_rows: int = 10000) -> dict:
        """
        Estimate memory usage for file.
        
        Args:
            filepath: Path to CSV
            sample_rows: Number of rows to sample
            
        Returns:
            Dictionary with memory estimates
        """
        sample = pd.read_csv(filepath, nrows=sample_rows)
        
        total_rows = sum(1 for _ in open(filepath)) - 1
        
        sample_memory = sample.memory_usage(deep=True).sum()
        estimated_total = sample_memory * (total_rows / sample_rows)
        
        return {
            'sample_memory_mb': sample_memory / 1024**2,
            'estimated_total_mb': estimated_total / 1024**2,
            'estimated_total_gb': estimated_total / 1024**3,
            'total_rows': total_rows,
            'chunks_needed': int(np.ceil(total_rows / self.chunk_size))
        }


class StreamingProcessor:
    """
    Stream-based processing for incremental operations.
    """
    
    def __init__(self, chunker: DataChunker):
        """Initialize with chunker."""
        self.chunker = chunker
        self.state = {}
    
    def stream_detect(
        self,
        filepath: str,
        detector,
        accumulate: bool = True
    ) -> Iterator[dict]:
        """
        Stream anomaly detection over chunks.
        
        Args:
            filepath: Input CSV
            detector: Detector instance
            accumulate: Accumulate state across chunks
            
        Yields:
            Detection results for each chunk
        """
        for chunk in self.chunker.chunk_csv(filepath):
            try:
                detected = detector.detect(chunk)
                results = {
                    'chunk_size': len(chunk),
                    'anomalies': detected.get('anomaly_flag', pd.Series([False]*len(detected))).sum(),
                    'data': detected
                }
                
                if accumulate:
                    self._update_state(results)
                
                yield results
            except Exception as e:
                print(f"Error processing chunk: {e}")
                yield {'error': str(e)}
    
    def _update_state(self, results: dict) -> None:
        """Update accumulated state."""
        if 'total_anomalies' not in self.state:
            self.state['total_anomalies'] = 0
        
        self.state['total_anomalies'] += results.get('anomalies', 0)


class MemoryMonitor:
    """
    Memory usage monitoring and optimization.
    """
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            
            return {
                'rss_mb': mem_info.rss / 1024**2,  # Resident Set Size
                'vms_mb': mem_info.vms / 1024**2,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'note': 'Install psutil for memory monitoring'}
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressively optimize DataFrame memory.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize dtypes
        df = DataChunker()._optimize_dtypes(df)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        reduction = (1 - final_memory / initial_memory) * 100
        
        print(f"Memory reduced from {initial_memory:.2f}MB to {final_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return df

