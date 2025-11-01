"""
Comprehensive Logging Infrastructure

Production-grade logging for anomaly detection system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LoggerManager:
    """
    Centralized logging management.
    
    Singleton Pattern: Single logging configuration.
    """
    
    _instance: Optional['LoggerManager'] = None
    
    def __new__(cls):
        """Singleton implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize if not already done."""
        if self._initialized:
            return
        
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        self._initialized = True
    
    def get_logger(
        self,
        name: str,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True
    ) -> logging.Logger:
        """
        Get or create logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_to_file: Write to file
            log_to_console: Write to console
            
        Returns:
            Configured logger
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # Remove default handlers
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        self.loggers[name] = logger
        return logger


# Global logger instance
def get_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to get logger."""
    return LoggerManager().get_logger(name, **kwargs)


class PerformanceLogger:
    """
    Performance and timing logger.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with logger."""
        self.logger = logger or get_logger('performance')
        self.timings = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timings[operation] = datetime.now()
    
    def stop_timer(self, operation: str) -> float:
        """
        Stop timer and log duration.
        
        Args:
            operation: Operation name
            
        Returns:
            Duration in seconds
        """
        if operation not in self.timings:
            self.logger.warning(f"No timer found for {operation}")
            return 0.0
        
        duration = (datetime.now() - self.timings[operation]).total_seconds()
        self.logger.info(f"{operation} took {duration:.2f} seconds")
        
        del self.timings[operation]
        return duration
    
    def log_memory_usage(self, context: str = "") -> None:
        """Log current memory usage."""
        from .chunker import MemoryMonitor
        mem = MemoryMonitor.get_memory_usage()
        self.logger.info(f"Memory {context}: {mem}")


class AnomalyDetectionLogger:
    """
    Specialized logger for anomaly detection events.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with logger."""
        self.logger = logger or get_logger('anomaly_detection')
        self.stats = {
            'total_cycles': 0,
            'total_anomalies': 0,
            'by_type': {},
            'by_severity': {}
        }
    
    def log_detector_event(
        self,
        detector_name: str,
        cycle_id: Any,
        severity: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log detector event.
        
        Args:
            detector_name: Detector identifier
            cycle_id: Cycle identifier
            severity: Severity level
            details: Additional details
        """
        self.stats['total_anomalies'] += 1
        
        if detector_name not in self.stats['by_type']:
            self.stats['by_type'][detector_name] = 0
        self.stats['by_type'][detector_name] += 1
        
        if severity not in self.stats['by_severity']:
            self.stats['by_severity'][severity] = 0
        self.stats['by_severity'][severity] += 1
        
        self.logger.warning(
            f"Anomaly detected - Detector: {detector_name}, "
            f"Cycle: {cycle_id}, Severity: {severity}, Details: {details}"
        )
    
    def log_pipeline_summary(self) -> None:
        """Log pipeline execution summary."""
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total cycles: {self.stats['total_cycles']}")
        self.logger.info(f"Total anomalies: {self.stats['total_anomalies']}")
        self.logger.info(f"By type: {self.stats['by_type']}")
        self.logger.info(f"By severity: {self.stats['by_severity']}")
        self.logger.info("=" * 60)
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'total_cycles': 0,
            'total_anomalies': 0,
            'by_type': {},
            'by_severity': {}
        }


# Convenience functions
def setup_logging(log_dir: Path = Path('logs')) -> None:
    """Set up logging infrastructure."""
    LoggerManager._instance = None  # Reset singleton
    LoggerManager().log_dir = log_dir
    log_dir.mkdir(exist_ok=True)


def get_detection_logger() -> AnomalyDetectionLogger:
    """Get specialized detection logger."""
    return AnomalyDetectionLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger."""
    return PerformanceLogger()

