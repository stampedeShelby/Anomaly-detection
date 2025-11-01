"""
Base Classes and Abstract Interfaces

Provides abstraction layer for detectors, models, and data processors.
Implements industry-standard design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    
    Strategy Pattern: Allows interchangeable detection algorithms.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize detector.
        
        Args:
            name: Detector identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in data.
        
        Args:
            df: Input DataFrame with time series data
            
        Returns:
            DataFrame with detection results (flags, scores, etc.)
        """
        pass
    
    @abstractmethod
    def get_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Get severity scores for detected anomalies.
        
        Args:
            df: DataFrame with detection results
            
        Returns:
            Series with severity scores (0.0 to 1.0)
        """
        pass
    
    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If data is invalid
        """
        required_cols = self.get_required_columns()
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required column names.
        
        Returns:
            List of column names
        """
        pass


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    
    Template Method Pattern: Defines model lifecycle.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize model."""
        self.name = name
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model on data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass
    
    def validate_fitted(self) -> None:
        """Ensure model is fitted before prediction."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} not fitted. Call fit() first.")


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors.
    
    Chain of Responsibility: Processors can be chained.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize processor."""
        self.name = name
        self.config = config or {}
        self.next_processor: Optional['BaseDataProcessor'] = None
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def set_next(self, processor: 'BaseDataProcessor') -> 'BaseDataProcessor':
        """
        Chain processors together.
        
        Args:
            processor: Next processor in chain
            
        Returns:
            processor (for fluent interface)
        """
        self.next_processor = processor
        return processor
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data and pass to next in chain."""
        result = self.process(df)
        if self.next_processor:
            return self.next_processor(result)
        return result


class BaseValidator:
    """
    Validation utilities following Factory pattern.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> None:
        """Validate DataFrame structure."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> None:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise TypeError(f"Expected dict, got {type(config)}")
        
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> None:
        """Validate numeric value in range."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


class Singleton(type):
    """
    Singleton metaclass for single-instance classes.
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigurableBase(ABC):
    """
    Mixin for configurable objects.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        self.config = config or {}
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration."""
        self.config.update(kwargs)


class Observable(ABC):
    """
    Observer pattern for event notifications.
    """
    
    def __init__(self):
        """Initialize with empty observers list."""
        self._observers: List['Observer'] = []
    
    def attach(self, observer: 'Observer') -> None:
        """Attach observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: 'Observer') -> None:
        """Detach observer."""
        self._observers.remove(observer)
    
    def notify(self, event: str, data: Any) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(event, data)


class Observer(ABC):
    """
    Observer interface for event handling.
    """
    
    @abstractmethod
    def update(self, event: str, data: Any) -> None:
        """Handle event notification."""
        pass


class LoggingObserver(Observer):
    """
    Observer that logs events.
    """
    
    def __init__(self, logger):
        """Initialize with logger."""
        self.logger = logger
    
    def update(self, event: str, data: Any) -> None:
        """Log event."""
        self.logger.info(f"Event: {event} | Data: {data}")

