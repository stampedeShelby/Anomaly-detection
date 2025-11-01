"""
Factory Pattern for Object Creation

Centralizes creation of models, detectors, and processors.
"""

from typing import Dict, Any, Type, Optional
from .base import BaseModel, BaseDetector, BaseDataProcessor


class ModelFactory:
    """
    Factory for creating ML models.
    
    Factory Pattern: Centralized model instantiation.
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class.
        
        Args:
            name: Model identifier (e.g., 'isolation_forest')
            model_class: Model class implementing BaseModel
        """
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create model instance.
        
        Args:
            model_type: Model identifier
            config: Configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type not registered
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        return model_class(name=model_type, config=config)
    
    @classmethod
    def get_available(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())


class DetectorFactory:
    """
    Factory for creating anomaly detectors.
    """
    
    _detectors: Dict[str, Type[BaseDetector]] = {}
    
    @classmethod
    def register(cls, name: str, detector_class: Type[BaseDetector]) -> None:
        """Register a detector class."""
        cls._detectors[name] = detector_class
    
    @classmethod
    def create(cls, detector_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDetector:
        """Create detector instance."""
        if detector_type not in cls._detectors:
            raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(cls._detectors.keys())}")
        
        detector_class = cls._detectors[detector_type]
        return detector_class(config=config)
    
    @classmethod
    def get_available(cls) -> list:
        """Get list of available detector types."""
        return list(cls._detectors.keys())


class ProcessorFactory:
    """
    Factory for creating data processors.
    """
    
    _processors: Dict[str, Type[BaseDataProcessor]] = {}
    
    @classmethod
    def register(cls, name: str, processor_class: Type[BaseDataProcessor]) -> None:
        """Register a processor class."""
        cls._processors[name] = processor_class
    
    @classmethod
    def create(cls, processor_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDataProcessor:
        """Create processor instance."""
        if processor_type not in cls._processors:
            raise ValueError(f"Unknown processor type: {processor_type}. Available: {list(cls._processors.keys())}")
        
        processor_class = cls._processors[processor_type]
        return processor_class(name=processor_type, config=config)
    
    @classmethod
    def get_available(cls) -> list:
        """Get list of available processor types."""
        return list(cls._processors.keys())

