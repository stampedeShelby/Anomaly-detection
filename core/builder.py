"""
Builder Pattern for Pipeline Configuration

Fluent interface for building anomaly detection pipelines.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from .base import BaseDetector, BaseModel, Observable, Observer
from .factory import ModelFactory, DetectorFactory, ProcessorFactory


class PipelineBuilder:
    """
    Builder for constructing anomaly detection pipelines.
    
    Builder Pattern: Step-by-step construction with fluent interface.
    """
    
    def __init__(self):
        """Initialize builder with empty configuration."""
        self._detectors: List[BaseDetector] = []
        self._models: List[BaseModel] = []
        self._observers: List[Observer] = []
        self._config: Dict[str, Any] = {}
        self._verbose: bool = True
    
    def with_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """
        Add configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            self (for fluent interface)
        """
        self._config.update(config)
        return self
    
    def with_detector(self, detector_type: str, config: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """
        Add detector to pipeline.
        
        Args:
            detector_type: Detector identifier
            config: Detector-specific configuration
            
        Returns:
            self (for fluent interface)
        """
        detector = DetectorFactory.create(detector_type, config)
        self._detectors.append(detector)
        return self
    
    def with_model(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> 'PipelineBuilder':
        """
        Add model to pipeline.
        
        Args:
            model_type: Model identifier
            config: Model-specific configuration
            
        Returns:
            self (for fluent interface)
        """
        model = ModelFactory.create(model_type, config or {})
        self._models.append(model)
        return self
    
    def with_observer(self, observer: Observer) -> 'PipelineBuilder':
        """
        Add observer for events.
        
        Args:
            observer: Observer instance
            
        Returns:
            self (for fluent interface)
        """
        self._observers.append(observer)
        return self
    
    def verbose(self, enabled: bool = True) -> 'PipelineBuilder':
        """
        Enable/disable verbose output.
        
        Args:
            enabled: Verbose flag
            
        Returns:
            self (for fluent interface)
        """
        self._verbose = enabled
        return self
    
    def build(self) -> 'AnomalyDetectionPipeline':
        """
        Build and return pipeline instance.
        
        Returns:
            Configured pipeline
        """
        return AnomalyDetectionPipeline(
            detectors=self._detectors,
            models=self._models,
            observers=self._observers,
            config=self._config,
            verbose=self._verbose
        )
    
    def reset(self) -> 'PipelineBuilder':
        """Reset builder to initial state."""
        self._detectors = []
        self._models = []
        self._observers = []
        self._config = {}
        self._verbose = True
        return self


class AnomalyDetectionPipeline:
    """
    Composed anomaly detection pipeline.
    
    Orchestrates detectors, models, and observers.
    """
    
    def __init__(
        self,
        detectors: List[BaseDetector],
        models: List[BaseModel],
        observers: List[Observer],
        config: Dict[str, Any],
        verbose: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            detectors: List of detector instances
            models: List of model instances
            observers: List of observer instances
            config: Configuration dictionary
            verbose: Verbose output flag
        """
        self.detectors = detectors
        self.models = models
        self.observers = observers
        self.config = config
        self.verbose = verbose
        
        # Make pipeline observable
        self.events = Observable()
        for observer in observers:
            self.events.attach(observer)
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute pipeline on data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        if self.verbose:
            print("Running anomaly detection pipeline...")
        
        # Run detectors
        for detector in self.detectors:
            try:
                df = detector.detect(df)
                if self.verbose:
                    print(f"✓ Detector '{detector.name}' completed")
            except Exception as e:
                self.events.notify('error', {'detector': detector.name, 'error': str(e)})
                if self.verbose:
                    print(f"✗ Detector '{detector.name}' failed: {e}")
        
        results['detector_results'] = df
        
        # Run models
        for model in self.models:
            try:
                if not model.is_fitted:
                    # Fit on data
                    features = self._extract_features(df)
                    model.fit(features)
                
                scores = model.score(self._extract_features(df))
                if self.verbose:
                    print(f"✓ Model '{model.name}' scored")
            except Exception as e:
                self.events.notify('error', {'model': model.name, 'error': str(e)})
                if self.verbose:
                    print(f"✗ Model '{model.name}' failed: {e}")
        
        results['model_results'] = scores if 'scores' in locals() else None
        
        self.events.notify('complete', {'status': 'success'})
        
        return results
    
    def _extract_features(self, df: pd.DataFrame) -> Any:
        """Extract features for models. Placeholder implementation."""
        # This should be implemented based on actual feature extraction
        return df

