"""
Core module with base classes and design patterns.
"""

from .base import (
    BaseDetector,
    BaseModel,
    BaseDataProcessor,
    BaseValidator,
    Singleton,
    ConfigurableBase,
    Observable,
    Observer,
    LoggingObserver
)

from .factory import (
    ModelFactory,
    DetectorFactory,
    ProcessorFactory
)

from .builder import (
    PipelineBuilder,
    AnomalyDetectionPipeline
)

from .chunker import (
    DataChunker,
    StreamingProcessor,
    MemoryMonitor
)

from .logger import (
    LoggerManager,
    get_logger,
    setup_logging,
    get_detection_logger,
    get_performance_logger,
    PerformanceLogger,
    AnomalyDetectionLogger
)

from .validators import (
    DataValidator,
    ModelValidator,
    PipelineValidator,
    validate_pipeline_inputs
)

__all__ = [
    # Base classes
    'BaseDetector',
    'BaseModel',
    'BaseDataProcessor',
    'BaseValidator',
    'Singleton',
    'ConfigurableBase',
    'Observable',
    'Observer',
    'LoggingObserver',
    # Factories
    'ModelFactory',
    'DetectorFactory',
    'ProcessorFactory',
    # Builders
    'PipelineBuilder',
    'AnomalyDetectionPipeline',
    # Chunking
    'DataChunker',
    'StreamingProcessor',
    'MemoryMonitor',
    # Logging
    'LoggerManager',
    'get_logger',
    'setup_logging',
    'get_detection_logger',
    'get_performance_logger',
    'PerformanceLogger',
    'AnomalyDetectionLogger',
    # Validators
    'DataValidator',
    'ModelValidator',
    'PipelineValidator',
    'validate_pipeline_inputs'
]

