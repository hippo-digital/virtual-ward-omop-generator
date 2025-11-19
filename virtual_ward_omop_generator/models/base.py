"""Base models and interfaces for the virtual ward OMOP generator."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from ..logging_config import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all data models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the model configuration and state.
        
        Returns:
            True if valid, raises exception if invalid
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation.
        
        Returns:
            Dictionary representation of the model
        """
        return {"config": self.config}


class BaseGenerator(ABC):
    """Abstract base class for all data generators."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        concept_dict: pd.DataFrame,
        random_generator: np.random.Generator
    ) -> None:
        self.config = config
        self.concept_dict = concept_dict
        self.rng = random_generator
        self.logger = get_logger(self.__class__.__name__)
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate generator-specific configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def generate(self, **kwargs: Any) -> pd.DataFrame:
        """Generate data according to the configuration.
        
        Args:
            **kwargs: Generator-specific parameters
            
        Returns:
            Generated data as DataFrame
        """
        pass
    
    def _log_generation_stats(self, data: pd.DataFrame, data_type: str) -> None:
        """Log statistics about generated data.
        
        Args:
            data: Generated DataFrame
            data_type: Type of data for logging
        """
        self.logger.info(
            f"Generated {len(data)} {data_type} records with "
            f"{len(data.columns)} columns"
        )
        
        # Log memory usage
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        self.logger.debug(f"{data_type} data memory usage: {memory_mb:.2f} MB")


class BaseValidator(ABC):
    """Abstract base class for all data validators."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    @abstractmethod
    def validate(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate data according to rules.
        
        Args:
            data: Dictionary of table name to DataFrame
            
        Returns:
            True if validation passes
        """
        pass
    
    def add_error(self, message: str) -> None:
        """Add a validation error.
        
        Args:
            message: Error message
        """
        self.validation_errors.append(message)
        self.logger.error(f"Validation error: {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning.
        
        Args:
            message: Warning message
        """
        self.validation_warnings.append(message)
        self.logger.warning(f"Validation warning: {message}")
    
    def clear_results(self) -> None:
        """Clear validation results."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
    
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.validation_warnings) > 0


class BaseWriter(ABC):
    """Abstract base class for all data writers."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def write(self, data: Dict[str, pd.DataFrame], output_path: str) -> None:
        """Write data to the specified output.
        
        Args:
            data: Dictionary of table name to DataFrame
            output_path: Path for output
        """
        pass
    
    @abstractmethod
    def create_schema(self) -> None:
        """Create the database schema."""
        pass
    
    def _log_write_stats(self, data: Dict[str, pd.DataFrame]) -> None:
        """Log statistics about written data.
        
        Args:
            data: Dictionary of table data
        """
        total_records = sum(len(df) for df in data.values())
        total_tables = len(data)
        
        self.logger.info(
            f"Writing {total_records} total records across {total_tables} tables"
        )
        
        for table_name, df in data.items():
            self.logger.debug(f"Table {table_name}: {len(df)} records")


class ConfigurableComponent:
    """Mixin class for components that need configuration access."""
    
    def get_config_value(
        self, 
        key_path: str, 
        default: Any = None,
        required: bool = False
    ) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value if key is not found
            required: Whether the key is required
            
        Returns:
            Configuration value
            
        Raises:
            ConfigurationError: If required key is missing
        """
        from ..exceptions import ConfigurationError
        
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if required:
                raise ConfigurationError(f"Required configuration key missing: {key_path}")
            return default