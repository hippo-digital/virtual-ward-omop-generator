"""Configuration management for the virtual ward OMOP generator."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages configuration loading, validation, and random seed management."""
    
    def __init__(self) -> None:
        self._config: Optional[Dict[str, Any]] = None
        self._concept_dict: Optional[pd.DataFrame] = None
        self._random_generator: Optional[np.random.Generator] = None
        
    def load_yaml_spec(self, spec_path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration specification.
        
        Args:
            spec_path: Path to the YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be loaded or is invalid
        """
        try:
            spec_file = Path(spec_path)
            if not spec_file.exists():
                raise ConfigurationError(f"Configuration file not found: {spec_path}")
                
            with open(spec_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not isinstance(config, dict):
                raise ConfigurationError("Configuration must be a dictionary")
                
            self._config = config
            self._validate_configuration(config)
            
            logger.info(f"Successfully loaded configuration from {spec_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def load_concept_dictionary(self, csv_path: str) -> pd.DataFrame:
        """Load concept dictionary from CSV file.
        
        Args:
            csv_path: Path to the concept dictionary CSV file
            
        Returns:
            DataFrame containing concept mappings
            
        Raises:
            ConfigurationError: If file cannot be loaded or has invalid format
        """
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                raise ConfigurationError(f"Concept dictionary not found: {csv_path}")
                
            concept_dict = pd.read_csv(csv_file)
            
            # Validate required columns
            required_columns = ['concept_id', 'concept_name', 'concept_code']
            missing_columns = [col for col in required_columns if col not in concept_dict.columns]
            if missing_columns:
                raise ConfigurationError(f"Missing required columns in concept dictionary: {missing_columns}")
                
            # Ensure concept_id is integer and >= 2,100,000,000 for custom concepts
            if not pd.api.types.is_integer_dtype(concept_dict['concept_id']):
                raise ConfigurationError("concept_id column must contain integers")
                
            custom_concepts = concept_dict[concept_dict['concept_id'] >= 2_100_000_000]
            if len(custom_concepts) == 0:
                logger.warning("No custom concepts found (concept_id >= 2,100,000,000)")
                
            self._concept_dict = concept_dict
            logger.info(f"Loaded {len(concept_dict)} concepts from {csv_path}")
            return concept_dict
            
        except pd.errors.EmptyDataError:
            raise ConfigurationError("Concept dictionary CSV file is empty")
        except Exception as e:
            raise ConfigurationError(f"Failed to load concept dictionary: {e}")
    

    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Public interface for configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        return self._validate_configuration(config)
    
    def _validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Required top-level sections
        required_sections = ['dataset', 'types']
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate dataset settings
        dataset_config = config['dataset']
        if not isinstance(dataset_config.get('persons'), int) or dataset_config['persons'] < 1:
            raise ConfigurationError("Dataset persons must be an integer >= 1")
        if dataset_config['persons'] > 20000:
            raise ConfigurationError("Dataset persons must be <= 20000")
            
        # Validate condition mix if present
        if 'condition_mix' in dataset_config:
            condition_mix = dataset_config['condition_mix']
            if not isinstance(condition_mix, dict):
                raise ConfigurationError("condition_mix must be a dictionary")
                
            # Validate percentages sum to 1.0 (not 100)
            total_pct = sum(condition_mix.values())
            if abs(total_pct - 1.0) > 0.01:  # Allow small floating point errors
                raise ConfigurationError(f"Condition mix percentages must sum to 1.0, got {total_pct}")
        
        # Validate episode settings if present
        if 'episodes_per_person' in dataset_config:
            episodes_config = dataset_config['episodes_per_person']
            if not isinstance(episodes_config, dict):
                raise ConfigurationError("episodes_per_person must be a dictionary")
            if 'min' not in episodes_config or 'max' not in episodes_config:
                raise ConfigurationError("episodes_per_person must have 'min' and 'max' keys")
                
        # Validate episode length settings if present
        if 'episode_length_days' in dataset_config:
            length_config = dataset_config['episode_length_days']
            if not isinstance(length_config, dict):
                raise ConfigurationError("episode_length_days must be a dictionary")
            if 'modes' not in length_config or 'probs' not in length_config:
                raise ConfigurationError("episode_length_days must have 'modes' and 'probs' keys")
            
            modes = length_config['modes']
            probs = length_config['probs']
            if len(modes) != len(probs):
                raise ConfigurationError("episode_length_days modes and probs must have same length")
            if abs(sum(probs) - 1.0) > 0.01:
                raise ConfigurationError("episode_length_days probs must sum to 1.0")
        
        # Validate types section
        types_config = config['types']
        required_types = ['observation_type_patient_reported', 'measurement_type_patient_device', 
                         'period_type_virtual_ward', 'yes', 'no']
        for type_key in required_types:
            if type_key not in types_config:
                logger.warning(f"Missing recommended type: {type_key}")
        
        # Validate vocabulary section if present
        if 'vocabulary' in config:
            vocab_config = config['vocabulary']
            if 'concept_file' not in vocab_config:
                logger.warning("vocabulary section missing concept_file")
        
        # Validate OMOP section if present
        if 'omop' in config:
            omop_config = config['omop']
            if 'tables' not in omop_config:
                logger.warning("omop section missing tables configuration")
            
        logger.info("Configuration validation passed")
        return True
    
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration."""
        return self._config
    
    def get_concept_dictionary(self) -> Optional[pd.DataFrame]:
        """Get the loaded concept dictionary."""
        return self._concept_dict
    
    def get_random_generator(self, seed: Optional[int] = None) -> np.random.Generator:
        """Get or create a random number generator with specified seed.
        
        Args:
            seed: Random seed for reproducible generation. If None, uses existing generator or creates new one.
            
        Returns:
            NumPy random generator instance
        """
        if self._random_generator is None or seed is not None:
            if seed is None and self._config:
                seed = self._config.get('dataset', {}).get('random_seed', 42)
            elif seed is None:
                seed = 42
                
            self._random_generator = np.random.default_rng(seed)
            logger.info(f"Initialized random generator with seed: {seed}")
            
        return self._random_generator
    
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration."""
        return self._config
    
    @property
    def concept_dictionary(self) -> Optional[pd.DataFrame]:
        """Get the loaded concept dictionary."""
        return self._concept_dict
    
    @property
    def random_generator(self) -> Optional[np.random.Generator]:
        """Get the current random generator."""
        return self._random_generator