"""Logging configuration for the virtual ward OMOP generator."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "detailed"
) -> None:
    """Set up structured logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
        format_type: Format type - 'simple', 'detailed', or 'json'
    """
    # Define log formats
    formats = {
        'simple': '%(levelname)s - %(message)s',
        'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'json': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    }
    
    # Base configuration
    config: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': formats.get(format_type, formats['detailed']),
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'virtual_ward_omop_generator': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'standard',
            'filename': str(log_path),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        
        # Add file handler to loggers
        config['loggers']['virtual_ward_omop_generator']['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    logging.config.dictConfig(config)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {log_file or 'None'}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)