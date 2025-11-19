"""Custom exceptions for the virtual ward OMOP generator."""

from typing import Dict, Any, Optional


class VirtualWardOMOPError(Exception):
    """Base exception for all virtual ward OMOP generator errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class ConfigurationError(VirtualWardOMOPError):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class DataGenerationError(VirtualWardOMOPError):
    """Raised when data generation fails."""
    pass


class ValidationError(VirtualWardOMOPError):
    """Raised when data validation fails."""
    pass


class DatabaseError(VirtualWardOMOPError):
    """Raised when database operations fail."""
    pass