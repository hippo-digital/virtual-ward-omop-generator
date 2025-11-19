"""Utility functions including error recovery mechanisms."""

import logging
import time
import functools
from typing import Callable, TypeVar, Any, Optional, Type, Tuple
from .exceptions import VirtualWardOMOPError, DataGenerationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (DataGenerationError,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry function calls on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Final error: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Error: {e}. Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"Function {func.__name__} failed with unexpected error: {e}")
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default_value: Optional[T] = None,
    log_errors: bool = True,
    **kwargs: Any
) -> Optional[T]:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_value: Value to return if function fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_value if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Safe execution of {func.__name__} failed: {e}")
        return default_value


def validate_and_convert_types(
    data: Any,
    expected_type: Type[T],
    field_name: str = "data"
) -> T:
    """Validate and convert data to expected type with error handling.
    
    Args:
        data: Data to validate and convert
        expected_type: Expected type for the data
        field_name: Name of the field for error messages
        
    Returns:
        Converted data
        
    Raises:
        DataGenerationError: If conversion fails
    """
    try:
        if isinstance(data, expected_type):
            return data
        
        # Try to convert
        if expected_type == int:
            return int(data)  # type: ignore
        elif expected_type == float:
            return float(data)  # type: ignore
        elif expected_type == str:
            return str(data)  # type: ignore
        else:
            return expected_type(data)  # type: ignore
            
    except (ValueError, TypeError) as e:
        raise DataGenerationError(
            f"Failed to convert {field_name} to {expected_type.__name__}: {e}",
            context={"data": data, "expected_type": expected_type.__name__}
        )


class ErrorContext:
    """Context manager for enhanced error reporting."""
    
    def __init__(self, operation: str, **context: Any) -> None:
        self.operation = operation
        self.context = context
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self) -> 'ErrorContext':
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type: Optional[Type[Exception]], 
                 exc_val: Optional[Exception], 
                 exc_tb: Any) -> None:
        if exc_type is not None:
            error_msg = f"Operation '{self.operation}' failed: {exc_val}"
            if self.context:
                error_msg += f" Context: {self.context}"
            self.logger.error(error_msg)
            
            # Re-raise as VirtualWardOMOPError if it's not already one
            if not isinstance(exc_val, VirtualWardOMOPError):
                raise DataGenerationError(
                    f"Operation '{self.operation}' failed: {exc_val}",
                    context=self.context
                ) from exc_val
        else:
            self.logger.debug(f"Operation completed successfully: {self.operation}")


def log_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time.
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper