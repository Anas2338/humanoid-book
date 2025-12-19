import logging
import sys
from datetime import datetime
from typing import Any, Dict
import json
from functools import wraps

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get root logger and add handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Set specific log levels for different modules
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

# Call setup at module level
setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

# Custom exception classes
class RAGChatbotError(Exception):
    """Base exception class for RAG Chatbot errors"""
    def __init__(self, message: str, error_code: str = "RAG_ERROR", details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details
            }
        }

class ValidationError(RAGChatbotError):
    """Exception raised for validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )

class DatabaseError(RAGChatbotError):
    """Exception raised for database errors"""
    def __init__(self, message: str, operation: str = None):
        details = {}
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details
        )

class VectorDBError(RAGChatbotError):
    """Exception raised for vector database errors"""
    def __init__(self, message: str, operation: str = None):
        details = {}
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code="VECTOR_DB_ERROR",
            details=details
        )

class APIError(RAGChatbotError):
    """Exception raised for API errors"""
    def __init__(self, message: str, status_code: int = 500, response: str = None):
        details = {"status_code": status_code}
        if response:
            details["response"] = response

        super().__init__(
            message=message,
            error_code="API_ERROR",
            details=details
        )

# Decorator for logging function calls
def log_function_call(logger: logging.Logger = None):
    """Decorator to log function calls with parameters and results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger

            # Log function entry
            func_logger.info(f"Calling function: {func.__name__}")

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log successful completion
                func_logger.info(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                # Log error
                func_logger.error(f"Function {func.__name__} failed with error: {str(e)}")
                raise

        return wrapper
    return decorator

# Utility function for structured logging
def log_api_call(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    status_code: int,
    response_time: float,
    user_id: str = None,
    session_id: str = None
):
    """Log API call with structured information"""
    log_data = {
        "type": "api_call",
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time,
        "timestamp": datetime.utcnow().isoformat()
    }

    if user_id:
        log_data["user_id"] = user_id
    if session_id:
        log_data["session_id"] = session_id

    logger.info(json.dumps(log_data))

def log_error_event(
    logger: logging.Logger,
    error_type: str,
    error_message: str,
    context: Dict[str, Any] = None
):
    """Log error events with context"""
    log_data = {
        "type": "error",
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat()
    }

    if context:
        log_data["context"] = context

    logger.error(json.dumps(log_data))

def log_performance_event(
    logger: logging.Logger,
    event_name: str,
    duration_ms: float,
    context: Dict[str, Any] = None
):
    """Log performance-related events"""
    log_data = {
        "type": "performance",
        "event": event_name,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat()
    }

    if context:
        log_data["context"] = context

    logger.info(json.dumps(log_data))