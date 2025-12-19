import time
import functools
from typing import Callable, Any
from utils.logging import log_performance_event, get_logger
from utils.logging import log_error_event

logger = get_logger(__name__)

class PerformanceMonitor:
    """Utility class for monitoring performance of various operations"""

    def __init__(self):
        self.metrics = {}

    def measure_api_call(self, endpoint: str, method: str):
        """Decorator to measure API call performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 200  # Default to success
                session_id = kwargs.get('sessionId', 'unknown')

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = 500  # Mark as error
                    raise
                finally:
                    duration = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Log performance event
                    log_performance_event(
                        logger,
                        f"api_call_{endpoint.replace('/', '_')}",
                        duration,
                        {
                            "method": method,
                            "status_code": status_code,
                            "session_id": session_id
                        }
                    )

                    # Track metrics for monitoring
                    self._update_metrics(f"api.{endpoint.replace('/', '.')}.{method}", duration, status_code)

            return wrapper
        return decorator

    def measure_function(self, name: str):
        """Decorator to measure function performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = (time.time() - start_time) * 1000  # Convert to milliseconds

                    # Log performance event
                    log_performance_event(
                        logger,
                        f"function.{name}",
                        duration,
                        {
                            "function_name": name,
                        }
                    )

                    # Track metrics
                    self._update_metrics(f"function.{name}", duration)

            return wrapper
        return decorator

    def _update_metrics(self, operation: str, duration: float, status_code: int = 200):
        """Update internal metrics for the operation"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'success_count': 0,
                'error_count': 0
            }

        metrics = self.metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)

        if status_code == 200:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1

    def get_metrics(self, operation: str = None) -> dict:
        """Get performance metrics, optionally for a specific operation"""
        if operation:
            return self.metrics.get(operation, {})
        return self.metrics.copy()

    def get_average_response_time(self, operation: str) -> float:
        """Get average response time for an operation"""
        metrics = self.metrics.get(operation, {})
        if metrics.get('count', 0) == 0:
            return 0
        return metrics['total_time'] / metrics['count']

    def get_error_rate(self, operation: str) -> float:
        """Get error rate for an operation"""
        metrics = self.metrics.get(operation, {})
        total = metrics.get('count', 0)
        if total == 0:
            return 0
        return (metrics.get('error_count', 0) / total) * 100

# Global performance monitor instance
perf_monitor = PerformanceMonitor()

# Convenience functions
def monitor_api(endpoint: str, method: str):
    """Convenience function to monitor API endpoints"""
    return perf_monitor.measure_api_call(endpoint, method)

def monitor_function(name: str):
    """Convenience function to monitor specific functions"""
    return perf_monitor.measure_function(name)