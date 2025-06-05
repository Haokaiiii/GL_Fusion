"""
Standardized logging configuration for the GL Fusion project.
Provides consistent logging setup across all modules.
"""

import logging
import sys
import os
from typing import Optional, List
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record):
        if self.use_color and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None,
    use_color: bool = True,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up standardized logging configuration.
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level
        log_file: Optional log file path
        console: Whether to log to console
        format_string: Custom format string
        use_color: Whether to use colored output for console
        propagate: Whether to propagate to parent logger
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(format_string, use_color=use_color)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Use standard formatter for file
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_distributed_logging(rank: int, world_size: int, **kwargs) -> logging.Logger:
    """
    Set up logging for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        **kwargs: Additional arguments for setup_logging
        
    Returns:
        Configured logger instance
    """
    # Only log to console from rank 0 by default
    console = kwargs.pop('console', rank == 0)
    
    # Add rank to logger name
    name = kwargs.get('name', 'distributed')
    name = f"{name}.rank{rank}"
    kwargs['name'] = name
    
    # Add rank to format
    format_string = kwargs.get('format_string', '%(asctime)s - [Rank %(rank)d] - %(name)s - %(levelname)s - %(message)s')
    kwargs['format_string'] = format_string
    
    # Create logger
    logger = setup_logging(console=console, **kwargs)
    
    # Add rank to all log records
    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True
    
    logger.addFilter(RankFilter())
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the project's standard configuration.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Check if already configured
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    # Configure with defaults
    return setup_logging(name=name)


def suppress_external_loggers(loggers: List[str], level: int = logging.WARNING):
    """
    Suppress verbose logging from external libraries.
    
    Args:
        loggers: List of logger names to suppress
        level: Minimum level to show from these loggers
    """
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(level)


# Common external loggers to suppress
EXTERNAL_LOGGERS = [
    'transformers',
    'torch.distributed',
    'deepspeed',
    'datasets',
    'tokenizers',
    'filelock',
    'huggingface_hub',
]


def configure_default_logging():
    """Configure default logging for the entire project."""
    # Set up root logger
    setup_logging(
        level=logging.INFO,
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        use_color=True
    )
    
    # Suppress external loggers
    suppress_external_loggers(EXTERNAL_LOGGERS)


# Convenience function for debug logging
def debug_log(message: str, data: Optional[dict] = None):
    """
    Convenience function for debug logging with optional data.
    
    Args:
        message: Log message
        data: Optional data dictionary to include
    """
    logger = logging.getLogger('debug')
    if data:
        message = f"{message} | Data: {data}"
    logger.debug(message) 