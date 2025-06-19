# utils/logger.py

"""
A centralized, reusable, and robust logging module for the entire project.
Initializes a logger that writes to both the console and a rotating file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """
    Sets up and configures a logger.

    Args:
        name (str): The name for the logger (e.g., 'pipeline', 'chatbot').
        log_file (str): The full path to the log file to be created.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Ensure the directory for the log file exists.
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if this function is called multiple times.
    if logger.hasHandlers():
        return logger

    # File handler (writes to a file, rotates when the file gets large)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler (prints to the terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] -> %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add both handlers to the logger instance
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger