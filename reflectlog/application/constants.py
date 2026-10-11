"""Centralized constants for ReflectLog application.

This module contains internal constants used throughout the application.
Configuration values should go in config/settings.py instead.
"""


# =============================================================================
# Search and Fusion Constants
# =============================================================================

# Minimum number of documents to fetch for better RRF fusion quality
# Using a small index (< 20 docs) doesn't provide enough diversity for fusion
MIN_OVERFETCH_LIMIT: int = 8
MAX_SEARCH_RECORDS_LIMIT: int = 50
MAX_RECORD_IDS: int = 100

# Tantivy BM25 scores are typically in the 0-10+ range
# Divide by this to normalize to 0-1 for comparison with semantic scores
TANTIVY_SCORE_DIVISOR: float = 10.0


# =============================================================================
# Logging Constants
# =============================================================================

# Maximum query length to include in log messages
# Prevents log spam from very long queries while preserving context
LOG_QUERY_TRUNCATE_LENGTH: int = 100

# Maximum number of memories to log individually during add operations
# Prevents log spam when adding many memories at once
LOG_ADD_MEMORY_PREVIEW_LIMIT: int = 20

# Length of separator lines in log output
LOG_SEPARATOR_LENGTH: int = 60


# =============================================================================
# Index Size Thresholds
# =============================================================================

# Small index threshold for adaptive overfetch (max multiplier)
# Indexes with <= this many documents get the max overfetch multiplier
ADAPTIVE_OVERFETCH_SMALL_INDEX_THRESHOLD: int = 100

# Large index threshold for adaptive overfetch (min multiplier)
# Indexes with >= this many documents get the min overfetch multiplier
ADAPTIVE_OVERFETCH_LARGE_INDEX_THRESHOLD: int = 10000


# =============================================================================
# Score Ranges
# =============================================================================

# Minimum value for percentage-based thresholds
MIN_PERCENTAGE_THRESHOLD: float = 0.0

# Maximum value for percentage-based thresholds
MAX_PERCENTAGE_THRESHOLD: float = 1.0


# =============================================================================
# API and Retry Constants
# =============================================================================

# Default maximum number of retries for API calls
DEFAULT_MAX_RETRIES: int = 3

# Base delay in seconds for exponential backoff
DEFAULT_RETRY_BASE_DELAY: float = 1.0


# =============================================================================
# Documentation
# =============================================================================

__all__ = [
    "ADAPTIVE_OVERFETCH_LARGE_INDEX_THRESHOLD",
    "ADAPTIVE_OVERFETCH_SMALL_INDEX_THRESHOLD",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_DELAY",
    "LOG_ADD_MEMORY_PREVIEW_LIMIT",
    "LOG_QUERY_TRUNCATE_LENGTH",
    "LOG_SEPARATOR_LENGTH",
    "MAX_PERCENTAGE_THRESHOLD",
    "MAX_RECORD_IDS",
    "MAX_SEARCH_RECORDS_LIMIT",
    "MIN_OVERFETCH_LIMIT",
    "MIN_PERCENTAGE_THRESHOLD",
    "TANTIVY_SCORE_DIVISOR",
]
