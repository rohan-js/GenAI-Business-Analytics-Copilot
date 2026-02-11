"""
Data package for handling data ingestion and profiling.
"""

from .ingestion import DataIngestionModule
from .profiler import DataProfiler

__all__ = ["DataIngestionModule", "DataProfiler"]
