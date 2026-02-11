"""
Data Ingestion Module

Handles CSV/Excel file uploads with:
- Automatic encoding detection
- Schema detection
- Type inference
- Missing value profiling
"""

import io
import chardet
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

from ..config import (
 SUPPORTED_EXTENSIONS,
 MISSING_VALUE_INDICATORS,
 MAX_FILE_SIZE_MB,
 MAX_SAMPLE_ROWS,
)


@dataclass
class IngestionResult:
 """Result of data ingestion with metadata."""
 
 df: pd.DataFrame
 filename: str
 file_size_mb: float
 encoding: str
 original_rows: int
 original_columns: int
 is_sampled: bool = False
 sample_rows: Optional[int] = None
 load_time_seconds: float = 0.0
 warnings: List[str] = field(default_factory=list)
 
 @property
 def shape_str(self) -> str:
 """Human-readable shape string."""
 return f"{self.original_rows:,} rows × {self.original_columns} columns"
 
 @property
 def summary(self) -> Dict[str, Any]:
 """Summary dictionary for display."""
 return {
 "filename": self.filename,
 "size": f"{self.file_size_mb:.2f} MB",
 "shape": self.shape_str,
 "encoding": self.encoding,
 "sampled": self.is_sampled,
 "load_time": f"{self.load_time_seconds:.2f}s",
 }


class DataIngestionModule:
 """
 Handles data file ingestion with automatic detection and validation.
 
 Features:
 - Multi-format support (CSV, Excel)
 - Automatic encoding detection for CSV
 - Large file handling with sampling
 - Missing value standardization
 """
 
 def __init__(self, max_sample_rows: int = MAX_SAMPLE_ROWS):
 """
 Initialize the ingestion module.
 
 Args:
 max_sample_rows: Maximum rows to keep for LLM operations
 """
 self.max_sample_rows = max_sample_rows
 self._supported_extensions = SUPPORTED_EXTENSIONS
 
 def load(
 self,
 source: Union[str, Path, io.BytesIO],
 filename: Optional[str] = None,
 sample_for_llm: bool = True,
 ) -> IngestionResult:
 """
 Load data from file path or uploaded bytes.
 
 Args:
 source: File path or BytesIO object from upload
 filename: Original filename (required for BytesIO)
 sample_for_llm: Whether to sample large datasets
 
 Returns:
 IngestionResult with loaded DataFrame and metadata
 """
 import time
 start_time = time.time()
 
 warnings = []
 
 # Handle file path vs bytes
 if isinstance(source, (str, Path)):
 filepath = Path(source)
 filename = filepath.name
 file_bytes = filepath.read_bytes()
 file_size_mb = len(file_bytes) / (1024 * 1024)
 else:
 # BytesIO from Streamlit upload
 file_bytes = source.getvalue()
 file_size_mb = len(file_bytes) / (1024 * 1024)
 if filename is None:
 raise ValueError("filename required for BytesIO source")
 
 # Validate file size
 if file_size_mb > MAX_FILE_SIZE_MB:
 raise ValueError(
 f"File too large ({file_size_mb:.1f}MB). "
 f"Maximum allowed: {MAX_FILE_SIZE_MB}MB"
 )
 
 # Detect file type and load
 extension = Path(filename).suffix.lower()
 
 if extension not in self._supported_extensions:
 raise ValueError(
 f"Unsupported file type: {extension}. "
 f"Supported: {', '.join(self._supported_extensions)}"
 )
 
 if extension == ".csv":
 df, encoding = self._load_csv(file_bytes)
 elif extension in [".xlsx", ".xls"]:
 df = self._load_excel(file_bytes)
 encoding = "N/A"
 else:
 raise ValueError(f"Unsupported extension: {extension}")
 
 # Store original shape
 original_rows, original_columns = df.shape
 
 # Standardize missing values
 df = self._standardize_missing_values(df)
 
 # Sample if needed
 is_sampled = False
 sample_rows = None
 
 if sample_for_llm and len(df) > self.max_sample_rows:
 is_sampled = True
 sample_rows = self.max_sample_rows
 df = df.sample(n=self.max_sample_rows, random_state=42)
 warnings.append(
 f"Dataset sampled from {original_rows:,} to {self.max_sample_rows:,} rows for analysis"
 )
 
 # Clean column names
 df = self._clean_column_names(df)
 
 load_time = time.time() - start_time
 
 return IngestionResult(
 df=df,
 filename=filename,
 file_size_mb=file_size_mb,
 encoding=encoding,
 original_rows=original_rows,
 original_columns=original_columns,
 is_sampled=is_sampled,
 sample_rows=sample_rows,
 load_time_seconds=load_time,
 warnings=warnings,
 )
 
 def _load_csv(self, file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
 """
 Load CSV with automatic encoding detection.
 
 Args:
 file_bytes: Raw file bytes
 
 Returns:
 Tuple of (DataFrame, detected encoding)
 """
 # Detect encoding
 detection = chardet.detect(file_bytes[:10000]) # Sample first 10KB
 encoding = detection.get("encoding", "utf-8")
 
 # Handle common encoding issues
 encoding_fallbacks = [encoding, "utf-8", "latin-1", "cp1252"]
 
 for enc in encoding_fallbacks:
 try:
 df = pd.read_csv(
 io.BytesIO(file_bytes),
 encoding=enc,
 low_memory=False,
 )
 return df, enc
 except (UnicodeDecodeError, pd.errors.ParserError):
 continue
 
 raise ValueError(
 f"Could not decode file. Tried encodings: {encoding_fallbacks}"
 )
 
 def _load_excel(self, file_bytes: bytes) -> pd.DataFrame:
 """
 Load Excel file.
 
 Args:
 file_bytes: Raw file bytes
 
 Returns:
 DataFrame from first sheet
 """
 # Read first sheet by default
 df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
 return df
 
 def _standardize_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
 """
 Standardize missing value representations to NaN.
 
 Args:
 df: Input DataFrame
 
 Returns:
 DataFrame with standardized missing values
 """
 # Replace common missing value indicators
 df = df.replace(MISSING_VALUE_INDICATORS, np.nan)
 
 # Strip whitespace from string columns and replace empty strings
 for col in df.select_dtypes(include=["object"]).columns:
 df[col] = df[col].apply(
 lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
 )
 
 return df
 
 def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
 """
 Clean and standardize column names.
 
 Args:
 df: Input DataFrame
 
 Returns:
 DataFrame with cleaned column names
 """
 # Strip whitespace
 df.columns = df.columns.str.strip()
 
 # Replace problematic characters
 df.columns = df.columns.str.replace(r'[\[\]\<\>]', '', regex=True)
 
 # Ensure unique column names
 seen = {}
 new_columns = []
 for col in df.columns:
 if col in seen:
 seen[col] += 1
 new_columns.append(f"{col}_{seen[col]}")
 else:
 seen[col] = 0
 new_columns.append(col)
 
 df.columns = new_columns
 
 return df
 
 def get_sample_data(
 self, df: pd.DataFrame, n_rows: int = 5
 ) -> pd.DataFrame:
 """
 Get sample rows for display/prompts.
 
 Args:
 df: Input DataFrame
 n_rows: Number of sample rows
 
 Returns:
 Sample DataFrame
 """
 return df.head(n_rows)
 
 def validate_file(
 self, filename: str, file_size_bytes: int
 ) -> Tuple[bool, Optional[str]]:
 """
 Validate file before loading.
 
 Args:
 filename: Name of the file
 file_size_bytes: Size in bytes
 
 Returns:
 Tuple of (is_valid, error_message)
 """
 extension = Path(filename).suffix.lower()
 
 if extension not in self._supported_extensions:
 return False, f"Unsupported file type: {extension}"
 
 file_size_mb = file_size_bytes / (1024 * 1024)
 if file_size_mb > MAX_FILE_SIZE_MB:
 return False, f"File too large ({file_size_mb:.1f}MB max: {MAX_FILE_SIZE_MB}MB)"
 
 return True, None
