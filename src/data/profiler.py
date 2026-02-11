"""
Data Profiler Module

Provides comprehensive data profiling including:
- Schema detection
- Type inference
- Missing value analysis
- Data quality scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..config import MAX_CATEGORICAL_UNIQUE


@dataclass
class ColumnProfile:
 """Profile for a single column."""
 
 name: str
 dtype: str
 inferred_type: str # 'numeric', 'categorical', 'datetime', 'text', 'boolean'
 missing_count: int
 missing_percent: float
 unique_count: int
 unique_percent: float
 sample_values: List[Any] = field(default_factory=list)
 
 # Numeric-specific
 min_value: Optional[float] = None
 max_value: Optional[float] = None
 mean_value: Optional[float] = None
 std_value: Optional[float] = None
 
 # Categorical-specific
 top_values: Optional[Dict[str, int]] = None
 
 @property
 def is_numeric(self) -> bool:
 return self.inferred_type == "numeric"
 
 @property
 def is_categorical(self) -> bool:
 return self.inferred_type == "categorical"
 
 @property
 def quality_score(self) -> float:
 """Column quality score (0-100)."""
 # Penalize missing values and high cardinality in non-text columns
 completeness = 100 - self.missing_percent
 
 if self.inferred_type == "text":
 return completeness
 
 # Penalize if unique values are suspicious for the type
 if self.inferred_type == "numeric" and self.unique_percent > 99:
 # Likely an ID column, not analytically useful
 return completeness * 0.5
 
 return completeness


@dataclass
class DataProfile:
 """Complete profile for a dataset."""
 
 n_rows: int
 n_columns: int
 memory_usage_mb: float
 columns: Dict[str, ColumnProfile]
 
 # Summary counts by type
 numeric_columns: List[str] = field(default_factory=list)
 categorical_columns: List[str] = field(default_factory=list)
 datetime_columns: List[str] = field(default_factory=list)
 text_columns: List[str] = field(default_factory=list)
 boolean_columns: List[str] = field(default_factory=list)
 
 # Overall quality
 overall_missing_percent: float = 0.0
 data_quality_score: float = 0.0
 
 # Issues detected
 issues: List[str] = field(default_factory=list)
 recommendations: List[str] = field(default_factory=list)
 
 @property
 def summary(self) -> Dict[str, Any]:
 """Summary for display."""
 return {
 "rows": f"{self.n_rows:,}",
 "columns": self.n_columns,
 "memory": f"{self.memory_usage_mb:.2f} MB",
 "quality_score": f"{self.data_quality_score:.1f}%",
 "missing": f"{self.overall_missing_percent:.1f}%",
 "numeric": len(self.numeric_columns),
 "categorical": len(self.categorical_columns),
 "datetime": len(self.datetime_columns),
 }
 
 def get_schema_string(self) -> str:
 """Get schema as formatted string for LLM prompts."""
 lines = []
 for name, profile in self.columns.items():
 line = f"- {name} ({profile.inferred_type})"
 if profile.is_numeric:
 line += f" [range: {profile.min_value:.2f} to {profile.max_value:.2f}]"
 elif profile.is_categorical and profile.top_values:
 top_3 = list(profile.top_values.keys())[:3]
 line += f" [values: {', '.join(map(str, top_3))}...]"
 lines.append(line)
 return "\n".join(lines)


class DataProfiler:
 """
 Profiles datasets to understand structure and quality.
 
 Performs:
 - Type inference (beyond pandas dtypes)
 - Missing value analysis
 - Quality scoring
 - Issue detection
 """
 
 def __init__(self, max_categorical_unique: int = MAX_CATEGORICAL_UNIQUE):
 """
 Initialize the profiler.
 
 Args:
 max_categorical_unique: Max unique values to treat as categorical
 """
 self.max_categorical_unique = max_categorical_unique
 
 def profile(self, df: pd.DataFrame) -> DataProfile:
 """
 Create comprehensive profile of a DataFrame.
 
 Args:
 df: Input DataFrame
 
 Returns:
 DataProfile with all metadata
 """
 n_rows, n_columns = df.shape
 memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
 
 # Profile each column
 column_profiles = {}
 numeric_cols = []
 categorical_cols = []
 datetime_cols = []
 text_cols = []
 boolean_cols = []
 
 for col in df.columns:
 profile = self._profile_column(df[col], n_rows)
 column_profiles[col] = profile
 
 # Categorize columns
 if profile.inferred_type == "numeric":
 numeric_cols.append(col)
 elif profile.inferred_type == "categorical":
 categorical_cols.append(col)
 elif profile.inferred_type == "datetime":
 datetime_cols.append(col)
 elif profile.inferred_type == "text":
 text_cols.append(col)
 elif profile.inferred_type == "boolean":
 boolean_cols.append(col)
 
 # Calculate overall metrics
 total_cells = n_rows * n_columns
 total_missing = sum(p.missing_count for p in column_profiles.values())
 overall_missing_percent = (total_missing / total_cells * 100) if total_cells > 0 else 0
 
 # Calculate quality score
 quality_scores = [p.quality_score for p in column_profiles.values()]
 data_quality_score = np.mean(quality_scores) if quality_scores else 100
 
 # Detect issues
 issues, recommendations = self._detect_issues(df, column_profiles)
 
 return DataProfile(
 n_rows=n_rows,
 n_columns=n_columns,
 memory_usage_mb=memory_usage_mb,
 columns=column_profiles,
 numeric_columns=numeric_cols,
 categorical_columns=categorical_cols,
 datetime_columns=datetime_cols,
 text_columns=text_cols,
 boolean_columns=boolean_cols,
 overall_missing_percent=overall_missing_percent,
 data_quality_score=data_quality_score,
 issues=issues,
 recommendations=recommendations,
 )
 
 def _profile_column(self, series: pd.Series, n_rows: int) -> ColumnProfile:
 """
 Profile a single column.
 
 Args:
 series: Column data
 n_rows: Total rows in dataset
 
 Returns:
 ColumnProfile for the column
 """
 name = series.name
 dtype = str(series.dtype)
 
 # Missing values
 missing_count = series.isna().sum()
 missing_percent = (missing_count / n_rows * 100) if n_rows > 0 else 0
 
 # Unique values
 unique_count = series.nunique()
 unique_percent = (unique_count / n_rows * 100) if n_rows > 0 else 0
 
 # Sample values (non-null)
 non_null = series.dropna()
 sample_values = non_null.head(5).tolist() if len(non_null) > 0 else []
 
 # Infer semantic type
 inferred_type = self._infer_type(series, unique_count)
 
 # Type-specific profiling
 min_val = max_val = mean_val = std_val = None
 top_values = None
 
 if inferred_type == "numeric":
 numeric_series = pd.to_numeric(series, errors="coerce")
 min_val = float(numeric_series.min()) if not pd.isna(numeric_series.min()) else None
 max_val = float(numeric_series.max()) if not pd.isna(numeric_series.max()) else None
 mean_val = float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else None
 std_val = float(numeric_series.std()) if not pd.isna(numeric_series.std()) else None
 
 elif inferred_type == "categorical":
 value_counts = series.value_counts().head(10).to_dict()
 top_values = {str(k): int(v) for k, v in value_counts.items()}
 
 return ColumnProfile(
 name=name,
 dtype=dtype,
 inferred_type=inferred_type,
 missing_count=missing_count,
 missing_percent=missing_percent,
 unique_count=unique_count,
 unique_percent=unique_percent,
 sample_values=sample_values,
 min_value=min_val,
 max_value=max_val,
 mean_value=mean_val,
 std_value=std_val,
 top_values=top_values,
 )
 
 def _infer_type(self, series: pd.Series, unique_count: int) -> str:
 """
 Infer semantic type of a column.
 
 Args:
 series: Column data
 unique_count: Number of unique values
 
 Returns:
 Inferred type string
 """
 dtype = series.dtype
 
 # Boolean check
 if dtype == bool or (unique_count <= 2 and set(series.dropna().unique()).issubset({0, 1, True, False, "Yes", "No", "Y", "N", "true", "false"})):
 return "boolean"
 
 # Numeric check
 if pd.api.types.is_numeric_dtype(series):
 return "numeric"
 
 # Datetime check
 if pd.api.types.is_datetime64_any_dtype(series):
 return "datetime"
 
 # Try to parse as datetime
 if dtype == object:
 try:
 sample = series.dropna().head(100)
 if len(sample) > 0:
 pd.to_datetime(sample, errors="raise")
 return "datetime"
 except (ValueError, TypeError):
 pass
 
 # Categorical vs text
 if dtype == object or dtype.name == "category":
 if unique_count <= self.max_categorical_unique:
 return "categorical"
 else:
 # Check average string length - long strings are likely text
 avg_len = series.dropna().astype(str).apply(len).mean()
 if avg_len > 50:
 return "text"
 else:
 return "categorical"
 
 return "text"
 
 def _detect_issues(
 self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]
 ) -> tuple[List[str], List[str]]:
 """
 Detect data quality issues and generate recommendations.
 
 Args:
 df: Input DataFrame
 profiles: Column profiles
 
 Returns:
 Tuple of (issues, recommendations)
 """
 issues = []
 recommendations = []
 
 for name, profile in profiles.items():
 # High missing values
 if profile.missing_percent > 30:
 issues.append(
 f"Column '{name}' has {profile.missing_percent:.1f}% missing values"
 )
 recommendations.append(
 f"Consider imputing or removing column '{name}'"
 )
 
 # Constant columns
 if profile.unique_count == 1:
 issues.append(f"Column '{name}' has only one unique value")
 recommendations.append(
 f"Column '{name}' provides no analytical value, consider removing"
 )
 
 # High cardinality in categorical
 if profile.inferred_type == "categorical" and profile.unique_percent > 90:
 issues.append(
 f"Column '{name}' has very high cardinality ({profile.unique_count} unique values)"
 )
 
 # Duplicate rows check
 dup_count = df.duplicated().sum()
 if dup_count > 0:
 dup_percent = dup_count / len(df) * 100
 if dup_percent > 5:
 issues.append(f"Dataset has {dup_count:,} duplicate rows ({dup_percent:.1f}%)")
 recommendations.append("Consider removing duplicate rows")
 
 return issues, recommendations
 
 def get_numeric_summary(self, df: pd.DataFrame) -> pd.DataFrame:
 """
 Get summary statistics for numeric columns.
 
 Args:
 df: Input DataFrame
 
 Returns:
 Summary statistics DataFrame
 """
 numeric_df = df.select_dtypes(include=[np.number])
 if numeric_df.empty:
 return pd.DataFrame()
 
 summary = numeric_df.describe().T
 summary["missing"] = numeric_df.isna().sum()
 summary["missing_pct"] = (summary["missing"] / len(df) * 100).round(2)
 
 return summary
 
 def get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
 """
 Get value counts for categorical columns.
 
 Args:
 df: Input DataFrame
 
 Returns:
 Dictionary of column name to value counts
 """
 cat_cols = df.select_dtypes(include=["object", "category"]).columns
 summaries = {}
 
 for col in cat_cols:
 if df[col].nunique() <= self.max_categorical_unique:
 summaries[col] = df[col].value_counts()
 
 return summaries
