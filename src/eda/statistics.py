"""
Statistics Calculator Module

Computes summary statistics for datasets:
- Descriptive statistics (mean, median, std, etc.)
- Distribution analysis
- Percentile calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ColumnStatistics:
 """Statistics for a single numeric column."""
 
 name: str
 count: int
 missing: int
 mean: float
 median: float
 std: float
 min: float
 max: float
 q1: float # 25th percentile
 q3: float # 75th percentile
 iqr: float
 skewness: float
 kurtosis: float
 
 @property
 def summary_dict(self) -> Dict[str, Any]:
 """Get as dictionary for display."""
 return {
 "Count": f"{self.count:,}",
 "Missing": f"{self.missing:,}",
 "Mean": f"{self.mean:,.2f}",
 "Median": f"{self.median:,.2f}",
 "Std Dev": f"{self.std:,.2f}",
 "Min": f"{self.min:,.2f}",
 "Max": f"{self.max:,.2f}",
 "IQR": f"{self.iqr:,.2f}",
 }
 
 @property
 def distribution_type(self) -> str:
 """Infer distribution shape from skewness."""
 if abs(self.skewness) < 0.5:
 return "approximately_normal"
 elif self.skewness > 0.5:
 return "right_skewed"
 else:
 return "left_skewed"


@dataclass
class CategoricalStatistics:
 """Statistics for a categorical column."""
 
 name: str
 count: int
 missing: int
 unique_count: int
 mode: str
 mode_frequency: int
 mode_percent: float
 value_counts: Dict[str, int] = field(default_factory=dict)
 
 @property
 def summary_dict(self) -> Dict[str, Any]:
 """Get as dictionary for display."""
 return {
 "Count": f"{self.count:,}",
 "Missing": f"{self.missing:,}",
 "Unique Values": self.unique_count,
 "Mode": str(self.mode),
 "Mode Frequency": f"{self.mode_frequency:,} ({self.mode_percent:.1f}%)",
 }


class StatisticsCalculator:
 """
 Calculates comprehensive statistics for datasets.
 
 Handles both numeric and categorical columns with
 appropriate statistical measures for each type.
 """
 
 def __init__(self):
 """Initialize the statistics calculator."""
 pass
 
 def compute_numeric_stats(
 self, df: pd.DataFrame, columns: Optional[List[str]] = None
 ) -> Dict[str, ColumnStatistics]:
 """
 Compute statistics for numeric columns.
 
 Args:
 df: Input DataFrame
 columns: Specific columns to analyze (None = all numeric)
 
 Returns:
 Dictionary of column name to ColumnStatistics
 """
 if columns is None:
 numeric_df = df.select_dtypes(include=[np.number])
 columns = numeric_df.columns.tolist()
 else:
 numeric_df = df[columns].select_dtypes(include=[np.number])
 columns = numeric_df.columns.tolist()
 
 results = {}
 
 for col in columns:
 series = df[col]
 clean_series = series.dropna()
 
 if len(clean_series) == 0:
 continue
 
 stats = ColumnStatistics(
 name=col,
 count=len(clean_series),
 missing=series.isna().sum(),
 mean=float(clean_series.mean()),
 median=float(clean_series.median()),
 std=float(clean_series.std()) if len(clean_series) > 1 else 0.0,
 min=float(clean_series.min()),
 max=float(clean_series.max()),
 q1=float(clean_series.quantile(0.25)),
 q3=float(clean_series.quantile(0.75)),
 iqr=float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
 skewness=float(clean_series.skew()) if len(clean_series) > 2 else 0.0,
 kurtosis=float(clean_series.kurtosis()) if len(clean_series) > 3 else 0.0,
 )
 results[col] = stats
 
 return results
 
 def compute_categorical_stats(
 self, df: pd.DataFrame, columns: Optional[List[str]] = None
 ) -> Dict[str, CategoricalStatistics]:
 """
 Compute statistics for categorical columns.
 
 Args:
 df: Input DataFrame
 columns: Specific columns to analyze (None = all categorical)
 
 Returns:
 Dictionary of column name to CategoricalStatistics
 """
 if columns is None:
 cat_df = df.select_dtypes(include=["object", "category"])
 columns = cat_df.columns.tolist()
 
 results = {}
 
 for col in columns:
 if col not in df.columns:
 continue
 
 series = df[col]
 value_counts = series.value_counts()
 
 if len(value_counts) == 0:
 continue
 
 mode = value_counts.index[0]
 mode_freq = int(value_counts.iloc[0])
 total_valid = series.notna().sum()
 
 stats = CategoricalStatistics(
 name=col,
 count=int(total_valid),
 missing=int(series.isna().sum()),
 unique_count=int(series.nunique()),
 mode=str(mode),
 mode_frequency=mode_freq,
 mode_percent=(mode_freq / total_valid * 100) if total_valid > 0 else 0,
 value_counts=value_counts.head(20).to_dict(),
 )
 results[col] = stats
 
 return results
 
 def compute_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
 """
 Create a summary statistics table.
 
 Args:
 df: Input DataFrame
 
 Returns:
 Summary DataFrame with key statistics
 """
 numeric_cols = df.select_dtypes(include=[np.number]).columns
 
 if len(numeric_cols) == 0:
 return pd.DataFrame()
 
 # Use pandas describe and enhance
 summary = df[numeric_cols].describe().T
 
 # Add additional statistics
 summary["missing"] = df[numeric_cols].isna().sum()
 summary["missing_pct"] = (summary["missing"] / len(df) * 100).round(2)
 summary["skew"] = df[numeric_cols].skew().round(3)
 summary["kurtosis"] = df[numeric_cols].kurtosis().round(3)
 
 # Reorder columns
 col_order = ["count", "missing", "missing_pct", "mean", "std", 
 "min", "25%", "50%", "75%", "max", "skew", "kurtosis"]
 summary = summary[[c for c in col_order if c in summary.columns]]
 
 return summary.round(3)
 
 def get_distribution_insights(
 self, stats: Dict[str, ColumnStatistics]
 ) -> List[str]:
 """
 Generate insights about distributions.
 
 Args:
 stats: Dictionary of column statistics
 
 Returns:
 List of insight strings
 """
 insights = []
 
 for name, stat in stats.items():
 # Skewness insights
 if stat.skewness > 1:
 insights.append(
 f"'{name}' is highly right-skewed (skew={stat.skewness:.2f}), "
 "indicating presence of large outliers or long tail on the right"
 )
 elif stat.skewness < -1:
 insights.append(
 f"'{name}' is highly left-skewed (skew={stat.skewness:.2f}), "
 "indicating concentration at higher values"
 )
 
 # Range insights
 if stat.std > 0 and stat.std / abs(stat.mean) > 1 if stat.mean != 0 else False:
 insights.append(
 f"'{name}' has high variability (CV={stat.std/abs(stat.mean):.2f})"
 )
 
 # Zero-heavy detection
 if stat.min == 0 and stat.median == 0:
 insights.append(
 f"'{name}' has many zero values (median=0)"
 )
 
 return insights
