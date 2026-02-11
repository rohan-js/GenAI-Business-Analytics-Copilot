"""
Outlier Detector Module

Detects outliers using multiple methods:
- IQR (Interquartile Range)
- Z-Score
- Modified Z-Score (MAD-based)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from ..config import OUTLIER_IQR_MULTIPLIER, OUTLIER_ZSCORE_THRESHOLD


@dataclass
class OutlierResult:
 """Results of outlier detection for a column."""
 
 column: str
 method: str
 outlier_count: int
 outlier_percent: float
 lower_bound: Optional[float]
 upper_bound: Optional[float]
 outlier_indices: List[int] = field(default_factory=list)
 outlier_values: List[float] = field(default_factory=list)
 
 @property
 def has_outliers(self) -> bool:
 """Check if any outliers detected."""
 return self.outlier_count > 0
 
 @property
 def severity(self) -> str:
 """Classify outlier severity."""
 if self.outlier_percent >= 10:
 return "high"
 elif self.outlier_percent >= 5:
 return "moderate"
 elif self.outlier_percent > 0:
 return "low"
 else:
 return "none"


class OutlierDetector:
 """
 Detects outliers in numeric columns using various methods.
 
 Supports:
 - IQR method (robust to non-normal distributions)
 - Z-score method (assumes normal distribution)
 - Modified Z-score (MAD-based, very robust)
 """
 
 def __init__(
 self,
 iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER,
 zscore_threshold: float = OUTLIER_ZSCORE_THRESHOLD,
 ):
 """
 Initialize the outlier detector.
 
 Args:
 iqr_multiplier: Multiplier for IQR method (typically 1.5)
 zscore_threshold: Threshold for Z-score method (typically 3)
 """
 self.iqr_multiplier = iqr_multiplier
 self.zscore_threshold = zscore_threshold
 
 def detect_iqr(
 self, series: pd.Series, return_bounds: bool = True
 ) -> OutlierResult:
 """
 Detect outliers using IQR method.
 
 Outliers are values below Q1 - k*IQR or above Q3 + k*IQR
 
 Args:
 series: Numeric series to analyze
 return_bounds: Whether to compute bounds
 
 Returns:
 OutlierResult with detection results
 """
 clean_series = series.dropna()
 
 if len(clean_series) == 0:
 return OutlierResult(
 column=str(series.name),
 method="iqr",
 outlier_count=0,
 outlier_percent=0.0,
 lower_bound=None,
 upper_bound=None,
 )
 
 q1 = clean_series.quantile(0.25)
 q3 = clean_series.quantile(0.75)
 iqr = q3 - q1
 
 lower_bound = q1 - self.iqr_multiplier * iqr
 upper_bound = q3 + self.iqr_multiplier * iqr
 
 # Find outliers
 outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
 outlier_indices = clean_series[outlier_mask].index.tolist()
 outlier_values = clean_series[outlier_mask].tolist()
 outlier_count = len(outlier_indices)
 outlier_percent = (outlier_count / len(clean_series) * 100)
 
 return OutlierResult(
 column=str(series.name),
 method="iqr",
 outlier_count=outlier_count,
 outlier_percent=outlier_percent,
 lower_bound=float(lower_bound),
 upper_bound=float(upper_bound),
 outlier_indices=outlier_indices,
 outlier_values=outlier_values[:100], # Limit stored values
 )
 
 def detect_zscore(self, series: pd.Series) -> OutlierResult:
 """
 Detect outliers using Z-score method.
 
 Outliers are values with |Z| > threshold
 
 Args:
 series: Numeric series to analyze
 
 Returns:
 OutlierResult with detection results
 """
 clean_series = series.dropna()
 
 if len(clean_series) < 2:
 return OutlierResult(
 column=str(series.name),
 method="zscore",
 outlier_count=0,
 outlier_percent=0.0,
 lower_bound=None,
 upper_bound=None,
 )
 
 mean = clean_series.mean()
 std = clean_series.std()
 
 if std == 0:
 return OutlierResult(
 column=str(series.name),
 method="zscore",
 outlier_count=0,
 outlier_percent=0.0,
 lower_bound=mean,
 upper_bound=mean,
 )
 
 z_scores = np.abs((clean_series - mean) / std)
 
 # Find outliers
 outlier_mask = z_scores > self.zscore_threshold
 outlier_indices = clean_series[outlier_mask].index.tolist()
 outlier_values = clean_series[outlier_mask].tolist()
 outlier_count = len(outlier_indices)
 outlier_percent = (outlier_count / len(clean_series) * 100)
 
 lower_bound = mean - self.zscore_threshold * std
 upper_bound = mean + self.zscore_threshold * std
 
 return OutlierResult(
 column=str(series.name),
 method="zscore",
 outlier_count=outlier_count,
 outlier_percent=outlier_percent,
 lower_bound=float(lower_bound),
 upper_bound=float(upper_bound),
 outlier_indices=outlier_indices,
 outlier_values=outlier_values[:100],
 )
 
 def detect_modified_zscore(self, series: pd.Series) -> OutlierResult:
 """
 Detect outliers using Modified Z-score (MAD-based).
 
 More robust than standard Z-score for skewed distributions.
 
 Args:
 series: Numeric series to analyze
 
 Returns:
 OutlierResult with detection results
 """
 clean_series = series.dropna()
 
 if len(clean_series) < 2:
 return OutlierResult(
 column=str(series.name),
 method="modified_zscore",
 outlier_count=0,
 outlier_percent=0.0,
 lower_bound=None,
 upper_bound=None,
 )
 
 median = clean_series.median()
 mad = np.median(np.abs(clean_series - median))
 
 # Modified Z-score constant (0.6745 relates MAD to standard deviation)
 if mad == 0:
 return OutlierResult(
 column=str(series.name),
 method="modified_zscore",
 outlier_count=0,
 outlier_percent=0.0,
 lower_bound=median,
 upper_bound=median,
 )
 
 modified_z = 0.6745 * (clean_series - median) / mad
 
 # Typically use threshold of 3.5 for modified Z-score
 threshold = 3.5
 outlier_mask = np.abs(modified_z) > threshold
 outlier_indices = clean_series[outlier_mask].index.tolist()
 outlier_values = clean_series[outlier_mask].tolist()
 outlier_count = len(outlier_indices)
 outlier_percent = (outlier_count / len(clean_series) * 100)
 
 # Approximate bounds
 lower_bound = median - threshold * mad / 0.6745
 upper_bound = median + threshold * mad / 0.6745
 
 return OutlierResult(
 column=str(series.name),
 method="modified_zscore",
 outlier_count=outlier_count,
 outlier_percent=outlier_percent,
 lower_bound=float(lower_bound),
 upper_bound=float(upper_bound),
 outlier_indices=outlier_indices,
 outlier_values=outlier_values[:100],
 )
 
 def detect_all(
 self, df: pd.DataFrame, columns: Optional[List[str]] = None
 ) -> Dict[str, OutlierResult]:
 """
 Detect outliers in all numeric columns using IQR method.
 
 Args:
 df: Input DataFrame
 columns: Specific columns to analyze (None = all numeric)
 
 Returns:
 Dictionary of column name to OutlierResult
 """
 if columns is None:
 numeric_df = df.select_dtypes(include=[np.number])
 columns = numeric_df.columns.tolist()
 
 results = {}
 
 for col in columns:
 if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
 results[col] = self.detect_iqr(df[col])
 
 return results
 
 def get_outlier_summary(
 self, results: Dict[str, OutlierResult]
 ) -> pd.DataFrame:
 """
 Create summary DataFrame of outlier detection results.
 
 Args:
 results: Dictionary of OutlierResult objects
 
 Returns:
 Summary DataFrame
 """
 rows = []
 
 for col, result in results.items():
 rows.append({
 "Column": col,
 "Method": result.method,
 "Outliers": result.outlier_count,
 "Percent": f"{result.outlier_percent:.2f}%",
 "Lower Bound": f"{result.lower_bound:.2f}" if result.lower_bound else "N/A",
 "Upper Bound": f"{result.upper_bound:.2f}" if result.upper_bound else "N/A",
 "Severity": result.severity,
 })
 
 return pd.DataFrame(rows)
 
 def get_outlier_insights(
 self, results: Dict[str, OutlierResult]
 ) -> List[str]:
 """
 Generate insights from outlier analysis.
 
 Args:
 results: Dictionary of OutlierResult objects
 
 Returns:
 List of insight strings
 """
 insights = []
 
 # Columns with high outliers
 high_outlier_cols = [
 r for r in results.values() if r.severity == "high"
 ]
 
 moderate_outlier_cols = [
 r for r in results.values() if r.severity == "moderate"
 ]
 
 if high_outlier_cols:
 names = [r.column for r in high_outlier_cols]
 insights.append(
 f" High outlier presence in: {', '.join(names)}. "
 "Consider investigating data quality or business reasons."
 )
 
 if moderate_outlier_cols:
 names = [r.column for r in moderate_outlier_cols]
 insights.append(
 f"Moderate outliers detected in: {', '.join(names)}. "
 "These may represent legitimate extreme values."
 )
 
 # Specific extreme values
 for result in results.values():
 if result.outlier_values and result.outlier_percent > 2:
 extreme_val = max(result.outlier_values, key=abs)
 insights.append(
 f"'{result.column}' has extreme value {extreme_val:.2f} "
 f"(bound: {result.lower_bound:.2f} to {result.upper_bound:.2f})"
 )
 
 return insights[:5] # Limit to top 5 insights
