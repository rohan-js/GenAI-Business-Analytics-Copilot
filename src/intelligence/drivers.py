"""
Driver Analysis Module

Identifies key drivers of business metrics:
- Feature importance estimation
- Correlation-based analysis
- Segment comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Driver:
 """A single identified driver."""
 
 name: str
 importance_score: float # 0-1 scale
 direction: str # 'positive', 'negative', 'mixed'
 evidence: str
 driver_type: str = "correlation" # 'correlation', 'segment', 'trend'
 
 @property
 def importance_level(self) -> str:
 """Classify importance level."""
 if self.importance_score >= 0.7:
 return "high"
 elif self.importance_score >= 0.4:
 return "medium"
 else:
 return "low"


@dataclass
class DriverAnalysisResult:
 """Results of driver analysis."""
 
 target_metric: str
 drivers: List[Driver] = field(default_factory=list)
 summary: str = ""
 
 @property
 def top_drivers(self) -> List[Driver]:
 """Get top 3 drivers by importance."""
 sorted_drivers = sorted(
 self.drivers,
 key=lambda d: d.importance_score,
 reverse=True
 )
 return sorted_drivers[:3]
 
 def get_positive_drivers(self) -> List[Driver]:
 """Get drivers with positive influence."""
 return [d for d in self.drivers if d.direction == "positive"]
 
 def get_negative_drivers(self) -> List[Driver]:
 """Get drivers with negative influence."""
 return [d for d in self.drivers if d.direction == "negative"]


class DriverAnalyzer:
 """
 Analyzes key drivers of business metrics.
 
 Uses multiple approaches:
 - Correlation analysis
 - Segment comparison
 - Variance decomposition
 """
 
 def __init__(self):
 """Initialize the driver analyzer."""
 pass
 
 def analyze_drivers(
 self,
 df: pd.DataFrame,
 target_column: str,
 feature_columns: Optional[List[str]] = None,
 ) -> DriverAnalysisResult:
 """
 Identify key drivers of a target metric.
 
 Args:
 df: DataFrame with data
 target_column: Name of target metric
 feature_columns: Columns to analyze as potential drivers
 
 Returns:
 DriverAnalysisResult with identified drivers
 """
 if target_column not in df.columns:
 return DriverAnalysisResult(
 target_metric=target_column,
 summary=f"Target column '{target_column}' not found in data",
 )
 
 drivers = []
 
 # Determine feature columns
 if feature_columns is None:
 feature_columns = [
 col for col in df.columns
 if col != target_column
 ]
 
 # Analyze numeric drivers (correlation)
 numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns
 for col in numeric_cols:
 driver = self._analyze_numeric_driver(df, target_column, col)
 if driver:
 drivers.append(driver)
 
 # Analyze categorical drivers (segment comparison)
 categorical_cols = df[feature_columns].select_dtypes(
 include=["object", "category"]
 ).columns
 
 for col in categorical_cols:
 driver = self._analyze_categorical_driver(df, target_column, col)
 if driver:
 drivers.append(driver)
 
 # Sort by importance
 drivers.sort(key=lambda d: d.importance_score, reverse=True)
 
 # Generate summary
 summary = self._generate_summary(target_column, drivers)
 
 return DriverAnalysisResult(
 target_metric=target_column,
 drivers=drivers,
 summary=summary,
 )
 
 def _analyze_numeric_driver(
 self,
 df: pd.DataFrame,
 target: str,
 feature: str,
 ) -> Optional[Driver]:
 """
 Analyze a numeric feature as potential driver.
 
 Args:
 df: DataFrame
 target: Target column name
 feature: Feature column name
 
 Returns:
 Driver object or None
 """
 try:
 # Calculate correlation
 clean_df = df[[target, feature]].dropna()
 
 if len(clean_df) < 10:
 return None
 
 correlation = clean_df[target].corr(clean_df[feature])
 
 if pd.isna(correlation) or abs(correlation) < 0.1:
 return None
 
 # Determine direction and importance
 direction = "positive" if correlation > 0 else "negative"
 importance = abs(correlation)
 
 # Generate evidence
 evidence = (
 f"Correlation with {target}: {correlation:.3f}. "
 f"When {feature} increases, {target} tends to "
 f"{'increase' if correlation > 0 else 'decrease'}."
 )
 
 return Driver(
 name=feature,
 importance_score=importance,
 direction=direction,
 evidence=evidence,
 driver_type="correlation",
 )
 
 except Exception:
 return None
 
 def _analyze_categorical_driver(
 self,
 df: pd.DataFrame,
 target: str,
 feature: str,
 ) -> Optional[Driver]:
 """
 Analyze a categorical feature as potential driver.
 
 Args:
 df: DataFrame
 target: Target column name
 feature: Feature column name
 
 Returns:
 Driver object or None
 """
 try:
 # Check if target is numeric
 if not pd.api.types.is_numeric_dtype(df[target]):
 return None
 
 # Check cardinality
 unique_values = df[feature].nunique()
 if unique_values < 2 or unique_values > 50:
 return None
 
 # Calculate segment means
 segment_means = df.groupby(feature)[target].mean()
 overall_mean = df[target].mean()
 
 if len(segment_means) < 2:
 return None
 
 # Calculate variation explained
 segment_counts = df.groupby(feature)[target].count()
 weighted_var = sum(
 (segment_means[seg] - overall_mean) ** 2 * segment_counts[seg]
 for seg in segment_means.index
 ) / len(df)
 
 total_var = df[target].var()
 
 if total_var == 0:
 return None
 
 importance = min(weighted_var / total_var, 1.0)
 
 if importance < 0.05:
 return None
 
 # Find best and worst segments
 best_segment = segment_means.idxmax()
 worst_segment = segment_means.idxmin()
 
 # Determine direction
 if segment_means[best_segment] > overall_mean * 1.1:
 direction = "positive"
 elif segment_means[worst_segment] < overall_mean * 0.9:
 direction = "negative"
 else:
 direction = "mixed"
 
 evidence = (
 f"Segments by {feature} show different {target} levels. "
 f"Best: '{best_segment}' (avg: {segment_means[best_segment]:.2f}), "
 f"Worst: '{worst_segment}' (avg: {segment_means[worst_segment]:.2f})"
 )
 
 return Driver(
 name=feature,
 importance_score=importance,
 direction=direction,
 evidence=evidence,
 driver_type="segment",
 )
 
 except Exception:
 return None
 
 def _generate_summary(
 self,
 target: str,
 drivers: List[Driver],
 ) -> str:
 """
 Generate summary of driver analysis.
 
 Args:
 target: Target metric name
 drivers: List of identified drivers
 
 Returns:
 Summary string
 """
 if not drivers:
 return f"No significant drivers identified for {target}."
 
 top_3 = drivers[:3]
 names = [d.name for d in top_3]
 
 summary = f"Key drivers of {target}: {', '.join(names)}. "
 
 if top_3[0].direction == "positive":
 summary += f"Increasing {top_3[0].name} is associated with higher {target}."
 else:
 summary += f"Higher {top_3[0].name} is associated with lower {target}."
 
 return summary
 
 def compare_segments(
 self,
 df: pd.DataFrame,
 segment_column: str,
 metric_column: str,
 ) -> pd.DataFrame:
 """
 Compare a metric across segments.
 
 Args:
 df: DataFrame
 segment_column: Column to segment by
 metric_column: Metric to compare
 
 Returns:
 Comparison DataFrame
 """
 if segment_column not in df.columns or metric_column not in df.columns:
 return pd.DataFrame()
 
 comparison = df.groupby(segment_column).agg({
 metric_column: ["count", "sum", "mean", "std", "min", "max"]
 }).round(2)
 
 comparison.columns = ["Count", "Sum", "Mean", "Std", "Min", "Max"]
 
 # Add percent of total
 total = comparison["Sum"].sum()
 if total > 0:
 comparison["% of Total"] = (comparison["Sum"] / total * 100).round(2)
 
 # Add vs average
 overall_mean = comparison["Mean"].mean()
 comparison["vs Avg"] = ((comparison["Mean"] - overall_mean) / overall_mean * 100).round(1)
 
 return comparison.sort_values("Sum", ascending=False)
 
 def find_anomalous_segments(
 self,
 df: pd.DataFrame,
 segment_column: str,
 metric_column: str,
 threshold: float = 1.5,
 ) -> List[Tuple[str, float, str]]:
 """
 Find segments with anomalous metric values.
 
 Args:
 df: DataFrame
 segment_column: Column to segment by
 metric_column: Metric to analyze
 threshold: Standard deviations from mean to flag
 
 Returns:
 List of (segment, value, direction) tuples
 """
 if segment_column not in df.columns or metric_column not in df.columns:
 return []
 
 segment_means = df.groupby(segment_column)[metric_column].mean()
 overall_mean = segment_means.mean()
 overall_std = segment_means.std()
 
 if overall_std == 0:
 return []
 
 anomalies = []
 
 for segment, value in segment_means.items():
 z_score = (value - overall_mean) / overall_std
 
 if abs(z_score) > threshold:
 direction = "above" if z_score > 0 else "below"
 anomalies.append((str(segment), float(value), direction))
 
 return sorted(anomalies, key=lambda x: abs(x[1] - overall_mean), reverse=True)
