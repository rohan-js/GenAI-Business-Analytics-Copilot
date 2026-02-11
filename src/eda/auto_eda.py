"""
Auto-EDA Engine

Orchestrates automatic exploratory data analysis:
- Runs all statistical analyses
- Detects patterns and anomalies
- Generates human-readable insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from ..data.profiler import DataProfiler, DataProfile
from .statistics import StatisticsCalculator, ColumnStatistics, CategoricalStatistics
from .correlations import CorrelationAnalyzer, CorrelationPair
from .outliers import OutlierDetector, OutlierResult


@dataclass
class EDAResult:
 """Complete EDA results for a dataset."""
 
 # Data profile
 profile: DataProfile
 
 # Statistics
 numeric_stats: Dict[str, ColumnStatistics] = field(default_factory=dict)
 categorical_stats: Dict[str, CategoricalStatistics] = field(default_factory=dict)
 summary_table: Optional[pd.DataFrame] = None
 
 # Correlations
 correlation_matrix: Optional[pd.DataFrame] = None
 high_correlations: List[CorrelationPair] = field(default_factory=list)
 
 # Outliers
 outlier_results: Dict[str, OutlierResult] = field(default_factory=dict)
 outlier_summary: Optional[pd.DataFrame] = None
 
 # Generated insights
 insights: List[str] = field(default_factory=list)
 warnings: List[str] = field(default_factory=list)
 
 @property
 def quick_summary(self) -> Dict[str, Any]:
 """Get quick summary for display."""
 return {
 "rows": f"{self.profile.n_rows:,}",
 "columns": self.profile.n_columns,
 "numeric_columns": len(self.profile.numeric_columns),
 "categorical_columns": len(self.profile.categorical_columns),
 "missing_percent": f"{self.profile.overall_missing_percent:.1f}%",
 "quality_score": f"{self.profile.data_quality_score:.0f}/100",
 "high_correlations": len(self.high_correlations),
 "columns_with_outliers": sum(1 for r in self.outlier_results.values() if r.has_outliers),
 "total_insights": len(self.insights),
 }


class AutoEDAEngine:
 """
 Automatic Exploratory Data Analysis Engine.
 
 Performs comprehensive EDA including:
 - Data profiling and quality assessment
 - Summary statistics for all column types
 - Correlation analysis
 - Outlier detection
 - Pattern recognition
 - Insight generation
 """
 
 def __init__(self):
 """Initialize the Auto-EDA engine with component analyzers."""
 self.profiler = DataProfiler()
 self.stats_calculator = StatisticsCalculator()
 self.correlation_analyzer = CorrelationAnalyzer()
 self.outlier_detector = OutlierDetector()
 
 def analyze(self, df: pd.DataFrame) -> EDAResult:
 """
 Perform complete EDA on a DataFrame.
 
 Args:
 df: Input DataFrame to analyze
 
 Returns:
 EDAResult with all analysis results
 """
 insights = []
 warnings = []
 
 # Step 1: Profile the data
 profile = self.profiler.profile(df)
 
 # Add profile issues as warnings
 warnings.extend(profile.issues)
 
 # Step 2: Compute statistics
 numeric_stats = self.stats_calculator.compute_numeric_stats(df)
 categorical_stats = self.stats_calculator.compute_categorical_stats(df)
 summary_table = self.stats_calculator.compute_summary_table(df)
 
 # Add distribution insights
 dist_insights = self.stats_calculator.get_distribution_insights(numeric_stats)
 insights.extend(dist_insights)
 
 # Step 3: Correlation analysis
 correlation_matrix = self.correlation_analyzer.compute_correlation_matrix(df)
 high_correlations = self.correlation_analyzer.find_high_correlations(df)
 
 # Add correlation insights
 corr_insights = self.correlation_analyzer.get_correlation_insights(high_correlations)
 insights.extend(corr_insights)
 
 # Step 4: Outlier detection
 outlier_results = self.outlier_detector.detect_all(df)
 outlier_summary = self.outlier_detector.get_outlier_summary(outlier_results)
 
 # Add outlier insights
 outlier_insights = self.outlier_detector.get_outlier_insights(outlier_results)
 insights.extend(outlier_insights)
 
 # Step 5: Generate additional insights
 additional_insights = self._generate_additional_insights(
 df, profile, numeric_stats, categorical_stats
 )
 insights.extend(additional_insights)
 
 return EDAResult(
 profile=profile,
 numeric_stats=numeric_stats,
 categorical_stats=categorical_stats,
 summary_table=summary_table,
 correlation_matrix=correlation_matrix,
 high_correlations=high_correlations,
 outlier_results=outlier_results,
 outlier_summary=outlier_summary,
 insights=insights,
 warnings=warnings,
 )
 
 def _generate_additional_insights(
 self,
 df: pd.DataFrame,
 profile: DataProfile,
 numeric_stats: Dict[str, ColumnStatistics],
 categorical_stats: Dict[str, CategoricalStatistics],
 ) -> List[str]:
 """
 Generate additional insights from the analysis.
 
 Args:
 df: Input DataFrame
 profile: Data profile
 numeric_stats: Numeric column statistics
 categorical_stats: Categorical column statistics
 
 Returns:
 List of insight strings
 """
 insights = []
 
 # Dataset size insight
 if profile.n_rows < 100:
 insights.append(
 " Small dataset detected (<100 rows). "
 "Statistical conclusions may not be reliable."
 )
 elif profile.n_rows > 100000:
 insights.append(
 f"Large dataset ({profile.n_rows:,} rows). "
 "Analysis is based on sampled data for performance."
 )
 
 # Dominant categories
 for name, stats in categorical_stats.items():
 if stats.mode_percent > 80:
 insights.append(
 f"'{name}' is heavily imbalanced: "
 f"'{stats.mode}' appears in {stats.mode_percent:.1f}% of records"
 )
 
 # Potential ID columns
 for name, col_profile in profile.columns.items():
 if col_profile.unique_percent > 99 and col_profile.inferred_type != "text":
 insights.append(
 f"'{name}' appears to be an ID column "
 f"({col_profile.unique_percent:.1f}% unique values)"
 )
 
 # Date range detection
 for col in profile.datetime_columns:
 try:
 date_col = pd.to_datetime(df[col], errors="coerce")
 min_date = date_col.min()
 max_date = date_col.max()
 if pd.notna(min_date) and pd.notna(max_date):
 date_range = (max_date - min_date).days
 insights.append(
 f"'{col}' spans {date_range:,} days "
 f"({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"
 )
 except Exception:
 pass
 
 return insights
 
 def get_segment_analysis(
 self,
 df: pd.DataFrame,
 segment_col: str,
 metric_col: str,
 ) -> pd.DataFrame:
 """
 Perform segment-level analysis.
 
 Args:
 df: Input DataFrame
 segment_col: Column to segment by
 metric_col: Metric to analyze
 
 Returns:
 DataFrame with segment statistics
 """
 if segment_col not in df.columns or metric_col not in df.columns:
 return pd.DataFrame()
 
 segment_stats = df.groupby(segment_col).agg({
 metric_col: ["count", "sum", "mean", "std", "min", "max"]
 }).round(2)
 
 segment_stats.columns = ["Count", "Sum", "Mean", "Std", "Min", "Max"]
 segment_stats["Pct of Total"] = (
 segment_stats["Sum"] / segment_stats["Sum"].sum() * 100
 ).round(2)
 
 return segment_stats.sort_values("Sum", ascending=False)
 
 def get_time_analysis(
 self,
 df: pd.DataFrame,
 date_col: str,
 metric_col: str,
 freq: str = "M",
 ) -> pd.DataFrame:
 """
 Perform time-based analysis.
 
 Args:
 df: Input DataFrame
 date_col: Date column
 metric_col: Metric to analyze
 freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
 
 Returns:
 DataFrame with time series statistics
 """
 if date_col not in df.columns or metric_col not in df.columns:
 return pd.DataFrame()
 
 df_copy = df.copy()
 df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
 df_copy = df_copy.dropna(subset=[date_col])
 
 if len(df_copy) == 0:
 return pd.DataFrame()
 
 df_copy.set_index(date_col, inplace=True)
 
 time_stats = df_copy.resample(freq).agg({
 metric_col: ["count", "sum", "mean"]
 }).round(2)
 
 time_stats.columns = ["Count", "Sum", "Mean"]
 
 # Add period-over-period change
 time_stats["Change %"] = time_stats["Sum"].pct_change() * 100
 
 return time_stats.round(2)
 
 def generate_eda_report(self, result: EDAResult) -> str:
 """
 Generate a text-based EDA report.
 
 Args:
 result: EDAResult from analysis
 
 Returns:
 Formatted report string
 """
 lines = []
 lines.append("=" * 60)
 lines.append("EXPLORATORY DATA ANALYSIS REPORT")
 lines.append("=" * 60)
 lines.append("")
 
 # Dataset Overview
 lines.append(" DATASET OVERVIEW")
 lines.append("-" * 40)
 summary = result.quick_summary
 for key, value in summary.items():
 lines.append(f" {key.replace('_', ' ').title()}: {value}")
 lines.append("")
 
 # Data Quality
 lines.append(" DATA QUALITY")
 lines.append("-" * 40)
 lines.append(f" Quality Score: {result.profile.data_quality_score:.1f}/100")
 lines.append(f" Missing Data: {result.profile.overall_missing_percent:.1f}%")
 if result.warnings:
 lines.append(" Issues Detected:")
 for warning in result.warnings[:5]:
 lines.append(f" {warning}")
 lines.append("")
 
 # Key Insights
 lines.append(" KEY INSIGHTS")
 lines.append("-" * 40)
 if result.insights:
 for i, insight in enumerate(result.insights[:10], 1):
 lines.append(f" {i}. {insight}")
 else:
 lines.append(" No significant insights detected.")
 lines.append("")
 
 lines.append("=" * 60)
 
 return "\n".join(lines)
