"""
Correlation Analyzer Module

Analyzes relationships between variables:
- Pearson correlation for numeric pairs
- Cramér's V for categorical pairs
- Point-biserial for numeric-categorical pairs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

from ..config import HIGH_CORRELATION_THRESHOLD


@dataclass
class CorrelationPair:
 """Represents a correlation between two columns."""
 
 column1: str
 column2: str
 correlation: float
 p_value: Optional[float]
 method: str # 'pearson', 'spearman', 'cramers_v', 'point_biserial'
 
 @property
 def is_significant(self) -> bool:
 """Check if correlation is statistically significant."""
 if self.p_value is None:
 return abs(self.correlation) >= HIGH_CORRELATION_THRESHOLD
 return self.p_value < 0.05
 
 @property
 def strength(self) -> str:
 """Interpret correlation strength."""
 r = abs(self.correlation)
 if r >= 0.8:
 return "very_strong"
 elif r >= 0.6:
 return "strong"
 elif r >= 0.4:
 return "moderate"
 elif r >= 0.2:
 return "weak"
 else:
 return "negligible"
 
 @property
 def direction(self) -> str:
 """Get correlation direction."""
 if self.correlation > 0:
 return "positive"
 elif self.correlation < 0:
 return "negative"
 else:
 return "none"


class CorrelationAnalyzer:
 """
 Analyzes correlations across different variable types.
 
 Supports:
 - Numeric-Numeric: Pearson/Spearman correlation
 - Categorical-Categorical: Cramér's V
 - Numeric-Categorical: ANOVA / Point-biserial
 """
 
 def __init__(self, threshold: float = HIGH_CORRELATION_THRESHOLD):
 """
 Initialize the correlation analyzer.
 
 Args:
 threshold: Threshold for "high" correlation flag
 """
 self.threshold = threshold
 
 def compute_correlation_matrix(
 self, df: pd.DataFrame, method: str = "pearson"
 ) -> pd.DataFrame:
 """
 Compute correlation matrix for numeric columns.
 
 Args:
 df: Input DataFrame
 method: 'pearson' or 'spearman'
 
 Returns:
 Correlation matrix DataFrame
 """
 numeric_df = df.select_dtypes(include=[np.number])
 
 if numeric_df.empty:
 return pd.DataFrame()
 
 if method == "spearman":
 return numeric_df.corr(method="spearman")
 else:
 return numeric_df.corr(method="pearson")
 
 def find_high_correlations(
 self, df: pd.DataFrame, threshold: Optional[float] = None
 ) -> List[CorrelationPair]:
 """
 Find pairs of columns with high correlation.
 
 Args:
 df: Input DataFrame
 threshold: Correlation threshold (uses default if None)
 
 Returns:
 List of CorrelationPair objects above threshold
 """
 if threshold is None:
 threshold = self.threshold
 
 corr_matrix = self.compute_correlation_matrix(df)
 
 if corr_matrix.empty:
 return []
 
 pairs = []
 columns = corr_matrix.columns.tolist()
 
 for i, col1 in enumerate(columns):
 for j, col2 in enumerate(columns):
 if i >= j: # Skip diagonal and lower triangle
 continue
 
 corr_val = corr_matrix.loc[col1, col2]
 
 if abs(corr_val) >= threshold:
 # Calculate p-value
 try:
 clean_data = df[[col1, col2]].dropna()
 if len(clean_data) > 2:
 _, p_value = stats.pearsonr(
 clean_data[col1], clean_data[col2]
 )
 else:
 p_value = None
 except Exception:
 p_value = None
 
 pairs.append(CorrelationPair(
 column1=col1,
 column2=col2,
 correlation=float(corr_val),
 p_value=float(p_value) if p_value is not None else None,
 method="pearson",
 ))
 
 # Sort by absolute correlation (descending)
 pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
 
 return pairs
 
 def compute_cramers_v(
 self, df: pd.DataFrame, col1: str, col2: str
 ) -> Tuple[float, float]:
 """
 Compute Cramér's V for two categorical columns.
 
 Args:
 df: Input DataFrame
 col1: First column name
 col2: Second column name
 
 Returns:
 Tuple of (cramers_v, chi2_pvalue)
 """
 contingency = pd.crosstab(df[col1], df[col2])
 chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
 
 n = contingency.sum().sum()
 min_dim = min(contingency.shape) - 1
 
 if min_dim == 0 or n == 0:
 return 0.0, p_value
 
 cramers_v = np.sqrt(chi2 / (n * min_dim))
 
 return float(cramers_v), float(p_value)
 
 def analyze_numeric_vs_categorical(
 self, df: pd.DataFrame, numeric_col: str, categorical_col: str
 ) -> Dict:
 """
 Analyze relationship between numeric and categorical column.
 
 Uses ANOVA to test if means differ across categories.
 
 Args:
 df: Input DataFrame
 numeric_col: Numeric column name
 categorical_col: Categorical column name
 
 Returns:
 Dictionary with analysis results
 """
 # Group by category
 groups = df.groupby(categorical_col)[numeric_col].apply(list)
 
 # Remove groups with insufficient data
 groups = {k: v for k, v in groups.items() if len(v) >= 2}
 
 if len(groups) < 2:
 return {
 "f_statistic": None,
 "p_value": None,
 "significant": False,
 "group_means": {},
 }
 
 # ANOVA test
 try:
 f_stat, p_value = stats.f_oneway(*groups.values())
 except Exception:
 f_stat, p_value = None, None
 
 # Group statistics
 group_means = df.groupby(categorical_col)[numeric_col].mean().to_dict()
 group_stds = df.groupby(categorical_col)[numeric_col].std().to_dict()
 
 return {
 "f_statistic": float(f_stat) if f_stat is not None else None,
 "p_value": float(p_value) if p_value is not None else None,
 "significant": p_value < 0.05 if p_value is not None else False,
 "group_means": group_means,
 "group_stds": group_stds,
 }
 
 def get_correlation_insights(
 self, pairs: List[CorrelationPair]
 ) -> List[str]:
 """
 Generate insights from correlation analysis.
 
 Args:
 pairs: List of CorrelationPair objects
 
 Returns:
 List of insight strings
 """
 insights = []
 
 for pair in pairs[:5]: # Top 5 correlations
 direction = "increases with" if pair.correlation > 0 else "decreases with"
 strength = pair.strength.replace("_", " ")
 
 insight = (
 f"'{pair.column1}' has a {strength} {pair.direction} correlation "
 f"with '{pair.column2}' (r={pair.correlation:.3f})"
 )
 
 if pair.is_significant and pair.p_value is not None:
 insight += f" — statistically significant (p={pair.p_value:.4f})"
 
 insights.append(insight)
 
 # Multicollinearity warning
 high_pairs = [p for p in pairs if abs(p.correlation) >= 0.9]
 if high_pairs:
 col_names = set()
 for p in high_pairs:
 col_names.add(p.column1)
 col_names.add(p.column2)
 insights.append(
 f" Multicollinearity detected: {', '.join(col_names)} "
 "are highly correlated — consider removing redundant variables"
 )
 
 return insights
