"""
Chart Selector Module

Automatically selects appropriate chart types based on:
- Data characteristics
- Question context
- Number of dimensions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChartType(str, Enum):
 """Supported chart types."""
 
 BAR = "bar"
 LINE = "line"
 SCATTER = "scatter"
 PIE = "pie"
 HISTOGRAM = "histogram"
 BOX = "box"
 HEATMAP = "heatmap"
 AREA = "area"
 TREEMAP = "treemap"
 TABLE = "table"


@dataclass
class ChartRecommendation:
 """Recommended chart configuration."""
 
 chart_type: ChartType
 x_column: Optional[str]
 y_column: Optional[str]
 color_column: Optional[str] = None
 title: str = ""
 confidence: float = 0.8
 reasoning: str = ""
 
 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 "chart_type": self.chart_type.value,
 "x": self.x_column,
 "y": self.y_column,
 "color": self.color_column,
 "title": self.title,
 "confidence": self.confidence,
 "reasoning": self.reasoning,
 }


class ChartSelector:
 """
 Selects appropriate chart types based on data and context.
 
 Selection logic:
 - Time series data → Line chart
 - Category + Numeric → Bar chart
 - Two numerics → Scatter plot
 - Parts of whole → Pie chart
 - Distribution → Histogram
 - Comparison across groups → Box plot
 """
 
 # Keywords that suggest specific chart types
 CHART_KEYWORDS = {
 ChartType.LINE: ["trend", "over time", "timeline", "time series", "monthly", "yearly", "daily"],
 ChartType.BAR: ["compare", "comparison", "by", "per", "across", "breakdown"],
 ChartType.PIE: ["proportion", "share", "percentage", "distribution of", "composition"],
 ChartType.SCATTER: ["relationship", "correlation", "vs", "versus", "against"],
 ChartType.HISTOGRAM: ["distribution", "spread", "frequency", "histogram"],
 ChartType.BOX: ["range", "variability", "outliers", "quartiles"],
 }
 
 def __init__(self):
 """Initialize the chart selector."""
 pass
 
 def select_chart(
 self,
 data: pd.DataFrame,
 question: Optional[str] = None,
 ) -> ChartRecommendation:
 """
 Select the best chart type for the data.
 
 Args:
 data: DataFrame to visualize
 question: Original question (for context)
 
 Returns:
 ChartRecommendation with configuration
 """
 if data is None or len(data) == 0:
 return ChartRecommendation(
 chart_type=ChartType.TABLE,
 x_column=None,
 y_column=None,
 title="No Data",
 reasoning="Empty dataset",
 )
 
 # Analyze data structure
 n_rows, n_cols = data.shape
 numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
 categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
 datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
 
 # Check for keywords in question
 if question:
 keyword_chart = self._match_keywords(question)
 if keyword_chart:
 return self._configure_by_type(
 keyword_chart, data, numeric_cols, categorical_cols, datetime_cols
 )
 
 # Decision tree based on data structure
 return self._select_by_structure(
 data, numeric_cols, categorical_cols, datetime_cols
 )
 
 def _match_keywords(self, question: str) -> Optional[ChartType]:
 """
 Match question to chart type via keywords.
 
 Args:
 question: User question
 
 Returns:
 ChartType if match found, else None
 """
 question_lower = question.lower()
 
 for chart_type, keywords in self.CHART_KEYWORDS.items():
 for keyword in keywords:
 if keyword in question_lower:
 return chart_type
 
 return None
 
 def _select_by_structure(
 self,
 data: pd.DataFrame,
 numeric_cols: list,
 categorical_cols: list,
 datetime_cols: list,
 ) -> ChartRecommendation:
 """
 Select chart based on data structure.
 
 Args:
 data: DataFrame
 numeric_cols: List of numeric columns
 categorical_cols: List of categorical columns
 datetime_cols: List of datetime columns
 
 Returns:
 ChartRecommendation
 """
 n_rows = len(data)
 n_numeric = len(numeric_cols)
 n_categorical = len(categorical_cols)
 n_datetime = len(datetime_cols)
 
 # Time series: datetime + numeric
 if n_datetime > 0 and n_numeric > 0:
 return ChartRecommendation(
 chart_type=ChartType.LINE,
 x_column=datetime_cols[0],
 y_column=numeric_cols[0],
 title=f"{numeric_cols[0]} Over Time",
 confidence=0.9,
 reasoning="Time series data detected",
 )
 
 # Single numeric column: histogram
 if n_numeric == 1 and n_categorical == 0:
 return ChartRecommendation(
 chart_type=ChartType.HISTOGRAM,
 x_column=numeric_cols[0],
 y_column=None,
 title=f"Distribution of {numeric_cols[0]}",
 confidence=0.85,
 reasoning="Single numeric column - distribution chart",
 )
 
 # Two numeric columns: scatter
 if n_numeric >= 2 and n_categorical == 0:
 return ChartRecommendation(
 chart_type=ChartType.SCATTER,
 x_column=numeric_cols[0],
 y_column=numeric_cols[1],
 title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
 confidence=0.85,
 reasoning="Two numeric columns - relationship chart",
 )
 
 # Category + Numeric: bar chart
 if n_categorical >= 1 and n_numeric >= 1:
 # Use pie for small number of categories showing totals
 if n_rows <= 8:
 return ChartRecommendation(
 chart_type=ChartType.PIE,
 x_column=categorical_cols[0],
 y_column=numeric_cols[0],
 title=f"{numeric_cols[0]} by {categorical_cols[0]}",
 confidence=0.8,
 reasoning="Small number of categories - pie chart",
 )
 else:
 return ChartRecommendation(
 chart_type=ChartType.BAR,
 x_column=categorical_cols[0],
 y_column=numeric_cols[0],
 color_column=categorical_cols[1] if n_categorical > 1 else None,
 title=f"{numeric_cols[0]} by {categorical_cols[0]}",
 confidence=0.9,
 reasoning="Categorical + numeric - bar chart",
 )
 
 # Fallback to table
 return ChartRecommendation(
 chart_type=ChartType.TABLE,
 x_column=None,
 y_column=None,
 title="Data Table",
 confidence=0.5,
 reasoning="No clear visualization pattern - showing table",
 )
 
 def _configure_by_type(
 self,
 chart_type: ChartType,
 data: pd.DataFrame,
 numeric_cols: list,
 categorical_cols: list,
 datetime_cols: list,
 ) -> ChartRecommendation:
 """
 Configure columns for a specific chart type.
 
 Args:
 chart_type: Desired chart type
 data: DataFrame
 numeric_cols: Numeric columns
 categorical_cols: Categorical columns
 datetime_cols: Datetime columns
 
 Returns:
 ChartRecommendation
 """
 x_col = None
 y_col = None
 color_col = None
 title = ""
 
 if chart_type == ChartType.LINE:
 if datetime_cols:
 x_col = datetime_cols[0]
 elif categorical_cols:
 x_col = categorical_cols[0]
 y_col = numeric_cols[0] if numeric_cols else None
 title = f"{y_col} Trend" if y_col else "Trend"
 
 elif chart_type == ChartType.BAR:
 x_col = categorical_cols[0] if categorical_cols else None
 y_col = numeric_cols[0] if numeric_cols else None
 title = f"{y_col} by {x_col}" if x_col and y_col else "Bar Chart"
 
 elif chart_type == ChartType.SCATTER:
 if len(numeric_cols) >= 2:
 x_col = numeric_cols[0]
 y_col = numeric_cols[1]
 title = f"{y_col} vs {x_col}"
 else:
 x_col = data.columns[0]
 y_col = data.columns[1] if len(data.columns) > 1 else None
 title = "Scatter Plot"
 
 elif chart_type == ChartType.PIE:
 x_col = categorical_cols[0] if categorical_cols else data.index.name
 y_col = numeric_cols[0] if numeric_cols else None
 title = f"{y_col} Distribution" if y_col else "Distribution"
 
 elif chart_type == ChartType.HISTOGRAM:
 x_col = numeric_cols[0] if numeric_cols else data.columns[0]
 title = f"Distribution of {x_col}"
 
 elif chart_type == ChartType.BOX:
 y_col = numeric_cols[0] if numeric_cols else None
 x_col = categorical_cols[0] if categorical_cols else None
 title = f"{y_col} Distribution" if y_col else "Box Plot"
 
 return ChartRecommendation(
 chart_type=chart_type,
 x_column=x_col,
 y_column=y_col,
 color_column=color_col,
 title=title,
 confidence=0.85,
 reasoning=f"Chart type matched from question context",
 )
 
 def get_alternative_charts(
 self,
 data: pd.DataFrame,
 primary: ChartRecommendation,
 ) -> list[ChartRecommendation]:
 """
 Get alternative chart recommendations.
 
 Args:
 data: DataFrame
 primary: Primary recommendation
 
 Returns:
 List of alternative recommendations
 """
 alternatives = []
 numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
 categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
 
 # Bar ↔ Line interchange
 if primary.chart_type == ChartType.BAR:
 alternatives.append(ChartRecommendation(
 chart_type=ChartType.LINE,
 x_column=primary.x_column,
 y_column=primary.y_column,
 title=primary.title.replace("by", "trend"),
 confidence=0.7,
 reasoning="Alternative: Line chart for trend view",
 ))
 
 # Pie ↔ Bar interchange
 if primary.chart_type == ChartType.PIE:
 alternatives.append(ChartRecommendation(
 chart_type=ChartType.BAR,
 x_column=primary.x_column,
 y_column=primary.y_column,
 title=primary.title,
 confidence=0.75,
 reasoning="Alternative: Bar chart for easier comparison",
 ))
 
 # Always offer table as fallback
 alternatives.append(ChartRecommendation(
 chart_type=ChartType.TABLE,
 x_column=None,
 y_column=None,
 title="Data Table",
 confidence=0.5,
 reasoning="Alternative: View raw data",
 ))
 
 return alternatives[:3]
