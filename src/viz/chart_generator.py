"""
Chart Generator Module

Creates interactive visualizations using Plotly:
- Business-friendly styling
- Interactive tooltips
- Export capabilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Union
import io
import base64

from .chart_selector import ChartSelector, ChartRecommendation, ChartType
from ..config import (
 CHART_THEME,
 COLOR_PALETTE,
 DEFAULT_CHART_WIDTH,
 DEFAULT_CHART_HEIGHT,
)


class ChartGenerator:
 """
 Generates interactive charts using Plotly.
 
 Features:
 - Auto chart type selection
 - Business-friendly styling
 - Interactive tooltips
 - Export to PNG/HTML
 """
 
 def __init__(self):
 """Initialize the chart generator."""
 self.selector = ChartSelector()
 self.theme = CHART_THEME
 self.colors = COLOR_PALETTE
 self.default_width = DEFAULT_CHART_WIDTH
 self.default_height = DEFAULT_CHART_HEIGHT
 
 def create_chart(
 self,
 data: pd.DataFrame,
 chart_type: Optional[ChartType] = None,
 x: Optional[str] = None,
 y: Optional[str] = None,
 color: Optional[str] = None,
 title: Optional[str] = None,
 question: Optional[str] = None,
 ) -> go.Figure:
 """
 Create a chart from data.
 
 Args:
 data: DataFrame to visualize
 chart_type: Specific chart type (auto-selected if None)
 x: X-axis column
 y: Y-axis column
 color: Color grouping column
 title: Chart title
 question: Original question for context
 
 Returns:
 Plotly Figure object
 """
 if data is None or len(data) == 0:
 return self._create_empty_chart("No data to display")
 
 # Auto-select chart if not specified
 if chart_type is None:
 recommendation = self.selector.select_chart(data, question)
 chart_type = recommendation.chart_type
 x = x or recommendation.x_column
 y = y or recommendation.y_column
 color = color or recommendation.color_column
 title = title or recommendation.title
 
 # Create chart based on type
 fig = self._create_by_type(data, chart_type, x, y, color, title)
 
 # Apply styling
 fig = self._apply_styling(fig, title)
 
 return fig
 
 def create_auto_chart(
 self,
 data: pd.DataFrame,
 question: str = "",
 ) -> tuple[go.Figure, ChartRecommendation]:
 """
 Automatically create the best chart for the data.
 
 Args:
 data: DataFrame to visualize
 question: Original question
 
 Returns:
 Tuple of (Figure, Recommendation)
 """
 recommendation = self.selector.select_chart(data, question)
 
 fig = self.create_chart(
 data,
 chart_type=recommendation.chart_type,
 x=recommendation.x_column,
 y=recommendation.y_column,
 color=recommendation.color_column,
 title=recommendation.title,
 )
 
 return fig, recommendation
 
 def _create_by_type(
 self,
 data: pd.DataFrame,
 chart_type: ChartType,
 x: Optional[str],
 y: Optional[str],
 color: Optional[str],
 title: str,
 ) -> go.Figure:
 """
 Create chart based on type.
 
 Args:
 data: DataFrame
 chart_type: Chart type to create
 x: X column
 y: Y column
 color: Color column
 title: Title
 
 Returns:
 Plotly Figure
 """
 # Handle Series data (convert to DataFrame)
 if isinstance(data, pd.Series):
 data = data.reset_index()
 if x is None:
 x = data.columns[0]
 if y is None:
 y = data.columns[1]
 
 # Reset index if it's meaningful
 if data.index.name is not None and x is None:
 data = data.reset_index()
 x = data.columns[0]
 
 try:
 if chart_type == ChartType.BAR:
 return self._create_bar_chart(data, x, y, color, title)
 
 elif chart_type == ChartType.LINE:
 return self._create_line_chart(data, x, y, color, title)
 
 elif chart_type == ChartType.SCATTER:
 return self._create_scatter_chart(data, x, y, color, title)
 
 elif chart_type == ChartType.PIE:
 return self._create_pie_chart(data, x, y, title)
 
 elif chart_type == ChartType.HISTOGRAM:
 return self._create_histogram(data, x, title)
 
 elif chart_type == ChartType.BOX:
 return self._create_box_plot(data, x, y, title)
 
 elif chart_type == ChartType.HEATMAP:
 return self._create_heatmap(data, title)
 
 else:
 return self._create_table_chart(data, title)
 
 except Exception as e:
 return self._create_empty_chart(f"Chart error: {str(e)}")
 
 def _create_bar_chart(
 self,
 data: pd.DataFrame,
 x: str,
 y: str,
 color: Optional[str],
 title: str,
 ) -> go.Figure:
 """Create bar chart."""
 if x not in data.columns:
 x = data.columns[0]
 if y not in data.columns:
 y = data.columns[-1]
 
 fig = px.bar(
 data,
 x=x,
 y=y,
 color=color if color and color in data.columns else None,
 title=title,
 color_discrete_sequence=self.colors,
 )
 
 # Sort by value for better visualization
 fig.update_layout(xaxis={'categoryorder': 'total descending'})
 
 return fig
 
 def _create_line_chart(
 self,
 data: pd.DataFrame,
 x: str,
 y: str,
 color: Optional[str],
 title: str,
 ) -> go.Figure:
 """Create line chart."""
 if x not in data.columns:
 x = data.columns[0]
 if y not in data.columns:
 y = data.columns[-1]
 
 fig = px.line(
 data,
 x=x,
 y=y,
 color=color if color and color in data.columns else None,
 title=title,
 color_discrete_sequence=self.colors,
 markers=True,
 )
 
 return fig
 
 def _create_scatter_chart(
 self,
 data: pd.DataFrame,
 x: str,
 y: str,
 color: Optional[str],
 title: str,
 ) -> go.Figure:
 """Create scatter plot."""
 if x not in data.columns:
 x = data.columns[0]
 if y not in data.columns:
 y = data.columns[1] if len(data.columns) > 1 else data.columns[0]
 
 fig = px.scatter(
 data,
 x=x,
 y=y,
 color=color if color and color in data.columns else None,
 title=title,
 color_discrete_sequence=self.colors,
 trendline="ols" if len(data) > 10 else None,
 )
 
 return fig
 
 def _create_pie_chart(
 self,
 data: pd.DataFrame,
 names: str,
 values: str,
 title: str,
 ) -> go.Figure:
 """Create pie chart."""
 if names not in data.columns:
 names = data.columns[0]
 if values not in data.columns:
 values = data.columns[-1]
 
 fig = px.pie(
 data,
 names=names,
 values=values,
 title=title,
 color_discrete_sequence=self.colors,
 hole=0.3, # Donut style
 )
 
 fig.update_traces(textposition='inside', textinfo='percent+label')
 
 return fig
 
 def _create_histogram(
 self,
 data: pd.DataFrame,
 x: str,
 title: str,
 ) -> go.Figure:
 """Create histogram."""
 if x not in data.columns:
 x = data.select_dtypes(include=[np.number]).columns[0]
 
 fig = px.histogram(
 data,
 x=x,
 title=title,
 color_discrete_sequence=self.colors,
 nbins=30,
 )
 
 # Add mean line
 mean_val = data[x].mean()
 fig.add_vline(
 x=mean_val,
 line_dash="dash",
 line_color="red",
 annotation_text=f"Mean: {mean_val:.2f}",
 )
 
 return fig
 
 def _create_box_plot(
 self,
 data: pd.DataFrame,
 x: Optional[str],
 y: str,
 title: str,
 ) -> go.Figure:
 """Create box plot."""
 if y not in data.columns:
 y = data.select_dtypes(include=[np.number]).columns[0]
 
 fig = px.box(
 data,
 x=x if x and x in data.columns else None,
 y=y,
 title=title,
 color_discrete_sequence=self.colors,
 )
 
 return fig
 
 def _create_heatmap(
 self,
 data: pd.DataFrame,
 title: str,
 ) -> go.Figure:
 """Create heatmap (typically for correlation matrix)."""
 # If data is correlation matrix
 if data.shape[0] == data.shape[1] and data.index.equals(data.columns):
 fig = px.imshow(
 data,
 title=title,
 color_continuous_scale="RdBu",
 aspect="auto",
 )
 else:
 # Generic heatmap
 fig = px.imshow(
 data,
 title=title,
 color_continuous_scale="Viridis",
 )
 
 return fig
 
 def _create_table_chart(
 self,
 data: pd.DataFrame,
 title: str,
 ) -> go.Figure:
 """Create table visualization."""
 fig = go.Figure(data=[go.Table(
 header=dict(
 values=list(data.columns),
 fill_color=self.colors[0],
 font=dict(color='white', size=12),
 align='left',
 ),
 cells=dict(
 values=[data[col] for col in data.columns],
 fill_color='white',
 align='left',
 ),
 )])
 
 fig.update_layout(title=title)
 
 return fig
 
 def _create_empty_chart(self, message: str) -> go.Figure:
 """Create placeholder for empty/error state."""
 fig = go.Figure()
 
 fig.add_annotation(
 text=message,
 xref="paper",
 yref="paper",
 x=0.5,
 y=0.5,
 showarrow=False,
 font=dict(size=16, color="gray"),
 )
 
 fig.update_layout(
 xaxis=dict(visible=False),
 yaxis=dict(visible=False),
 )
 
 return fig
 
 def _apply_styling(self, fig: go.Figure, title: str) -> go.Figure:
 """Apply consistent styling to chart."""
 fig.update_layout(
 title=dict(
 text=title,
 font=dict(size=18),
 ),
 template=self.theme,
 width=self.default_width,
 height=self.default_height,
 margin=dict(l=40, r=40, t=60, b=40),
 legend=dict(
 orientation="h",
 yanchor="bottom",
 y=1.02,
 xanchor="right",
 x=1,
 ),
 font=dict(family="Inter, sans-serif"),
 )
 
 return fig
 
 def create_correlation_heatmap(
 self,
 data: pd.DataFrame,
 title: str = "Correlation Matrix",
 ) -> go.Figure:
 """
 Create correlation heatmap for numeric columns.
 
 Args:
 data: DataFrame
 title: Chart title
 
 Returns:
 Plotly Figure
 """
 numeric_data = data.select_dtypes(include=[np.number])
 
 if numeric_data.empty:
 return self._create_empty_chart("No numeric columns for correlation")
 
 corr_matrix = numeric_data.corr()
 
 fig = go.Figure(data=go.Heatmap(
 z=corr_matrix.values,
 x=corr_matrix.columns,
 y=corr_matrix.columns,
 colorscale="RdBu",
 zmin=-1,
 zmax=1,
 text=corr_matrix.round(2).values,
 texttemplate="%{text}",
 textfont=dict(size=10),
 hoverongaps=False,
 ))
 
 fig = self._apply_styling(fig, title)
 fig.update_layout(height=500)
 
 return fig
 
 def create_summary_dashboard(
 self,
 eda_result, # EDAResult type
 ) -> go.Figure:
 """
 Create a summary dashboard from EDA results.
 
 Args:
 eda_result: EDAResult from Auto-EDA
 
 Returns:
 Plotly Figure with subplots
 """
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=(
 "Missing Values by Column",
 "Data Types Distribution",
 "Numeric Column Distributions",
 "Top Correlations",
 ),
 specs=[
 [{"type": "bar"}, {"type": "pie"}],
 [{"type": "box"}, {"type": "bar"}],
 ],
 )
 
 # Plot 1: Missing values
 missing_data = {
 col: profile.missing_percent
 for col, profile in eda_result.profile.columns.items()
 if profile.missing_percent > 0
 }
 
 if missing_data:
 fig.add_trace(
 go.Bar(
 x=list(missing_data.keys())[:10],
 y=list(missing_data.values())[:10],
 marker_color=self.colors[0],
 name="Missing %",
 ),
 row=1, col=1,
 )
 
 # Plot 2: Data types
 type_counts = {
 "Numeric": len(eda_result.profile.numeric_columns),
 "Categorical": len(eda_result.profile.categorical_columns),
 "Datetime": len(eda_result.profile.datetime_columns),
 "Text": len(eda_result.profile.text_columns),
 }
 
 fig.add_trace(
 go.Pie(
 labels=list(type_counts.keys()),
 values=list(type_counts.values()),
 marker_colors=self.colors[:4],
 ),
 row=1, col=2,
 )
 
 # Plot 3 & 4 would require actual data
 # Simplified for now
 
 fig.update_layout(
 title="Data Quality Dashboard",
 template=self.theme,
 height=600,
 showlegend=False,
 )
 
 return fig
 
 def export_to_png(self, fig: go.Figure) -> bytes:
 """
 Export figure to PNG bytes.
 
 Args:
 fig: Plotly Figure
 
 Returns:
 PNG image bytes
 """
 return fig.to_image(format="png", width=800, height=500)
 
 def export_to_html(self, fig: go.Figure) -> str:
 """
 Export figure to HTML string.
 
 Args:
 fig: Plotly Figure
 
 Returns:
 HTML string
 """
 return fig.to_html(include_plotlyjs="cdn", full_html=False)
