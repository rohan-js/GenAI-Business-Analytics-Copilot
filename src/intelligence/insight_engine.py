"""
Insight Engine Module

Generates data-backed insights from query results:
- Pattern recognition
- Trend analysis
- Anomaly highlighting
- Business interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from ..nlp.llm_engine import LLMEngine
from ..nlp.prompts import INSIGHT_GENERATION_PROMPT, INSIGHT_QUICK_PROMPT, format_prompt


@dataclass
class Insight:
 """A single generated insight."""
 
 title: str
 description: str
 evidence: List[str] = field(default_factory=list)
 insight_type: str = "general" # 'trend', 'anomaly', 'comparison', 'summary', 'general'
 confidence: str = "medium" # 'high', 'medium', 'low'
 business_impact: str = ""
 
 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 "title": self.title,
 "description": self.description,
 "evidence": self.evidence,
 "type": self.insight_type,
 "confidence": self.confidence,
 "impact": self.business_impact,
 }


@dataclass
class InsightResult:
 """Collection of insights from analysis."""
 
 question: str
 insights: List[Insight] = field(default_factory=list)
 summary: str = ""
 raw_llm_response: str = ""
 
 @property
 def primary_insight(self) -> Optional[Insight]:
 """Get the most important insight."""
 return self.insights[0] if self.insights else None
 
 def get_all_evidence(self) -> List[str]:
 """Get all evidence points across insights."""
 evidence = []
 for insight in self.insights:
 evidence.extend(insight.evidence)
 return evidence


class InsightEngine:
 """
 Generates business insights from data analysis results.
 
 Uses a combination of:
 - Rule-based pattern detection
 - LLM-powered interpretation
 - Statistical analysis
 """
 
 def __init__(self, llm_engine: Optional[LLMEngine] = None):
 """
 Initialize the insight engine.
 
 Args:
 llm_engine: LLM engine for natural language generation
 """
 self.llm_engine = llm_engine
 
 def generate_insights(
 self,
 question: str,
 result: Any,
 df: Optional[pd.DataFrame] = None,
 dataset_name: str = "Dataset",
 use_llm: bool = True,
 ) -> InsightResult:
 """
 Generate insights from a query result.
 
 Args:
 question: Original user question
 result: Query execution result
 df: Full DataFrame for context
 dataset_name: Name of dataset
 use_llm: Whether to use LLM for interpretation
 
 Returns:
 InsightResult with generated insights
 """
 insights = []
 summary = ""
 raw_response = ""
 
 # Rule-based insights
 rule_insights = self._generate_rule_based_insights(result, question)
 insights.extend(rule_insights)
 
 # LLM-based interpretation
 if use_llm and self.llm_engine is not None:
 llm_insight, raw_response = self._generate_llm_insight(
 question, result, df, dataset_name
 )
 if llm_insight:
 insights.append(llm_insight)
 summary = llm_insight.description
 
 # Generate summary if not from LLM
 if not summary and insights:
 summary = insights[0].description
 
 return InsightResult(
 question=question,
 insights=insights,
 summary=summary,
 raw_llm_response=raw_response,
 )
 
 def _generate_rule_based_insights(
 self,
 result: Any,
 question: str,
 ) -> List[Insight]:
 """
 Generate insights using rule-based analysis.
 
 Args:
 result: Query result
 question: Original question
 
 Returns:
 List of Insight objects
 """
 insights = []
 
 if result is None:
 return insights
 
 # Handle DataFrame results
 if isinstance(result, pd.DataFrame):
 insights.extend(self._analyze_dataframe(result, question))
 
 # Handle Series results
 elif isinstance(result, pd.Series):
 insights.extend(self._analyze_series(result, question))
 
 # Handle scalar results
 elif isinstance(result, (int, float)):
 insight = Insight(
 title="Calculated Value",
 description=f"The result is {result:,.2f}" if isinstance(result, float) else f"The result is {result:,}",
 evidence=[f"Value = {result}"],
 insight_type="summary",
 confidence="high",
 )
 insights.append(insight)
 
 return insights
 
 def _analyze_dataframe(
 self, df: pd.DataFrame, question: str
 ) -> List[Insight]:
 """
 Analyze a DataFrame result.
 
 Args:
 df: DataFrame to analyze
 question: Original question
 
 Returns:
 List of insights
 """
 insights = []
 
 # Size insight
 if len(df) == 0:
 insights.append(Insight(
 title="No Results",
 description="The query returned no matching records",
 insight_type="summary",
 confidence="high",
 ))
 return insights
 
 # Identify numeric columns
 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
 # Top/Bottom analysis for sorted data
 if len(df) <= 10 and len(numeric_cols) > 0:
 main_numeric = numeric_cols[0]
 top_val = df[main_numeric].iloc[0]
 bottom_val = df[main_numeric].iloc[-1]
 
 if len(df.columns) > 1:
 label_col = df.columns[0] if df.columns[0] != main_numeric else df.columns[1]
 top_label = df[label_col].iloc[0]
 
 insights.append(Insight(
 title="Top Performer",
 description=f"'{top_label}' leads with {main_numeric} of {top_val:,.2f}",
 evidence=[
 f"Top value: {top_val:,.2f}",
 f"Bottom value: {bottom_val:,.2f}",
 f"Range: {top_val - bottom_val:,.2f}",
 ],
 insight_type="comparison",
 confidence="high",
 ))
 
 # Distribution analysis for numeric columns
 for col in numeric_cols[:2]: # Limit to first 2 numeric columns
 series = df[col].dropna()
 if len(series) > 0:
 mean_val = series.mean()
 std_val = series.std()
 
 if std_val > 0:
 cv = std_val / mean_val if mean_val != 0 else 0
 
 if cv > 0.5:
 insights.append(Insight(
 title=f"High Variability in {col}",
 description=f"'{col}' shows high variability (CV={cv:.2f}), ranging from {series.min():,.2f} to {series.max():,.2f}",
 evidence=[
 f"Mean: {mean_val:,.2f}",
 f"Std Dev: {std_val:,.2f}",
 f"Coefficient of Variation: {cv:.2%}",
 ],
 insight_type="anomaly",
 confidence="high",
 ))
 
 return insights
 
 def _analyze_series(
 self, series: pd.Series, question: str
 ) -> List[Insight]:
 """
 Analyze a Series result.
 
 Args:
 series: Series to analyze
 question: Original question
 
 Returns:
 List of insights
 """
 insights = []
 
 if len(series) == 0:
 insights.append(Insight(
 title="No Results",
 description="The query returned no data",
 insight_type="summary",
 confidence="high",
 ))
 return insights
 
 # For value counts or groupby results
 if len(series) <= 20:
 top_idx = series.index[0]
 top_val = series.iloc[0]
 total = series.sum()
 
 if isinstance(top_val, (int, float)) and total > 0:
 pct = top_val / total * 100
 
 insights.append(Insight(
 title="Top Category",
 description=f"'{top_idx}' is the leader with {top_val:,.2f} ({pct:.1f}% of total)",
 evidence=[
 f"Top: {top_idx} = {top_val:,.2f}",
 f"Total across all: {total:,.2f}",
 f"Concentration: {pct:.1f}%",
 ],
 insight_type="comparison",
 confidence="high",
 ))
 
 # Concentration analysis (Pareto)
 if len(series) >= 3 and total > 0:
 cumsum = series.cumsum()
 pareto_idx = (cumsum <= total * 0.8).sum()
 
 if pareto_idx < len(series) * 0.3:
 insights.append(Insight(
 title="High Concentration",
 description=f"Top {pareto_idx + 1} categories account for 80%+ of the total (Pareto effect)",
 evidence=[
 f"80% threshold reached at category #{pareto_idx + 1}",
 f"Total categories: {len(series)}",
 ],
 insight_type="summary",
 confidence="high",
 business_impact="Consider focusing resources on top performers",
 ))
 
 return insights
 
 def _generate_llm_insight(
 self,
 question: str,
 result: Any,
 df: Optional[pd.DataFrame],
 dataset_name: str,
 ) -> tuple[Optional[Insight], str]:
 """
 Generate insight using LLM.
 
 Args:
 question: User question
 result: Query result
 df: Full DataFrame
 dataset_name: Dataset name
 
 Returns:
 Tuple of (Insight, raw_response)
 """
 if self.llm_engine is None:
 return None, ""
 
 # Format result for prompt
 if isinstance(result, pd.DataFrame):
 result_str = result.head(20).to_string()
 columns = ", ".join(result.columns.tolist())
 elif isinstance(result, pd.Series):
 result_str = result.head(20).to_string()
 columns = str(result.name)
 else:
 result_str = str(result)
 columns = "N/A"
 
 total_rows = len(df) if df is not None else 0
 
 # Build prompt
 prompt = format_prompt(
 INSIGHT_GENERATION_PROMPT,
 question=question,
 result=result_str,
 dataset_name=dataset_name,
 total_rows=total_rows,
 columns=columns,
 )
 
 try:
 generation = self.llm_engine.generate(
 prompt,
 max_new_tokens=400,
 temperature=0.3,
 )
 
 # Parse LLM response into insight
 insight = self._parse_llm_response(generation.text)
 
 return insight, generation.text
 
 except Exception as e:
 return None, f"Error: {str(e)}"
 
 def _parse_llm_response(self, response: str) -> Insight:
 """
 Parse LLM response into structured insight.
 
 Args:
 response: Raw LLM response
 
 Returns:
 Insight object
 """
 # Extract sections (simple parsing)
 lines = response.strip().split("\n")
 
 title = "Analysis Result"
 description = response[:500]
 evidence = []
 impact = ""
 
 current_section = None
 
 for line in lines:
 line = line.strip()
 if not line:
 continue
 
 lower = line.lower()
 
 if "key finding" in lower or "finding:" in lower:
 current_section = "title"
 title = line.split(":", 1)[-1].strip() if ":" in line else line
 elif "detail" in lower or "evidence" in lower:
 current_section = "evidence"
 elif "business" in lower or "implication" in lower or "impact" in lower:
 current_section = "impact"
 elif current_section == "evidence":
 if line.startswith("-") or line.startswith("•"):
 evidence.append(line[1:].strip())
 elif current_section == "impact":
 impact += line + " "
 
 # Use full response as description if parsing didn't work well
 if len(title) < 5:
 title = "Key Insight"
 
 return Insight(
 title=title[:100],
 description=description,
 evidence=evidence[:5],
 insight_type="general",
 confidence="medium",
 business_impact=impact.strip()[:300],
 )
 
 def generate_quick_insight(
 self,
 result: Any,
 question: str,
 ) -> str:
 """
 Generate a quick one-liner insight.
 
 Args:
 result: Query result
 question: Original question
 
 Returns:
 Quick insight string
 """
 if result is None:
 return "No results found for this query."
 
 if isinstance(result, (int, float)):
 return f"The answer is {result:,.2f}" if isinstance(result, float) else f"The answer is {result:,}"
 
 if isinstance(result, pd.DataFrame):
 if len(result) == 0:
 return "The query returned no matching records."
 return f"Found {len(result):,} records. Top result: {result.iloc[0].to_dict()}"
 
 if isinstance(result, pd.Series):
 if len(result) == 0:
 return "No results found."
 return f"Top result: {result.index[0]} = {result.iloc[0]:,.2f}" if isinstance(result.iloc[0], (int, float)) else f"Top result: {result.index[0]}"
 
 return f"Result: {str(result)[:200]}"
