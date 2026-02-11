"""
Recommendation Engine Module

Generates actionable business recommendations:
- Data-backed suggestions
- Prioritized by impact
- Clear assumptions stated
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..nlp.llm_engine import LLMEngine
from ..nlp.prompts import RECOMMENDATION_PROMPT, format_prompt


@dataclass
class Recommendation:
 """A single business recommendation."""
 
 title: str
 action: str
 rationale: str
 expected_impact: str
 assumptions: List[str] = field(default_factory=list)
 priority: str = "medium" # 'high', 'medium', 'low'
 category: str = "general" # 'growth', 'efficiency', 'risk', 'general'
 
 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 "title": self.title,
 "action": self.action,
 "rationale": self.rationale,
 "expected_impact": self.expected_impact,
 "assumptions": self.assumptions,
 "priority": self.priority,
 "category": self.category,
 }
 
 def to_markdown(self) -> str:
 """Convert to markdown format."""
 md = f"### {self.title}\n\n"
 md += f"**Action:** {self.action}\n\n"
 md += f"**Rationale:** {self.rationale}\n\n"
 md += f"**Expected Impact:** {self.expected_impact}\n\n"
 if self.assumptions:
 md += "**Assumptions:**\n"
 for assumption in self.assumptions:
 md += f"- {assumption}\n"
 return md


@dataclass
class RecommendationResult:
 """Collection of recommendations."""
 
 recommendations: List[Recommendation] = field(default_factory=list)
 context_summary: str = ""
 raw_llm_response: str = ""
 
 def get_by_priority(self, priority: str) -> List[Recommendation]:
 """Get recommendations by priority level."""
 return [r for r in self.recommendations if r.priority == priority]
 
 def get_by_category(self, category: str) -> List[Recommendation]:
 """Get recommendations by category."""
 return [r for r in self.recommendations if r.category == category]
 
 def to_markdown(self) -> str:
 """Convert all recommendations to markdown."""
 if not self.recommendations:
 return "No recommendations generated."
 
 md = "## Recommendations\n\n"
 for i, rec in enumerate(self.recommendations, 1):
 md += f"**{i}. {rec.title}** ({rec.priority.upper()} priority)\n\n"
 md += f"- **Action:** {rec.action}\n"
 md += f"- **Why:** {rec.rationale}\n"
 md += f"- **Impact:** {rec.expected_impact}\n\n"
 return md


class RecommendationEngine:
 """
 Generates business recommendations from analysis results.
 
 Features:
 - Rule-based recommendations for common patterns
 - LLM-powered contextual recommendations
 - Priority and category classification
 """
 
 def __init__(self, llm_engine: Optional[LLMEngine] = None):
 """
 Initialize the recommendation engine.
 
 Args:
 llm_engine: LLM engine for natural language generation
 """
 self.llm_engine = llm_engine
 
 def generate_recommendations(
 self,
 analysis_summary: str,
 findings: List[str],
 context: str = "",
 use_llm: bool = True,
 max_recommendations: int = 3,
 ) -> RecommendationResult:
 """
 Generate recommendations based on analysis results.
 
 Args:
 analysis_summary: Summary of the analysis performed
 findings: List of key findings
 context: Additional context about the data/business
 use_llm: Whether to use LLM for generation
 max_recommendations: Maximum number of recommendations
 
 Returns:
 RecommendationResult with recommendations
 """
 recommendations = []
 raw_response = ""
 
 # Rule-based recommendations
 rule_recs = self._generate_rule_based_recommendations(findings)
 recommendations.extend(rule_recs)
 
 # LLM-based recommendations
 if use_llm and self.llm_engine is not None:
 llm_recs, raw_response = self._generate_llm_recommendations(
 analysis_summary, findings, context
 )
 recommendations.extend(llm_recs)
 
 # Deduplicate and prioritize
 recommendations = self._prioritize_recommendations(
 recommendations, max_recommendations
 )
 
 return RecommendationResult(
 recommendations=recommendations,
 context_summary=analysis_summary,
 raw_llm_response=raw_response,
 )
 
 def generate_from_eda(
 self,
 eda_result, # EDAResult type
 dataset_name: str = "Dataset",
 ) -> RecommendationResult:
 """
 Generate recommendations from EDA results.
 
 Args:
 eda_result: EDAResult from Auto-EDA
 dataset_name: Name of the dataset
 
 Returns:
 RecommendationResult
 """
 recommendations = []
 
 # Data quality recommendations
 if eda_result.profile.overall_missing_percent > 10:
 recommendations.append(Recommendation(
 title="Address Missing Data",
 action=f"Investigate and impute missing values in columns with >10% missing data",
 rationale=f"Dataset has {eda_result.profile.overall_missing_percent:.1f}% missing values overall, which may bias analysis results",
 expected_impact="Improved data quality and more reliable insights",
 assumptions=["Missing data is not systematically biased"],
 priority="high",
 category="efficiency",
 ))
 
 # High correlation recommendations
 if eda_result.high_correlations:
 corr = eda_result.high_correlations[0]
 recommendations.append(Recommendation(
 title="Investigate Variable Relationship",
 action=f"Analyze the relationship between '{corr.column1}' and '{corr.column2}'",
 rationale=f"These variables show {corr.strength} {corr.direction} correlation (r={corr.correlation:.3f})",
 expected_impact="Understanding this relationship could reveal business drivers",
 assumptions=["Correlation indicates meaningful business relationship"],
 priority="medium",
 category="general",
 ))
 
 # Outlier recommendations
 outlier_cols = [
 r.column for r in eda_result.outlier_results.values()
 if r.severity in ["high", "moderate"]
 ]
 
 if outlier_cols:
 recommendations.append(Recommendation(
 title="Review Outlier Records",
 action=f"Investigate outliers in: {', '.join(outlier_cols[:3])}",
 rationale="Outliers may represent data errors, fraud, or exceptional business cases",
 expected_impact="Data cleaning or identification of special cases requiring attention",
 assumptions=["Outliers are not already known/expected business scenarios"],
 priority="medium",
 category="risk",
 ))
 
 return RecommendationResult(
 recommendations=recommendations[:5],
 context_summary=f"EDA analysis of {dataset_name}",
 )
 
 def generate_quick_recommendations(
 self,
 finding: str,
 ) -> List[str]:
 """
 Generate quick recommendation bullets from a finding.
 
 Args:
 finding: A single finding statement
 
 Returns:
 List of recommendation strings
 """
 recommendations = []
 
 # Pattern matching for common findings
 finding_lower = finding.lower()
 
 if "decline" in finding_lower or "drop" in finding_lower or "decrease" in finding_lower:
 recommendations.extend([
 "Investigate root causes through customer/sales team interviews",
 "Compare affected period to historical benchmarks",
 "Review any process or policy changes during the period",
 ])
 
 elif "increase" in finding_lower or "growth" in finding_lower:
 recommendations.extend([
 "Identify success factors to replicate in other areas",
 "Allocate additional resources to capitalize on momentum",
 "Document best practices for institutional knowledge",
 ])
 
 elif "correlation" in finding_lower or "relationship" in finding_lower:
 recommendations.extend([
 "Test causality through A/B experiments if possible",
 "Build predictive models using correlated variables",
 "Consider this relationship in strategic planning",
 ])
 
 elif "outlier" in finding_lower or "anomaly" in finding_lower:
 recommendations.extend([
 "Review outlier records for data quality issues",
 "Investigate if outliers represent fraud or errors",
 "Consider winsorizing outliers for statistical analysis",
 ])
 
 else:
 recommendations.extend([
 "Share this finding with relevant stakeholders",
 "Monitor this metric going forward",
 "Consider deeper analysis to understand drivers",
 ])
 
 return recommendations[:3]
 
 def _generate_rule_based_recommendations(
 self,
 findings: List[str],
 ) -> List[Recommendation]:
 """
 Generate recommendations based on pattern matching.
 
 Args:
 findings: List of finding statements
 
 Returns:
 List of Recommendation objects
 """
 recommendations = []
 
 for finding in findings[:5]: # Limit processing
 quick_recs = self.generate_quick_recommendations(finding)
 
 if quick_recs:
 recommendations.append(Recommendation(
 title="Action Based on Finding",
 action=quick_recs[0],
 rationale=finding,
 expected_impact="Improved understanding and potential optimization",
 priority="medium",
 category="general",
 ))
 
 return recommendations
 
 def _generate_llm_recommendations(
 self,
 analysis_summary: str,
 findings: List[str],
 context: str,
 ) -> tuple[List[Recommendation], str]:
 """
 Generate recommendations using LLM.
 
 Args:
 analysis_summary: Summary of analysis
 findings: List of findings
 context: Additional context
 
 Returns:
 Tuple of (recommendations, raw_response)
 """
 if self.llm_engine is None:
 return [], ""
 
 prompt = format_prompt(
 RECOMMENDATION_PROMPT,
 analysis_summary=analysis_summary,
 findings="\n".join(f"- {f}" for f in findings),
 context=context or "General business dataset",
 )
 
 try:
 generation = self.llm_engine.generate(
 prompt,
 max_new_tokens=500,
 temperature=0.4,
 )
 
 recommendations = self._parse_llm_recommendations(generation.text)
 return recommendations, generation.text
 
 except Exception as e:
 return [], f"Error: {str(e)}"
 
 def _parse_llm_recommendations(
 self,
 response: str,
 ) -> List[Recommendation]:
 """
 Parse LLM response into structured recommendations.
 
 Args:
 response: Raw LLM response
 
 Returns:
 List of Recommendation objects
 """
 recommendations = []
 
 # Split by recommendation markers
 sections = response.split("RECOMMENDATION")
 
 for section in sections[1:]: # Skip first empty split
 try:
 rec = self._parse_single_recommendation(section)
 if rec:
 recommendations.append(rec)
 except Exception:
 continue
 
 # Fallback: treat entire response as one recommendation
 if not recommendations and len(response) > 50:
 recommendations.append(Recommendation(
 title="Suggested Action",
 action=response[:200],
 rationale="Based on data analysis",
 expected_impact="Potential improvement",
 priority="medium",
 ))
 
 return recommendations[:3]
 
 def _parse_single_recommendation(
 self,
 section: str,
 ) -> Optional[Recommendation]:
 """
 Parse a single recommendation section.
 
 Args:
 section: Text section for one recommendation
 
 Returns:
 Recommendation object or None
 """
 lines = section.strip().split("\n")
 
 title = ""
 action = ""
 rationale = ""
 impact = ""
 
 for line in lines:
 line = line.strip()
 lower = line.lower()
 
 if ":" in line:
 key, value = line.split(":", 1)
 key_lower = key.lower()
 value = value.strip()
 
 if "action" in key_lower or "what" in key_lower:
 action = value
 elif "why" in key_lower or "rationale" in key_lower:
 rationale = value
 elif "impact" in key_lower or "expected" in key_lower:
 impact = value
 
 # Use first non-empty line as title if not found
 if not title:
 for line in lines:
 clean = line.strip().strip("*#-:")
 if len(clean) > 5:
 title = clean[:100]
 break
 
 if action or title:
 return Recommendation(
 title=title or "Recommendation",
 action=action or title,
 rationale=rationale or "Based on analysis findings",
 expected_impact=impact or "Potential improvement",
 priority="medium",
 )
 
 return None
 
 def _prioritize_recommendations(
 self,
 recommendations: List[Recommendation],
 max_count: int,
 ) -> List[Recommendation]:
 """
 Prioritize and deduplicate recommendations.
 
 Args:
 recommendations: List of all recommendations
 max_count: Maximum to return
 
 Returns:
 Prioritized list
 """
 # Sort by priority
 priority_order = {"high": 0, "medium": 1, "low": 2}
 
 sorted_recs = sorted(
 recommendations,
 key=lambda r: priority_order.get(r.priority, 1)
 )
 
 # Simple deduplication by title similarity
 seen_titles = set()
 unique_recs = []
 
 for rec in sorted_recs:
 title_key = rec.title.lower()[:30]
 if title_key not in seen_titles:
 seen_titles.add(title_key)
 unique_recs.append(rec)
 
 return unique_recs[:max_count]
