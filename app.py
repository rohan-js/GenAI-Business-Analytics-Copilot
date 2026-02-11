"""
GenAI Business Analytics Copilot - Main Application

A production-grade, 100% free, locally-runnable GenAI analytics assistant
that enables business users to query datasets using natural language.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
 APP_TITLE,
 APP_SUBTITLE,
 PAGE_CONFIG,
 AVAILABLE_MODELS,
 DEFAULT_MODEL,
)
from src.data import DataIngestionModule, DataProfiler
from src.eda import AutoEDAEngine
from src.nlp import LLMEngine, QueryTranslator, SafeCodeExecutor
from src.intelligence import InsightEngine, RecommendationEngine, DriverAnalyzer
from src.viz import ChartGenerator


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(**PAGE_CONFIG)

# Custom CSS for styling
st.markdown("""
<style>
 /* Main header styling */
 .main-header {
 font-size: 2.5rem;
 font-weight: 700;
 background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
 -webkit-background-clip: text;
 -webkit-text-fill-color: transparent;
 margin-bottom: 0.5rem;
 }
 
 .sub-header {
 font-size: 1.1rem;
 color: #6b7280;
 margin-bottom: 2rem;
 }
 
 /* Metric cards */
 .metric-card {
 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
 padding: 1rem;
 border-radius: 0.75rem;
 color: white;
 }
 
 /* Insight box */
 .insight-box {
 background: #f0f9ff;
 border-left: 4px solid #0ea5e9;
 padding: 1rem;
 border-radius: 0 0.5rem 0.5rem 0;
 margin: 1rem 0;
 }
 
 /* Recommendation box */
 .recommendation-box {
 background: #f0fdf4;
 border-left: 4px solid #22c55e;
 padding: 1rem;
 border-radius: 0 0.5rem 0.5rem 0;
 margin: 1rem 0;
 }
 
 /* Chat message styling */
 .user-message {
 background: #e5e7eb;
 padding: 0.75rem 1rem;
 border-radius: 1rem;
 margin: 0.5rem 0;
 }
 
 .assistant-message {
 background: #dbeafe;
 padding: 0.75rem 1rem;
 border-radius: 1rem;
 margin: 0.5rem 0;
 }
 
 /* Hide Streamlit branding */
 #MainMenu {visibility: hidden;}
 footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
 """Initialize session state variables."""
 if "df" not in st.session_state:
 st.session_state.df = None
 
 if "eda_result" not in st.session_state:
 st.session_state.eda_result = None
 
 if "profile" not in st.session_state:
 st.session_state.profile = None
 
 if "ingestion_result" not in st.session_state:
 st.session_state.ingestion_result = None
 
 if "chat_history" not in st.session_state:
 st.session_state.chat_history = []
 
 if "llm_engine" not in st.session_state:
 st.session_state.llm_engine = None
 
 if "model_loaded" not in st.session_state:
 st.session_state.model_loaded = False


init_session_state()


# =============================================================================
# Component Initialization
# =============================================================================

@st.cache_resource
def get_data_ingestion():
 """Get cached data ingestion module."""
 return DataIngestionModule()


@st.cache_resource
def get_eda_engine():
 """Get cached EDA engine."""
 return AutoEDAEngine()


@st.cache_resource
def get_chart_generator():
 """Get cached chart generator."""
 return ChartGenerator()


def get_llm_engine(model_name: str = DEFAULT_MODEL):
 """Get or create LLM engine."""
 if st.session_state.llm_engine is None:
 st.session_state.llm_engine = LLMEngine(model_name=model_name)
 return st.session_state.llm_engine


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
 st.markdown("### Settings")
 
 # Model selection
 model_options = list(AVAILABLE_MODELS.keys())
 model_descriptions = [AVAILABLE_MODELS[m]["description"] for m in model_options]
 
 selected_model = st.selectbox(
 "LLM Model",
 options=model_options,
 format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]['description']}",
 help="Select the language model for query processing",
 )
 
 # Model loading status
 if st.session_state.model_loaded:
 st.success(" Model loaded")
 else:
 st.info("ℹ Model will load on first query")
 
 st.divider()
 
 # Data status
 st.markdown("### Data Status")
 
 if st.session_state.df is not None:
 st.success(f" Data loaded")
 if st.session_state.ingestion_result:
 st.write(f" {st.session_state.ingestion_result.filename}")
 st.write(f" {st.session_state.ingestion_result.shape_str}")
 
 if st.button(" Clear Data", use_container_width=True):
 st.session_state.df = None
 st.session_state.eda_result = None
 st.session_state.profile = None
 st.session_state.ingestion_result = None
 st.session_state.chat_history = []
 st.rerun()
 else:
 st.info("No data loaded")
 
 st.divider()
 
 # Sample datasets
 st.markdown("### Sample Datasets")
 st.markdown("""
 Try these free datasets:
 - [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
 - [Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
 - [Insurance Claims](https://www.kaggle.com/datasets/simranjain17/insurance)
 """)
 
 st.divider()
 
 # About
 st.markdown("### ℹ About")
 st.markdown("""
 **Business Analytics Copilot**
 
 100% free, runs locally, no API keys needed.
 
 Built with:
 - HuggingFace Transformers
 - Pandas & Plotly
 - Streamlit
 """)


# =============================================================================
# Main Content
# =============================================================================

# Header
st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

# Tab layout
tab_upload, tab_eda, tab_chat, tab_insights = st.tabs([
 " Upload Data",
 " Auto-EDA",
 " Ask Questions",
 " Insights & Recommendations",
])


# =============================================================================
# Tab 1: Data Upload
# =============================================================================

with tab_upload:
 st.markdown("### Upload Your Dataset")
 st.markdown("Upload a CSV or Excel file to get started. Your data stays 100% local.")
 
 uploaded_file = st.file_uploader(
 "Drag and drop or click to upload",
 type=["csv", "xlsx", "xls"],
 help="Supported formats: CSV, XLSX, XLS (max 100MB)",
 )
 
 if uploaded_file is not None:
 with st.spinner("Loading data..."):
 try:
 ingestion_module = get_data_ingestion()
 
 result = ingestion_module.load(
 uploaded_file,
 filename=uploaded_file.name,
 sample_for_llm=True,
 )
 
 st.session_state.df = result.df
 st.session_state.ingestion_result = result
 
 # Show success
 st.success(f" Loaded: {result.filename}")
 
 # Show warnings if any
 for warning in result.warnings:
 st.warning(warning)
 
 # Display summary metrics
 col1, col2, col3, col4 = st.columns(4)
 
 with col1:
 st.metric("Rows", f"{result.original_rows:,}")
 
 with col2:
 st.metric("Columns", result.original_columns)
 
 with col3:
 st.metric("Size", f"{result.file_size_mb:.2f} MB")
 
 with col4:
 st.metric("Load Time", f"{result.load_time_seconds:.2f}s")
 
 # Show data preview
 st.markdown("### Data Preview")
 st.dataframe(result.df.head(10), use_container_width=True)
 
 # Show column info
 with st.expander(" Column Information"):
 col_info = pd.DataFrame({
 "Column": result.df.columns,
 "Type": result.df.dtypes.astype(str),
 "Non-Null": result.df.notna().sum(),
 "Null %": (result.df.isna().sum() / len(result.df) * 100).round(2),
 })
 st.dataframe(col_info, use_container_width=True)
 
 except Exception as e:
 st.error(f" Error loading file: {str(e)}")
 
 else:
 # Show placeholder when no file uploaded
 st.info(" Upload a CSV or Excel file to begin analysis")
 
 # Sample data option
 st.markdown("---")
 st.markdown("### Or try with sample data")
 
 if st.button(" Load Sample Sales Data", use_container_width=True):
 # Create sample data
 np.random.seed(42)
 n_rows = 500
 
 sample_df = pd.DataFrame({
 "Order_Date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
 "Region": np.random.choice(["North", "South", "East", "West"], n_rows),
 "Category": np.random.choice(["Electronics", "Clothing", "Food", "Home"], n_rows),
 "Sales": np.random.uniform(100, 10000, n_rows).round(2),
 "Quantity": np.random.randint(1, 50, n_rows),
 "Profit": np.random.uniform(-500, 2000, n_rows).round(2),
 "Customer_Segment": np.random.choice(["Consumer", "Corporate", "Home Office"], n_rows),
 })
 
 st.session_state.df = sample_df
 st.session_state.ingestion_result = type("obj", (object,), {
 "filename": "sample_sales_data.csv",
 "shape_str": f"{n_rows} rows × {len(sample_df.columns)} columns",
 "original_rows": n_rows,
 "original_columns": len(sample_df.columns),
 "file_size_mb": 0.1,
 "load_time_seconds": 0.01,
 "warnings": [],
 })()
 
 st.success(" Sample data loaded!")
 st.rerun()


# =============================================================================
# Tab 2: Auto-EDA
# =============================================================================

with tab_eda:
 if st.session_state.df is None:
 st.info(" Please upload data first in the 'Upload Data' tab")
 else:
 st.markdown("### Automatic Exploratory Data Analysis")
 
 if st.button(" Run Auto-EDA", type="primary", use_container_width=True):
 with st.spinner("Analyzing data..."):
 eda_engine = get_eda_engine()
 eda_result = eda_engine.analyze(st.session_state.df)
 st.session_state.eda_result = eda_result
 st.session_state.profile = eda_result.profile
 
 if st.session_state.eda_result is not None:
 eda_result = st.session_state.eda_result
 
 # Summary metrics
 st.markdown("### Dataset Overview")
 
 col1, col2, col3, col4 = st.columns(4)
 
 with col1:
 st.metric(
 "Data Quality Score",
 f"{eda_result.profile.data_quality_score:.0f}/100",
 help="Based on completeness and data consistency",
 )
 
 with col2:
 st.metric(
 "Missing Data",
 f"{eda_result.profile.overall_missing_percent:.1f}%",
 )
 
 with col3:
 st.metric(
 "Numeric Columns",
 len(eda_result.profile.numeric_columns),
 )
 
 with col4:
 st.metric(
 "Categorical Columns",
 len(eda_result.profile.categorical_columns),
 )
 
 # Warnings
 if eda_result.warnings:
 st.markdown("### Data Quality Alerts")
 for warning in eda_result.warnings:
 st.warning(warning)
 
 # Insights
 if eda_result.insights:
 st.markdown("### Key Insights")
 for insight in eda_result.insights[:5]:
 st.markdown(f'<div class="insight-box"> {insight}</div>', unsafe_allow_html=True)
 
 # Statistics
 st.markdown("### Summary Statistics")
 
 if eda_result.summary_table is not None and not eda_result.summary_table.empty:
 st.dataframe(eda_result.summary_table, use_container_width=True)
 else:
 st.info("No numeric columns for statistical analysis")
 
 # Correlations
 if eda_result.correlation_matrix is not None and not eda_result.correlation_matrix.empty:
 st.markdown("### Correlations")
 
 chart_gen = get_chart_generator()
 corr_fig = chart_gen.create_correlation_heatmap(
 st.session_state.df,
 title="Correlation Matrix",
 )
 st.plotly_chart(corr_fig, use_container_width=True)
 
 # High correlations
 if eda_result.high_correlations:
 with st.expander(" High Correlations Details"):
 for pair in eda_result.high_correlations[:5]:
 st.write(
 f"**{pair.column1}** ↔ **{pair.column2}**: "
 f"r = {pair.correlation:.3f} ({pair.strength})"
 )
 
 # Outliers
 if eda_result.outlier_summary is not None and not eda_result.outlier_summary.empty:
 st.markdown("### Outlier Detection")
 
 outlier_cols = [
 r for r in eda_result.outlier_results.values()
 if r.has_outliers
 ]
 
 if outlier_cols:
 st.dataframe(eda_result.outlier_summary, use_container_width=True)
 else:
 st.success(" No significant outliers detected")


# =============================================================================
# Tab 3: Chat Interface
# =============================================================================

with tab_chat:
 if st.session_state.df is None:
 st.info(" Please upload data first in the 'Upload Data' tab")
 else:
 st.markdown("### Ask Questions in Natural Language")
 st.markdown("Ask business questions about your data. The AI will translate them to analytics queries.")
 
 # Suggested questions
 if st.session_state.profile:
 numeric_cols = st.session_state.profile.numeric_columns
 categorical_cols = st.session_state.profile.categorical_columns
 
 suggestions = []
 if numeric_cols:
 suggestions.append(f"What is the total {numeric_cols[0]}?")
 if numeric_cols and categorical_cols:
 suggestions.append(f"Show {numeric_cols[0]} by {categorical_cols[0]}")
 suggestions.append(f"Which {categorical_cols[0]} has the highest {numeric_cols[0]}?")
 if len(numeric_cols) >= 2:
 suggestions.append(f"What is the correlation between {numeric_cols[0]} and {numeric_cols[1]}?")
 
 if suggestions:
 st.markdown("**Try these questions:**")
 cols = st.columns(min(len(suggestions), 3))
 for i, suggestion in enumerate(suggestions[:3]):
 with cols[i]:
 if st.button(f" {suggestion[:40]}...", key=f"suggest_{i}"):
 st.session_state.current_question = suggestion
 
 # Chat history
 for message in st.session_state.chat_history:
 role = message["role"]
 content = message["content"]
 
 if role == "user":
 st.markdown(f'<div class="user-message"> {content}</div>', unsafe_allow_html=True)
 else:
 st.markdown(f'<div class="assistant-message"> {content}</div>', unsafe_allow_html=True)
 
 # Show result data if present
 if "result_data" in message and message["result_data"] is not None:
 result = message["result_data"]
 
 if isinstance(result, pd.DataFrame):
 st.dataframe(result.head(20), use_container_width=True)
 elif isinstance(result, pd.Series):
 st.dataframe(result.head(20), use_container_width=True)
 else:
 st.write(result)
 
 # Show chart if present
 if "chart" in message and message["chart"] is not None:
 st.plotly_chart(message["chart"], use_container_width=True)
 
 # Query input
 st.markdown("---")
 
 question = st.text_input(
 "Your question:",
 placeholder="e.g., What are the top 5 products by sales?",
 key="question_input",
 )
 
 col1, col2 = st.columns([1, 4])
 
 with col1:
 submit = st.button(" Ask", type="primary", use_container_width=True)
 
 with col2:
 if st.button(" Clear Chat", use_container_width=True):
 st.session_state.chat_history = []
 st.rerun()
 
 if submit and question:
 # Add user message
 st.session_state.chat_history.append({
 "role": "user",
 "content": question,
 })
 
 with st.spinner(" Thinking..."):
 try:
 # Get or create LLM engine
 llm_engine = get_llm_engine(AVAILABLE_MODELS[selected_model]["name"])
 
 # Load model if needed
 if not st.session_state.model_loaded:
 with st.status("Loading AI model (first time only)..."):
 llm_engine.load_model()
 st.session_state.model_loaded = True
 
 # Create query translator
 translator = QueryTranslator(llm_engine=llm_engine)
 
 # Translate and execute
 query_result = translator.translate_and_execute(
 question,
 st.session_state.df,
 )
 
 if query_result.success:
 # Generate insight
 insight_engine = InsightEngine(llm_engine=llm_engine)
 insight_result = insight_engine.generate_insights(
 question,
 query_result.result,
 st.session_state.df,
 use_llm=True,
 )
 
 # Generate chart
 chart_gen = get_chart_generator()
 
 result_data = query_result.result
 chart = None
 
 if isinstance(result_data, (pd.DataFrame, pd.Series)):
 if isinstance(result_data, pd.Series):
 chart_data = result_data.reset_index()
 chart_data.columns = ["Category", "Value"]
 else:
 chart_data = result_data
 
 if len(chart_data) > 0:
 chart, _ = chart_gen.create_auto_chart(chart_data, question)
 
 # Format response
 response = insight_result.summary if insight_result.summary else "Here are the results:"
 
 # Add assistant message
 st.session_state.chat_history.append({
 "role": "assistant",
 "content": response,
 "result_data": result_data,
 "chart": chart,
 "code": query_result.generated_code,
 })
 
 else:
 st.session_state.chat_history.append({
 "role": "assistant",
 "content": f" Sorry, I couldn't process that query. Error: {query_result.error}",
 })
 
 except Exception as e:
 st.session_state.chat_history.append({
 "role": "assistant",
 "content": f" An error occurred: {str(e)}",
 })
 
 st.rerun()


# =============================================================================
# Tab 4: Insights & Recommendations
# =============================================================================

with tab_insights:
 if st.session_state.df is None:
 st.info(" Please upload data first in the 'Upload Data' tab")
 elif st.session_state.eda_result is None:
 st.info(" Please run Auto-EDA first in the 'Auto-EDA' tab")
 else:
 st.markdown("### Business Insights & Recommendations")
 
 eda_result = st.session_state.eda_result
 
 # Generate recommendations
 rec_engine = RecommendationEngine()
 recommendations = rec_engine.generate_from_eda(
 eda_result,
 dataset_name=st.session_state.ingestion_result.filename if st.session_state.ingestion_result else "Dataset",
 )
 
 # Display recommendations
 if recommendations.recommendations:
 st.markdown("### Recommended Actions")
 
 for i, rec in enumerate(recommendations.recommendations, 1):
 with st.expander(f"**{i}. {rec.title}** ({rec.priority.upper()} priority)", expanded=(i == 1)):
 st.markdown(f"**What to do:** {rec.action}")
 st.markdown(f"**Why:** {rec.rationale}")
 st.markdown(f"**Expected Impact:** {rec.expected_impact}")
 
 if rec.assumptions:
 st.markdown("**Assumptions:**")
 for assumption in rec.assumptions:
 st.markdown(f"- {assumption}")
 
 # Driver analysis
 st.markdown("### Key Driver Analysis")
 
 numeric_cols = st.session_state.profile.numeric_columns
 
 if numeric_cols:
 target_col = st.selectbox(
 "Select target metric to analyze:",
 options=numeric_cols,
 help="Which numeric column do you want to understand the drivers for?",
 )
 
 if st.button(" Analyze Drivers", type="primary"):
 with st.spinner("Analyzing drivers..."):
 driver_analyzer = DriverAnalyzer()
 driver_result = driver_analyzer.analyze_drivers(
 st.session_state.df,
 target_col,
 )
 
 st.markdown(f"**Summary:** {driver_result.summary}")
 
 if driver_result.drivers:
 st.markdown("#### Top Drivers:")
 
 for driver in driver_result.top_drivers:
 col1, col2 = st.columns([1, 3])
 
 with col1:
 importance_pct = driver.importance_score * 100
 st.metric(
 driver.name,
 f"{importance_pct:.1f}%",
 delta=driver.direction,
 )
 
 with col2:
 st.markdown(f"*{driver.evidence}*")
 else:
 st.info("No significant drivers identified for this metric.")
 else:
 st.info("No numeric columns available for driver analysis.")
 
 # Export report
 st.markdown("### Export Report")
 
 if st.button(" Generate Report", use_container_width=True):
 # Generate text report
 eda_engine = get_eda_engine()
 report = eda_engine.generate_eda_report(eda_result)
 
 st.download_button(
 label=" Download EDA Report",
 data=report,
 file_name="eda_report.txt",
 mime="text/plain",
 )


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
 """
 <div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
 Business Analytics Copilot | 100% Local, 100% Free | 
 Built with for the analytics community
 </div>
 """,
 unsafe_allow_html=True,
)
