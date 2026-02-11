# GenAI Business Analytics Copilot

 **AI-Powered Business Analytics for Everyone**

A production-grade, 100% free, locally-runnable GenAI analytics assistant that enables business users to query datasets using natural language and receive data-backed insights, visualizations, and recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- ** Smart Data Ingestion** — Upload CSV/Excel files with automatic schema detection
- ** Auto-EDA** — Instant summary statistics, correlations, and outlier detection
- ** Natural Language Queries** — Ask business questions in plain English
- ** Insight Generation** — AI-powered explanations with data-backed reasoning
- ** Auto-Visualization** — Smart chart selection based on data characteristics
- ** Recommendations** — Actionable business suggestions with clear assumptions

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB RAM recommended (for LLM inference)
- ~5GB disk space (for model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genai-business-analytics-copilot.git
cd genai-business-analytics-copilot

# Create virtual environment
python -m venv venv
venv\Scripts\activate # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### First Run

On first run, the app will download the LLM model (~2-3GB). This is a one-time download.

## Sample Queries

Try asking:
- *"Why did sales drop in Q3?"*
- *"Which customers are high risk?"*
- *"What factors impact revenue the most?"*
- *"Compare performance across regions"*
- *"Give 3 recommendations to improve revenue"*

## Project Structure

```
 app.py # Main Streamlit application
 requirements.txt # Python dependencies
 src/
 data/ # Data ingestion & profiling
 eda/ # Auto-EDA engine
 nlp/ # LLM & query translation
 intelligence/ # Insights & recommendations
 viz/ # Visualization generation
 prompts/ # LLM prompt templates
 data/sample/ # Sample datasets
```

## Safety Features

- **Sandboxed Execution** — Generated code runs in a restricted environment
- **Input Validation** — All user inputs are validated before processing
- **No External Calls** — 100% local, no data leaves your machine

## Technology Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| LLM | HuggingFace Transformers (Phi-2/Gemma) |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Orchestration | LangChain |

## Sample Datasets

The app comes with sample datasets for testing:
1. **Superstore Sales** — Retail analytics
2. **Telco Customer Churn** — Customer analytics
3. **Insurance Claims** — Risk analytics

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

Built with for the analytics community
