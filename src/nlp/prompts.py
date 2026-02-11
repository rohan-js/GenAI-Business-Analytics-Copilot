"""
Prompt Templates for the LLM

Contains all prompt templates for:
- NL to Pandas code translation
- Insight generation
- Recommendation generation
"""

# =============================================================================
# NL to Pandas Translation Prompts
# =============================================================================

NL_TO_PANDAS_PROMPT = """You are a data analyst assistant. Convert the user's business question into executable Pandas code.

DATASET SCHEMA:
{schema}

SAMPLE DATA (first 3 rows):
{sample_data}

COLUMN DETAILS:
{column_details}

USER QUESTION: {question}

RULES:
1. The dataframe is named 'df' and is already loaded
2. Use only pandas and numpy operations (imported as pd and np)
3. Return a single expression or statements ending with assignment to 'result'
4. Do NOT use print statements
5. Handle missing values appropriately with .fillna() or .dropna()
6. For date operations, use pd.to_datetime()
7. For aggregations, use appropriate .groupby() operations
8. Assign the final answer to a variable called 'result'

EXAMPLES:
- "What is the total sales?" → result = df['Sales'].sum()
- "Show sales by region" → result = df.groupby('Region')['Sales'].sum()
- "Top 5 customers by revenue" → result = df.groupby('Customer')['Revenue'].sum().nlargest(5)

OUTPUT: Only Python code, no explanations or markdown.

```python
"""

NL_TO_PANDAS_SIMPLE_PROMPT = """Convert this question to Pandas code.
DataFrame: df
Columns: {columns}
Question: {question}

Rules:
- Assign result to 'result'
- Use pandas (pd) and numpy (np) only
- No print statements

Code:
```python
"""


# =============================================================================
# Insight Generation Prompts
# =============================================================================

INSIGHT_GENERATION_PROMPT = """You are a senior business analyst. Analyze the data result and provide clear insights.

ORIGINAL QUESTION: {question}

DATA RESULT:
{result}

CONTEXT:
- Dataset: {dataset_name}
- Total rows in dataset: {total_rows}
- Analysis columns: {columns}

Provide a structured response with:

1. **KEY FINDING**: One clear sentence stating the main insight

2. **DETAILS**: 
 - Specific numbers from the data
 - Comparisons or trends if relevant
 - Notable patterns

3. **BUSINESS IMPLICATION**: What this means for the business

4. **CAVEATS**: Any limitations or things to consider

Be specific, cite actual numbers, avoid generic statements. Use a professional, consulting-style tone.
"""

INSIGHT_QUICK_PROMPT = """Analyze this data result concisely:

Question: {question}
Result: {result}

Provide:
1. Key finding (1 sentence)
2. Why it matters (1-2 sentences)
3. One actionable suggestion

Be specific and data-driven.
"""


# =============================================================================
# Recommendation Generation Prompts
# =============================================================================

RECOMMENDATION_PROMPT = """Based on the analysis below, provide actionable business recommendations.

ANALYSIS SUMMARY:
{analysis_summary}

KEY FINDINGS:
{findings}

DATASET CONTEXT:
{context}

Provide exactly 3 recommendations in this format:

**RECOMMENDATION 1: [Action Title]**
- **What to do**: [Specific, actionable step]
- **Why**: [Data-backed rationale]
- **Expected impact**: [Quantified if possible]
- **Assumption**: [What must be true for this to work]

**RECOMMENDATION 2: [Action Title]**
...

**RECOMMENDATION 3: [Action Title]**
...

Be consulting-grade specific. Avoid generic advice like "analyze more data."
"""

RECOMMENDATION_QUICK_PROMPT = """Given this finding: {finding}

Suggest 3 quick actions a business could take. Be specific and practical.
Format: numbered list with brief rationale for each.
"""


# =============================================================================
# Explanation Prompts
# =============================================================================

WHY_EXPLANATION_PROMPT = """Explain why the following pattern exists in the data.

OBSERVATION: {observation}

RELEVANT DATA:
{data_context}

POSSIBLE FACTORS TO CONSIDER:
{factors}

Provide:
1. Most likely explanation based on the data
2. Supporting evidence from the numbers
3. Alternative hypotheses to investigate
4. What additional data would confirm the explanation

Be analytical and avoid speculation without data support.
"""


DRIVER_ANALYSIS_PROMPT = """Identify the key drivers of {metric} in this dataset.

DATA SUMMARY:
{summary}

CORRELATIONS:
{correlations}

Explain:
1. Top 3 factors most correlated with {metric}
2. Direction of influence (positive/negative)
3. Strength of relationship
4. Potential confounding factors

Use the actual correlation values in your explanation.
"""


# =============================================================================
# Comparison Prompts
# =============================================================================

COMPARISON_PROMPT = """Compare {entity1} vs {entity2} based on the data.

COMPARISON DATA:
{comparison_data}

Provide:
1. **Winner**: Which performs better overall and by how much
2. **Key differences**: Top 3 metrics where they differ most
3. **Similarities**: Where they're comparable
4. **Recommendation**: Which to focus on and why

Use specific numbers from the data.
"""


# =============================================================================
# Chart Suggestion Prompts
# =============================================================================

CHART_SUGGESTION_PROMPT = """Suggest the best chart type for this data and question.

Question: {question}
Data shape: {data_shape}
Column types: {column_types}
Unique values: {unique_values}

Options: bar, line, scatter, pie, histogram, box, heatmap

Respond with only:
CHART_TYPE: [type]
TITLE: [suggested title]
X_AXIS: [column for x]
Y_AXIS: [column for y]
"""


# =============================================================================
# Helper function to format prompts
# =============================================================================

def format_prompt(template: str, **kwargs) -> str:
 """
 Format a prompt template with variables.
 
 Args:
 template: Prompt template string
 **kwargs: Variable values
 
 Returns:
 Formatted prompt string
 """
 try:
 return template.format(**kwargs)
 except KeyError as e:
 raise ValueError(f"Missing prompt variable: {e}")


def get_schema_string(df, max_cols: int = 20) -> str:
 """
 Generate schema string for prompts.
 
 Args:
 df: DataFrame
 max_cols: Maximum columns to include
 
 Returns:
 Formatted schema string
 """
 import pandas as pd
 
 lines = []
 for i, (col, dtype) in enumerate(df.dtypes.items()):
 if i >= max_cols:
 lines.append(f"... and {len(df.columns) - max_cols} more columns")
 break
 
 # Get sample values
 sample = df[col].dropna().head(3).tolist()
 sample_str = ", ".join(str(v)[:30] for v in sample)
 
 lines.append(f"- {col} ({dtype}): e.g., {sample_str}")
 
 return "\n".join(lines)
