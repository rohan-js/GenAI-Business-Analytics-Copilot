"""
Query Translator Module

Translates natural language questions to executable Pandas code:
- Prompt construction with schema context
- LLM-based code generation
- Result post-processing
"""

import pandas as pd
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .llm_engine import LLMEngine, GenerationResult
from .sandbox import SafeCodeExecutor, ExecutionResult
from .prompts import (
 NL_TO_PANDAS_PROMPT,
 NL_TO_PANDAS_SIMPLE_PROMPT,
 get_schema_string,
)


@dataclass
class QueryResult:
 """Complete result of a query translation and execution."""
 
 question: str
 generated_code: str
 execution_result: ExecutionResult
 generation_result: GenerationResult
 
 @property
 def success(self) -> bool:
 return self.execution_result.success
 
 @property
 def result(self) -> Any:
 return self.execution_result.result
 
 @property
 def error(self) -> Optional[str]:
 if not self.execution_result.success:
 return self.execution_result.error_message
 return None
 
 def get_summary(self) -> Dict[str, Any]:
 """Get summary for display."""
 return {
 "question": self.question,
 "success": self.success,
 "result_type": self.execution_result.result_type,
 "generation_time": f"{self.generation_result.generation_time_seconds:.2f}s",
 "execution_time": f"{self.execution_result.execution_time_seconds:.2f}s",
 "total_time": f"{self.generation_result.generation_time_seconds + self.execution_result.execution_time_seconds:.2f}s",
 }


class QueryTranslator:
 """
 Translates natural language questions to Pandas code and executes them.
 
 Pipeline:
 1. Construct prompt with schema context
 2. Generate code using LLM
 3. Validate and clean generated code
 4. Execute in sandbox
 5. Return structured result
 """
 
 def __init__(
 self,
 llm_engine: Optional[LLMEngine] = None,
 executor: Optional[SafeCodeExecutor] = None,
 ):
 """
 Initialize the query translator.
 
 Args:
 llm_engine: LLM engine for code generation (creates default if None)
 executor: Code executor (creates default if None)
 """
 self.llm_engine = llm_engine or LLMEngine()
 self.executor = executor or SafeCodeExecutor()
 
 def translate_and_execute(
 self,
 question: str,
 df: pd.DataFrame,
 max_retries: int = 2,
 ) -> QueryResult:
 """
 Translate a question to Pandas code and execute it.
 
 Args:
 question: Natural language question
 df: DataFrame to query
 max_retries: Number of retries on failure
 
 Returns:
 QueryResult with all details
 """
 # Build prompt
 prompt = self._build_prompt(question, df)
 
 # Generate code
 generation_result = self.llm_engine.generate(
 prompt,
 max_new_tokens=300,
 temperature=0.1,
 )
 
 # Extract and clean code
 code = self._extract_code(generation_result.text)
 
 # Execute
 execution_result = self.executor.execute(code, df)
 
 # Retry if failed
 attempts = 1
 while not execution_result.success and attempts < max_retries:
 # Try with simplified prompt
 simple_prompt = self._build_simple_prompt(question, df)
 generation_result = self.llm_engine.generate(
 simple_prompt,
 max_new_tokens=200,
 temperature=0.05, # Even more deterministic
 )
 code = self._extract_code(generation_result.text)
 execution_result = self.executor.execute(code, df)
 attempts += 1
 
 return QueryResult(
 question=question,
 generated_code=code,
 execution_result=execution_result,
 generation_result=generation_result,
 )
 
 def translate_only(
 self,
 question: str,
 df: pd.DataFrame,
 ) -> Tuple[str, GenerationResult]:
 """
 Translate question to code without executing.
 
 Args:
 question: Natural language question
 df: DataFrame for context
 
 Returns:
 Tuple of (generated_code, generation_result)
 """
 prompt = self._build_prompt(question, df)
 generation_result = self.llm_engine.generate(
 prompt,
 max_new_tokens=300,
 temperature=0.1,
 )
 code = self._extract_code(generation_result.text)
 return code, generation_result
 
 def _build_prompt(self, question: str, df: pd.DataFrame) -> str:
 """
 Build the full prompt for code generation.
 
 Args:
 question: User question
 df: DataFrame
 
 Returns:
 Formatted prompt string
 """
 # Get schema information
 schema = get_schema_string(df)
 
 # Get sample data
 sample_data = df.head(3).to_string()
 
 # Get column details
 column_details = self._get_column_details(df)
 
 return NL_TO_PANDAS_PROMPT.format(
 schema=schema,
 sample_data=sample_data,
 column_details=column_details,
 question=question,
 )
 
 def _build_simple_prompt(self, question: str, df: pd.DataFrame) -> str:
 """
 Build a simplified prompt for retry attempts.
 
 Args:
 question: User question
 df: DataFrame
 
 Returns:
 Simplified prompt string
 """
 columns = ", ".join(df.columns.tolist()[:15])
 
 return NL_TO_PANDAS_SIMPLE_PROMPT.format(
 columns=columns,
 question=question,
 )
 
 def _get_column_details(self, df: pd.DataFrame) -> str:
 """
 Get detailed column information.
 
 Args:
 df: DataFrame
 
 Returns:
 Column details string
 """
 details = []
 
 for col in df.columns[:15]: # Limit columns
 dtype = df[col].dtype
 non_null = df[col].notna().sum()
 
 if pd.api.types.is_numeric_dtype(df[col]):
 min_val = df[col].min()
 max_val = df[col].max()
 details.append(
 f"- {col}: numeric, range [{min_val:.2f}, {max_val:.2f}], {non_null} non-null"
 )
 elif pd.api.types.is_datetime64_any_dtype(df[col]):
 details.append(f"- {col}: datetime, {non_null} non-null")
 else:
 unique = df[col].nunique()
 details.append(
 f"- {col}: categorical, {unique} unique values, {non_null} non-null"
 )
 
 return "\n".join(details)
 
 def _extract_code(self, response: str) -> str:
 """
 Extract Python code from LLM response.
 
 Args:
 response: Raw LLM response
 
 Returns:
 Cleaned Python code
 """
 # Look for code blocks
 code_block_pattern = r"```(?:python)?\s*(.*?)```"
 matches = re.findall(code_block_pattern, response, re.DOTALL)
 
 if matches:
 code = matches[0].strip()
 else:
 # No code block, use entire response
 code = response.strip()
 
 # Remove any remaining markdown
 code = re.sub(r"^```python\s*", "", code)
 code = re.sub(r"\s*```$", "", code)
 
 # Ensure result assignment exists
 if "result" not in code and "=" in code:
 # Find the last assignment and rename it to result
 lines = code.strip().split("\n")
 if lines:
 last_line = lines[-1]
 if "=" in last_line and not last_line.strip().startswith("#"):
 # Check if it's already an assignment
 var_name = last_line.split("=")[0].strip()
 if var_name and not var_name.startswith("result"):
 lines.append(f"result = {var_name}")
 code = "\n".join(lines)
 
 # If still no result, wrap the code
 if "result" not in code:
 code = f"result = {code}"
 
 return code
 
 def suggest_questions(self, df: pd.DataFrame) -> list[str]:
 """
 Suggest sample questions based on the dataset.
 
 Args:
 df: DataFrame
 
 Returns:
 List of suggested questions
 """
 suggestions = []
 
 numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
 categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
 # Basic aggregation questions
 if numeric_cols:
 col = numeric_cols[0]
 suggestions.append(f"What is the total {col}?")
 suggestions.append(f"What is the average {col}?")
 
 # Group by questions
 if numeric_cols and categorical_cols:
 num_col = numeric_cols[0]
 cat_col = categorical_cols[0]
 suggestions.append(f"Show {num_col} by {cat_col}")
 suggestions.append(f"Which {cat_col} has the highest {num_col}?")
 
 # Top N questions
 if numeric_cols:
 col = numeric_cols[0]
 suggestions.append(f"What are the top 5 records by {col}?")
 
 # Comparison questions
 if len(categorical_cols) >= 2:
 suggestions.append(f"Compare {categorical_cols[0]} across different {categorical_cols[1]}")
 
 return suggestions[:5]
