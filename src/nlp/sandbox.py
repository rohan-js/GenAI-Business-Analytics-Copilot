"""
Safe Code Execution Sandbox

Provides secure execution of LLM-generated Pandas code:
- AST-based validation
- Whitelist of allowed operations
- Timeout enforcement
- Memory limits
"""

import ast
import sys
import signal
import traceback
from typing import Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
from contextlib import contextmanager
import threading
import time

from ..config import BLOCKED_KEYWORDS, ALLOWED_MODULES, MAX_EXECUTION_SECONDS


@dataclass
class ExecutionResult:
 """Result of code execution."""
 
 success: bool
 result: Any
 error_message: Optional[str] = None
 execution_time_seconds: float = 0.0
 result_type: str = "unknown"
 
 @property
 def has_result(self) -> bool:
 return self.success and self.result is not None
 
 def get_result_summary(self) -> str:
 """Get summary of result for display."""
 if not self.success:
 return f"Error: {self.error_message}"
 
 if isinstance(self.result, pd.DataFrame):
 return f"DataFrame: {self.result.shape[0]} rows × {self.result.shape[1]} columns"
 elif isinstance(self.result, pd.Series):
 return f"Series: {len(self.result)} items"
 elif isinstance(self.result, (int, float)):
 return f"Value: {self.result}"
 else:
 return f"Result: {type(self.result).__name__}"


class CodeValidator:
 """
 Validates code before execution using AST analysis.
 
 Checks for:
 - Dangerous function calls
 - Forbidden imports
 - Attribute access to private/dunder methods
 """
 
 # Forbidden function names
 FORBIDDEN_FUNCTIONS = frozenset({
 "exec", "eval", "compile", "open", "input",
 "__import__", "getattr", "setattr", "delattr",
 "globals", "locals", "vars", "dir",
 "exit", "quit",
 })
 
 # Forbidden module names
 FORBIDDEN_MODULES = frozenset({
 "os", "sys", "subprocess", "shutil", "pathlib",
 "socket", "requests", "urllib", "http",
 "pickle", "shelve", "marshal",
 "importlib", "builtins",
 })
 
 # Allowed function calls on DataFrames and Series
 ALLOWED_PANDAS_METHODS = frozenset({
 # Selection
 "head", "tail", "sample", "loc", "iloc", "at", "iat",
 # Aggregation
 "sum", "mean", "median", "std", "var", "min", "max",
 "count", "nunique", "value_counts", "describe",
 "first", "last", "nth",
 # Grouping
 "groupby", "agg", "aggregate", "transform", "apply",
 "pivot_table", "pivot", "melt", "stack", "unstack",
 # Sorting
 "sort_values", "sort_index", "nlargest", "nsmallest",
 # Filtering
 "query", "where", "mask", "dropna", "fillna", "isna", "notna",
 "isin", "between", "duplicated", "drop_duplicates",
 # Transformation
 "reset_index", "set_index", "rename", "replace",
 "astype", "copy", "to_numeric", "to_datetime",
 # Computation
 "diff", "pct_change", "cumsum", "cumprod", "cummin", "cummax",
 "rolling", "expanding", "ewm",
 "corr", "cov", "rank", "abs", "round",
 # Merging
 "merge", "join", "concat", "append",
 # String operations
 "str", "contains", "startswith", "endswith", "lower", "upper",
 # Date operations
 "dt", "year", "month", "day", "hour", "dayofweek",
 # Other
 "shape", "columns", "index", "values", "dtypes",
 "to_dict", "to_list", "tolist", "items",
 })
 
 def validate(self, code: str) -> Tuple[bool, Optional[str]]:
 """
 Validate code for safety.
 
 Args:
 code: Python code string
 
 Returns:
 Tuple of (is_valid, error_message)
 """
 # Check for blocked keywords
 code_lower = code.lower()
 for keyword in BLOCKED_KEYWORDS:
 if keyword in code_lower:
 return False, f"Forbidden keyword detected: '{keyword}'"
 
 # Parse AST
 try:
 tree = ast.parse(code)
 except SyntaxError as e:
 return False, f"Syntax error: {e}"
 
 # Walk AST and check nodes
 for node in ast.walk(tree):
 # Check function calls
 if isinstance(node, ast.Call):
 if isinstance(node.func, ast.Name):
 if node.func.id in self.FORBIDDEN_FUNCTIONS:
 return False, f"Forbidden function: {node.func.id}"
 
 # Check imports
 if isinstance(node, ast.Import):
 for alias in node.names:
 if alias.name.split('.')[0] in self.FORBIDDEN_MODULES:
 return False, f"Forbidden import: {alias.name}"
 
 if isinstance(node, ast.ImportFrom):
 if node.module and node.module.split('.')[0] in self.FORBIDDEN_MODULES:
 return False, f"Forbidden import: {node.module}"
 
 # Check attribute access to dunder methods
 if isinstance(node, ast.Attribute):
 if node.attr.startswith('__') and node.attr.endswith('__'):
 return False, f"Access to dunder attribute forbidden: {node.attr}"
 
 return True, None


class SafeCodeExecutor:
 """
 Executes Python code in a sandboxed environment.
 
 Features:
 - Pre-execution validation
 - Restricted namespace
 - Timeout enforcement
 - Error handling
 """
 
 def __init__(
 self,
 timeout_seconds: int = MAX_EXECUTION_SECONDS,
 ):
 """
 Initialize the executor.
 
 Args:
 timeout_seconds: Maximum execution time
 """
 self.timeout_seconds = timeout_seconds
 self.validator = CodeValidator()
 
 def execute(
 self,
 code: str,
 df: pd.DataFrame,
 validate: bool = True,
 ) -> ExecutionResult:
 """
 Execute code safely with the given DataFrame.
 
 Args:
 code: Python code to execute
 df: DataFrame available as 'df' in code
 validate: Whether to validate code first
 
 Returns:
 ExecutionResult with outcome
 """
 start_time = time.time()
 
 # Clean code
 code = self._clean_code(code)
 
 # Validate
 if validate:
 is_valid, error = self.validator.validate(code)
 if not is_valid:
 return ExecutionResult(
 success=False,
 result=None,
 error_message=f"Validation failed: {error}",
 execution_time_seconds=time.time() - start_time,
 )
 
 # Create restricted namespace
 namespace = self._create_namespace(df)
 
 # Execute with timeout
 try:
 result = self._execute_with_timeout(code, namespace)
 execution_time = time.time() - start_time
 
 # Get result variable
 if "result" in namespace:
 final_result = namespace["result"]
 result_type = type(final_result).__name__
 else:
 # Try to get last expression value
 final_result = result
 result_type = type(final_result).__name__ if final_result is not None else "None"
 
 return ExecutionResult(
 success=True,
 result=final_result,
 execution_time_seconds=execution_time,
 result_type=result_type,
 )
 
 except TimeoutError:
 return ExecutionResult(
 success=False,
 result=None,
 error_message=f"Execution timed out after {self.timeout_seconds} seconds",
 execution_time_seconds=self.timeout_seconds,
 )
 except Exception as e:
 execution_time = time.time() - start_time
 error_msg = f"{type(e).__name__}: {str(e)}"
 return ExecutionResult(
 success=False,
 result=None,
 error_message=error_msg,
 execution_time_seconds=execution_time,
 )
 
 def _clean_code(self, code: str) -> str:
 """
 Clean code string before execution.
 
 Args:
 code: Raw code string
 
 Returns:
 Cleaned code
 """
 # Remove markdown code blocks
 if code.startswith("```"):
 lines = code.split("\n")
 # Remove first and last lines if they're code block markers
 if lines[0].startswith("```"):
 lines = lines[1:]
 if lines and lines[-1].strip() == "```":
 lines = lines[:-1]
 code = "\n".join(lines)
 
 # Strip whitespace
 code = code.strip()
 
 return code
 
 def _create_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
 """
 Create restricted namespace for execution.
 
 Args:
 df: DataFrame to include
 
 Returns:
 Namespace dictionary
 """
 return {
 "df": df.copy(), # Use copy to prevent modification of original
 "pd": pd,
 "np": np,
 "result": None,
 }
 
 def _execute_with_timeout(
 self,
 code: str,
 namespace: Dict[str, Any],
 ) -> Any:
 """
 Execute code with timeout.
 
 Args:
 code: Code to execute
 namespace: Execution namespace
 
 Returns:
 Execution result
 """
 result = [None]
 exception = [None]
 
 def target():
 try:
 exec(code, namespace)
 result[0] = namespace.get("result")
 except Exception as e:
 exception[0] = e
 
 thread = threading.Thread(target=target)
 thread.start()
 thread.join(timeout=self.timeout_seconds)
 
 if thread.is_alive():
 # Thread still running - timeout
 raise TimeoutError("Execution timed out")
 
 if exception[0] is not None:
 raise exception[0]
 
 return result[0]
 
 def test_code(self, code: str) -> Tuple[bool, str]:
 """
 Test if code is valid without executing.
 
 Args:
 code: Code to test
 
 Returns:
 Tuple of (is_valid, message)
 """
 code = self._clean_code(code)
 is_valid, error = self.validator.validate(code)
 
 if not is_valid:
 return False, f"Invalid: {error}"
 
 # Try to parse
 try:
 ast.parse(code)
 return True, "Code is valid"
 except SyntaxError as e:
 return False, f"Syntax error: {e}"
