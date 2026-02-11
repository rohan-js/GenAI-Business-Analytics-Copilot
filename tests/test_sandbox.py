"""
Test Safe Code Execution Sandbox
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.sandbox import SafeCodeExecutor, CodeValidator, ExecutionResult


class TestCodeValidator:
 """Tests for CodeValidator."""
 
 @pytest.fixture
 def validator(self):
 return CodeValidator()
 
 def test_valid_pandas_code(self, validator):
 """Test that valid pandas code passes validation."""
 code = "result = df['Sales'].sum()"
 is_valid, error = validator.validate(code)
 assert is_valid
 assert error is None
 
 def test_valid_groupby_code(self, validator):
 """Test groupby operations."""
 code = "result = df.groupby('Region')['Sales'].mean()"
 is_valid, error = validator.validate(code)
 assert is_valid
 
 def test_blocks_exec(self, validator):
 """Test that exec is blocked."""
 code = "exec('print(1)')"
 is_valid, error = validator.validate(code)
 assert not is_valid
 assert "exec" in error.lower()
 
 def test_blocks_eval(self, validator):
 """Test that eval is blocked."""
 code = "result = eval('1+1')"
 is_valid, error = validator.validate(code)
 assert not is_valid
 assert "eval" in error.lower()
 
 def test_blocks_import(self, validator):
 """Test that imports are blocked."""
 code = "import os"
 is_valid, error = validator.validate(code)
 assert not is_valid
 
 def test_blocks_os_module(self, validator):
 """Test that os module access is blocked."""
 code = "import os; os.system('ls')"
 is_valid, error = validator.validate(code)
 assert not is_valid
 
 def test_blocks_open_file(self, validator):
 """Test that file operations are blocked."""
 code = "f = open('test.txt', 'w')"
 is_valid, error = validator.validate(code)
 assert not is_valid
 
 def test_blocks_dunder_access(self, validator):
 """Test that dunder method access is blocked."""
 code = "df.__class__.__bases__"
 is_valid, error = validator.validate(code)
 assert not is_valid
 assert "dunder" in error.lower()
 
 def test_syntax_error(self, validator):
 """Test that syntax errors are caught."""
 code = "result = df['Sales'" # Missing bracket
 is_valid, error = validator.validate(code)
 assert not is_valid
 assert "syntax" in error.lower()


class TestSafeCodeExecutor:
 """Tests for SafeCodeExecutor."""
 
 @pytest.fixture
 def executor(self):
 return SafeCodeExecutor(timeout_seconds=5)
 
 @pytest.fixture
 def sample_df(self):
 return pd.DataFrame({
 'Sales': [100, 200, 300, 400],
 'Region': ['North', 'South', 'North', 'South'],
 'Product': ['A', 'B', 'A', 'B'],
 })
 
 def test_execute_simple_aggregation(self, executor, sample_df):
 """Test simple aggregation execution."""
 code = "result = df['Sales'].sum()"
 result = executor.execute(code, sample_df)
 
 assert result.success
 assert result.result == 1000
 
 def test_execute_groupby(self, executor, sample_df):
 """Test groupby execution."""
 code = "result = df.groupby('Region')['Sales'].sum()"
 result = executor.execute(code, sample_df)
 
 assert result.success
 assert isinstance(result.result, pd.Series)
 assert result.result['North'] == 400
 assert result.result['South'] == 600
 
 def test_execute_filtering(self, executor, sample_df):
 """Test filtering execution."""
 code = "result = df[df['Sales'] > 200]"
 result = executor.execute(code, sample_df)
 
 assert result.success
 assert isinstance(result.result, pd.DataFrame)
 assert len(result.result) == 2
 
 def test_rejects_dangerous_code(self, executor, sample_df):
 """Test that dangerous code is rejected."""
 code = "import os; result = os.listdir('.')"
 result = executor.execute(code, sample_df)
 
 assert not result.success
 assert "Validation failed" in result.error_message
 
 def test_clean_code_removes_markdown(self, executor):
 """Test that markdown code blocks are cleaned."""
 code = "```python\nresult = 42\n```"
 cleaned = executor._clean_code(code)
 assert "```" not in cleaned
 assert "result = 42" in cleaned
 
 def test_execution_result_properties(self, executor, sample_df):
 """Test ExecutionResult properties."""
 code = "result = df['Sales'].mean()"
 result = executor.execute(code, sample_df)
 
 assert result.has_result
 assert result.execution_time_seconds >= 0
 
 def test_test_code_valid(self, executor):
 """Test code testing without execution."""
 is_valid, msg = executor.test_code("result = df.sum()")
 assert is_valid
 assert "valid" in msg.lower()
 
 def test_test_code_invalid(self, executor):
 """Test code testing catches errors."""
 is_valid, msg = executor.test_code("import os")
 assert not is_valid


class TestExecutionResult:
 """Tests for ExecutionResult dataclass."""
 
 def test_get_result_summary_dataframe(self):
 """Test summary for DataFrame result."""
 df = pd.DataFrame({'A': [1, 2, 3]})
 result = ExecutionResult(
 success=True,
 result=df,
 result_type="DataFrame"
 )
 
 summary = result.get_result_summary()
 assert "3 rows" in summary
 assert "1 column" in summary
 
 def test_get_result_summary_scalar(self):
 """Test summary for scalar result."""
 result = ExecutionResult(
 success=True,
 result=42.5,
 result_type="float"
 )
 
 summary = result.get_result_summary()
 assert "42.5" in summary
 
 def test_get_result_summary_error(self):
 """Test summary for error result."""
 result = ExecutionResult(
 success=False,
 result=None,
 error_message="Test error"
 )
 
 summary = result.get_result_summary()
 assert "Error" in summary


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
