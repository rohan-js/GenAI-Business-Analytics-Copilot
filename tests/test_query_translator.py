"""
Test Query Translator Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.query_translator import QueryTranslator, QueryResult
from src.nlp.sandbox import SafeCodeExecutor


class TestQueryTranslator:
 """Tests for QueryTranslator."""
 
 @pytest.fixture
 def sample_df(self):
 return pd.DataFrame({
 'Sales': [100, 200, 300, 400, 500],
 'Region': ['North', 'South', 'North', 'South', 'East'],
 'Product': ['A', 'B', 'A', 'B', 'C'],
 })
 
 @pytest.fixture
 def mock_llm_engine(self):
 """Create a mock LLM engine."""
 mock = MagicMock()
 mock.generate.return_value = MagicMock(
 text="result = df['Sales'].sum()",
 generation_time_seconds=0.5,
 )
 return mock
 
 @pytest.fixture
 def translator_with_mock(self, mock_llm_engine):
 """Create translator with mock LLM."""
 return QueryTranslator(
 llm_engine=mock_llm_engine,
 executor=SafeCodeExecutor(),
 )
 
 def test_extract_code_from_markdown(self, translator_with_mock):
 """Test extracting code from markdown blocks."""
 response = "```python\nresult = df['Sales'].sum()\n```"
 code = translator_with_mock._extract_code(response)
 
 assert "result = df['Sales'].sum()" in code
 assert "```" not in code
 
 def test_extract_code_plain(self, translator_with_mock):
 """Test extracting plain code."""
 response = "result = df['Sales'].mean()"
 code = translator_with_mock._extract_code(response)
 
 assert code == "result = df['Sales'].mean()"
 
 def test_build_prompt_includes_schema(self, translator_with_mock, sample_df):
 """Test that prompt includes schema information."""
 question = "What is the total sales?"
 prompt = translator_with_mock._build_prompt(question, sample_df)
 
 assert "Sales" in prompt
 assert "Region" in prompt
 assert question in prompt
 
 def test_suggest_questions(self, translator_with_mock, sample_df):
 """Test question suggestions."""
 suggestions = translator_with_mock.suggest_questions(sample_df)
 
 assert len(suggestions) > 0
 # Should suggest questions based on column names
 any_sales = any("Sales" in s for s in suggestions)
 assert any_sales


class TestQueryResult:
 """Tests for QueryResult dataclass."""
 
 def test_success_property(self):
 """Test success property."""
 execution_result = MagicMock()
 execution_result.success = True
 execution_result.result = 100
 
 generation_result = MagicMock()
 generation_result.generation_time_seconds = 0.5
 
 result = QueryResult(
 question="test",
 generated_code="result = 100",
 execution_result=execution_result,
 generation_result=generation_result,
 )
 
 assert result.success
 assert result.result == 100
 
 def test_error_property(self):
 """Test error property on failure."""
 execution_result = MagicMock()
 execution_result.success = False
 execution_result.error_message = "Test error"
 
 generation_result = MagicMock()
 
 result = QueryResult(
 question="test",
 generated_code="bad code",
 execution_result=execution_result,
 generation_result=generation_result,
 )
 
 assert not result.success
 assert result.error == "Test error"


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
