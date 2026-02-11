"""
Test Data Ingestion Module
"""

import pytest
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestionModule, IngestionResult


class TestDataIngestionModule:
 """Tests for DataIngestionModule."""
 
 @pytest.fixture
 def ingestion_module(self):
 """Create ingestion module instance."""
 return DataIngestionModule()
 
 @pytest.fixture
 def sample_csv_bytes(self):
 """Create sample CSV data."""
 csv_content = """Name,Age,Salary,Department
John,30,50000,Sales
Jane,25,60000,Marketing
Bob,35,70000,Engineering
Alice,28,55000,Sales
"""
 return csv_content.encode('utf-8')
 
 def test_load_csv_from_bytes(self, ingestion_module, sample_csv_bytes):
 """Test loading CSV from bytes."""
 file_obj = io.BytesIO(sample_csv_bytes)
 
 result = ingestion_module.load(file_obj, filename="test.csv")
 
 assert isinstance(result, IngestionResult)
 assert isinstance(result.df, pd.DataFrame)
 assert len(result.df) == 4
 assert len(result.df.columns) == 4
 assert result.filename == "test.csv"
 
 def test_missing_value_standardization(self, ingestion_module):
 """Test that missing values are standardized."""
 csv_content = """A,B,C
1,NA,test
2,,valid
3,NULL,
"""
 file_obj = io.BytesIO(csv_content.encode('utf-8'))
 
 result = ingestion_module.load(file_obj, filename="test.csv")
 
 # Check that NA, NULL, and empty strings are converted to NaN
 assert result.df['B'].isna().sum() == 2 # NA and empty
 assert result.df['C'].isna().sum() == 1 # empty string
 
 def test_column_name_cleaning(self, ingestion_module):
 """Test that column names are cleaned."""
 csv_content = """ Name ,Age[years],Value<test>
John,30,100
"""
 file_obj = io.BytesIO(csv_content.encode('utf-8'))
 
 result = ingestion_module.load(file_obj, filename="test.csv")
 
 # Check column names are stripped and cleaned
 assert 'Name' in result.df.columns
 assert 'Ageyears' in result.df.columns # brackets removed
 
 def test_sampling_large_dataset(self, ingestion_module):
 """Test that large datasets are sampled."""
 # Create a module with low sample threshold
 module = DataIngestionModule(max_sample_rows=10)
 
 # Create larger dataset
 data = pd.DataFrame({
 'A': range(100),
 'B': range(100),
 })
 csv_bytes = data.to_csv(index=False).encode('utf-8')
 file_obj = io.BytesIO(csv_bytes)
 
 result = module.load(file_obj, filename="test.csv", sample_for_llm=True)
 
 assert result.is_sampled
 assert len(result.df) == 10
 assert result.original_rows == 100
 
 def test_file_validation(self, ingestion_module):
 """Test file validation."""
 # Valid file
 is_valid, error = ingestion_module.validate_file("test.csv", 1000)
 assert is_valid
 assert error is None
 
 # Invalid extension
 is_valid, error = ingestion_module.validate_file("test.txt", 1000)
 assert not is_valid
 assert "Unsupported" in error
 
 def test_get_sample_data(self, ingestion_module, sample_csv_bytes):
 """Test getting sample rows."""
 file_obj = io.BytesIO(sample_csv_bytes)
 result = ingestion_module.load(file_obj, filename="test.csv")
 
 sample = ingestion_module.get_sample_data(result.df, n_rows=2)
 
 assert len(sample) == 2


class TestIngestionResult:
 """Tests for IngestionResult dataclass."""
 
 def test_shape_str(self):
 """Test shape string formatting."""
 df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
 
 result = IngestionResult(
 df=df,
 filename="test.csv",
 file_size_mb=0.1,
 encoding="utf-8",
 original_rows=3,
 original_columns=2,
 )
 
 assert "3 rows" in result.shape_str
 assert "2 columns" in result.shape_str
 
 def test_summary(self):
 """Test summary dictionary."""
 df = pd.DataFrame({'A': [1, 2, 3]})
 
 result = IngestionResult(
 df=df,
 filename="test.csv",
 file_size_mb=0.5,
 encoding="utf-8",
 original_rows=3,
 original_columns=1,
 load_time_seconds=1.5,
 )
 
 summary = result.summary
 
 assert summary["filename"] == "test.csv"
 assert "0.50 MB" in summary["size"]


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
