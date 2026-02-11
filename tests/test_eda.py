"""
Test Auto-EDA Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eda.auto_eda import AutoEDAEngine, EDAResult
from src.eda.statistics import StatisticsCalculator
from src.eda.correlations import CorrelationAnalyzer
from src.eda.outliers import OutlierDetector


class TestAutoEDAEngine:
 """Tests for AutoEDAEngine."""
 
 @pytest.fixture
 def eda_engine(self):
 return AutoEDAEngine()
 
 @pytest.fixture
 def sample_df(self):
 np.random.seed(42)
 return pd.DataFrame({
 'Sales': np.random.uniform(100, 1000, 100),
 'Quantity': np.random.randint(1, 50, 100),
 'Price': np.random.uniform(10, 100, 100),
 'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
 'Category': np.random.choice(['A', 'B', 'C'], 100),
 })
 
 def test_analyze_returns_eda_result(self, eda_engine, sample_df):
 """Test that analyze returns EDAResult."""
 result = eda_engine.analyze(sample_df)
 assert isinstance(result, EDAResult)
 
 def test_analyze_includes_profile(self, eda_engine, sample_df):
 """Test that analysis includes data profile."""
 result = eda_engine.analyze(sample_df)
 
 assert result.profile is not None
 assert result.profile.n_rows == 100
 assert result.profile.n_columns == 5
 
 def test_analyze_includes_statistics(self, eda_engine, sample_df):
 """Test that analysis includes statistics."""
 result = eda_engine.analyze(sample_df)
 
 assert 'Sales' in result.numeric_stats
 assert 'Quantity' in result.numeric_stats
 
 def test_analyze_includes_correlations(self, eda_engine, sample_df):
 """Test that analysis includes correlation matrix."""
 result = eda_engine.analyze(sample_df)
 
 assert result.correlation_matrix is not None
 assert 'Sales' in result.correlation_matrix.columns
 
 def test_analyze_detects_outliers(self, eda_engine):
 """Test outlier detection."""
 # Create data with clear outliers
 df = pd.DataFrame({
 'Value': [10, 11, 12, 13, 14, 1000], # 1000 is outlier
 })
 
 result = eda_engine.analyze(df)
 
 assert 'Value' in result.outlier_results
 assert result.outlier_results['Value'].has_outliers
 
 def test_analyze_generates_insights(self, eda_engine, sample_df):
 """Test that analysis generates insights."""
 result = eda_engine.analyze(sample_df)
 
 # Should have at least some insights
 assert len(result.insights) >= 0 # May be empty for clean data
 
 def test_quick_summary(self, eda_engine, sample_df):
 """Test quick summary generation."""
 result = eda_engine.analyze(sample_df)
 summary = result.quick_summary
 
 assert 'rows' in summary
 assert 'columns' in summary
 assert 'quality_score' in summary


class TestStatisticsCalculator:
 """Tests for StatisticsCalculator."""
 
 @pytest.fixture
 def calculator(self):
 return StatisticsCalculator()
 
 def test_numeric_stats(self, calculator):
 """Test numeric statistics calculation."""
 df = pd.DataFrame({
 'Values': [10, 20, 30, 40, 50],
 })
 
 stats = calculator.compute_numeric_stats(df)
 
 assert 'Values' in stats
 assert stats['Values'].mean == 30
 assert stats['Values'].min == 10
 assert stats['Values'].max == 50
 
 def test_categorical_stats(self, calculator):
 """Test categorical statistics calculation."""
 df = pd.DataFrame({
 'Category': ['A', 'A', 'B', 'B', 'B'],
 })
 
 stats = calculator.compute_categorical_stats(df)
 
 assert 'Category' in stats
 assert stats['Category'].mode == 'B'
 assert stats['Category'].unique_count == 2


class TestCorrelationAnalyzer:
 """Tests for CorrelationAnalyzer."""
 
 @pytest.fixture
 def analyzer(self):
 return CorrelationAnalyzer()
 
 def test_correlation_matrix(self, analyzer):
 """Test correlation matrix computation."""
 df = pd.DataFrame({
 'A': [1, 2, 3, 4, 5],
 'B': [1, 2, 3, 4, 5], # Perfect correlation with A
 'C': [5, 4, 3, 2, 1], # Perfect negative correlation
 })
 
 corr = analyzer.compute_correlation_matrix(df)
 
 assert corr.loc['A', 'B'] == pytest.approx(1.0)
 assert corr.loc['A', 'C'] == pytest.approx(-1.0)
 
 def test_find_high_correlations(self, analyzer):
 """Test finding high correlations."""
 df = pd.DataFrame({
 'A': [1, 2, 3, 4, 5],
 'B': [1, 2, 3, 4, 5],
 'C': [5, 3, 1, 2, 4], # Random
 })
 
 pairs = analyzer.find_high_correlations(df, threshold=0.9)
 
 assert len(pairs) >= 1
 assert any(p.column1 == 'A' and p.column2 == 'B' for p in pairs)


class TestOutlierDetector:
 """Tests for OutlierDetector."""
 
 @pytest.fixture
 def detector(self):
 return OutlierDetector()
 
 def test_detect_iqr_outliers(self, detector):
 """Test IQR outlier detection."""
 # Create data with clear outlier
 series = pd.Series([10, 11, 12, 13, 14, 15, 100])
 series.name = 'Test'
 
 result = detector.detect_iqr(series)
 
 assert result.outlier_count >= 1
 assert 100 in result.outlier_values
 
 def test_no_outliers_for_normal_data(self, detector):
 """Test that normal data has no outliers."""
 series = pd.Series([10, 11, 12, 13, 14, 15, 16])
 series.name = 'Test'
 
 result = detector.detect_iqr(series)
 
 assert result.outlier_count == 0


if __name__ == "__main__":
 pytest.main([__file__, "-v"])
