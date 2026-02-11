"""
EDA (Exploratory Data Analysis) package.
"""

from .auto_eda import AutoEDAEngine
from .statistics import StatisticsCalculator
from .correlations import CorrelationAnalyzer
from .outliers import OutlierDetector

__all__ = [
 "AutoEDAEngine",
 "StatisticsCalculator", 
 "CorrelationAnalyzer",
 "OutlierDetector",
]
