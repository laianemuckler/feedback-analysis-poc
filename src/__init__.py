"""
Sistema de Análise de Feedbacks - PoC
Módulos principais para processamento e análise
"""

__version__ = "1.0.0"
__author__ = "Seu Nome"

from .utils import clean_text, CONFIG
from .data_generator import create_synthetic_dataset
from .preprocessor import FeedbackPreprocessor