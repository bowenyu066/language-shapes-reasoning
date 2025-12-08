#!/usr/bin/env python3
"""
Plot the final results of models.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import plot_multi_bar

if __name__ == "__main__":
    accuracy_en = {
        "ChatGPT-5.1": 0.7995,
        "Gemini-2.5-Flash": 0.8182,
        "DeepSeek-V3.2": 0.8850,
    }
    accuracy_zh = {
        "ChatGPT-5.1": 0.7834,
        "Gemini-2.5-Flash": 0.8075,
        "DeepSeek-V3.2": 0.8770,
    }
    max_param = 0.9
    min_param = 0.7
    plot_multi_bar(
        accuracy_en,
        accuracy_zh,
        max_param,
        min_param,
        "Models",
        "results/mmath/en_vs_zh.pdf",
    )
    
    