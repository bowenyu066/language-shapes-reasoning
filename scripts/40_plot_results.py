#!/usr/bin/env python3
"""
Plot the final results of models.
"""
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import plot_multi_bar, plot_multi_bar_v2, plot_length_distribution

if __name__ == "__main__":
    # accuracy_en = {
    #     "ChatGPT-5.1": 0.7995,
    #     "Gemini-2.5-Flash": 0.8182,
    #     "DeepSeek-V3.2": 0.8850,
    # }
    # accuracy_zh = {
    #     "ChatGPT-5.1": 0.7834,
    #     "Gemini-2.5-Flash": 0.8075,
    #     "DeepSeek-V3.2": 0.8770,
    # }
    # max_param = 0.9
    # min_param = 0.7
    # plot_multi_bar(
    #     accuracy_en,
    #     accuracy_zh,
    #     max_param,
    #     min_param,
    #     "Models",
    #     "results/mmath/en_vs_zh.pdf",
    # )
    # accuracy_en = {
    #     "Qwen3-8B": 0.8870,
    #     "Llama-3.1-8B-Instruct": 0.8029,
    # }
    # accuracy_zh = {
    #     "Qwen3-8B": 0.8939,
    #     "Llama-3.1-8B-Instruct": 0.5284,
    # }
    # accuracy_zh_translate_then_solve = {
    #     "Qwen3-8B": 0.8984,
    #     "Llama-3.1-8B-Instruct": 0.4367,
    # }
    # max_param = 0.95
    # min_param = 0.4
    # plot_multi_bar_v2(
    #     accuracy_en,
    #     accuracy_zh,
    #     accuracy_zh_translate_then_solve,
    #     max_param,
    #     min_param,
    #     "Models",
    #     "results/gsm8k/en_vs_zh_vs_zh_translate_then_solve.pdf",
    # )
    chatgpt_en_path = "results/mmath/summary_chatgpt-5.1_en_direct_8192.json"
    chatgpt_zh_path = "results/mmath/summary_chatgpt-5.1_zh_direct_8192.json"
    deepseek_en_path = "results/mmath/summary_deepseek-v3.2_en_direct_8192.json"
    deepseek_zh_path = "results/mmath/summary_deepseek-v3.2_zh_direct_8192.json"
    gemini_en_path = "results/mmath/summary_gemini-2.5_en_direct_8192.json"
    gemini_zh_path = "results/mmath/summary_gemini-2.5_zh_direct_8192.json"

    chatgpt_en_lengths = json.load(open(chatgpt_en_path))["output_lengths"]
    chatgpt_zh_lengths = json.load(open(chatgpt_zh_path))["output_lengths"]
    deepseek_en_lengths = json.load(open(deepseek_en_path))["output_lengths"]
    deepseek_zh_lengths = json.load(open(deepseek_zh_path))["output_lengths"]
    gemini_en_lengths = json.load(open(gemini_en_path))["output_lengths"]
    gemini_zh_lengths = json.load(open(gemini_zh_path))["output_lengths"]

    plot_length_distribution(
        [gemini_en_lengths, gemini_zh_lengths],
        "results/mmath/length_distribution_gemini.pdf",
        xname="Output Lengths (Gemini-2.5-Flash)",
        legend_labels=["Gemini-2.5-Flash (English)", "Gemini-2.5-Flash (Chinese)"],
    )
    plot_length_distribution(
        [deepseek_en_lengths, deepseek_zh_lengths],
        "results/mmath/length_distribution_deepseek.pdf",
        xname="Output Lengths (DeepSeek-V3.2)",
        legend_labels=["DeepSeek-V3.2 (English)", "DeepSeek-V3.2 (Chinese)"],
    )
    plot_length_distribution(
        [chatgpt_en_lengths, chatgpt_zh_lengths],
        "results/mmath/length_distribution_chatgpt.pdf",
        xname="Output Lengths (ChatGPT-5.1)",
        legend_labels=["ChatGPT-5.1 (English)", "ChatGPT-5.1 (Chinese)"],
    )