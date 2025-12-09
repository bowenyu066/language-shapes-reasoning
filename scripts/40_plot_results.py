#!/usr/bin/env python3
"""
Plot the final results of models.
"""
import os
import sys
import json
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import plot_multi_bar, plot_multi_bar_v2, plot_length_distribution, plot_length_distribution_ridge

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
    # accuracy_es = {
    #     "ChatGPT-5.1": 0.7781,
    #     "Gemini-2.5-Flash": 0.8289,
    #     "DeepSeek-V3.2": 0.8850,
    # }
    # accuracy_th = {
    #     "ChatGPT-5.1": 0.7086,
    #     "Gemini-2.5-Flash": 0.8021,
    #     "DeepSeek-V3.2": 0.8503,
    # }
    # max_param = 0.9
    # min_param = 0.7
    # plot_multi_bar_v2(
    #     [accuracy_en, accuracy_zh, accuracy_es, accuracy_th],
    #     ["English", "Chinese", "Spanish", "Thai"],
    #     max_param,
    #     min_param,
    #     "Models",
    #     "results/mmath/en_vs_zh_vs_es_vs_th.pdf",
    #     yname="MMATH Accuracy",
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
    # chatgpt_en_path = "results/mmath/summary_chatgpt-5.1_en_direct_8192.json"
    # chatgpt_zh_path = "results/mmath/summary_chatgpt-5.1_zh_direct_8192.json"
    # chatgpt_es_path = "results/mmath/summary_chatgpt-5.1_es_direct_8192.json"
    # chatgpt_th_path = "results/mmath/summary_chatgpt-5.1_th_direct_8192.json"
    # deepseek_en_path = "results/mmath/summary_deepseek-v3.2_en_direct_8192.json"
    # deepseek_zh_path = "results/mmath/summary_deepseek-v3.2_zh_direct_8192.json"
    # deepseek_es_path = "results/mmath/summary_deepseek-v3.2_es_direct_8192.json"
    # deepseek_th_path = "results/mmath/summary_deepseek-v3.2_th_direct_8192.json"
    # gemini_en_path = "results/mmath/summary_gemini-2.5_en_direct_8192.json"
    # gemini_zh_path = "results/mmath/summary_gemini-2.5_zh_direct_8192.json"
    # gemini_es_path = "results/mmath/summary_gemini-2.5_es_direct_8192.json"
    # gemini_th_path = "results/mmath/summary_gemini-2.5_th_direct_8192.json"

    # chatgpt_en_lengths = json.load(open(chatgpt_en_path))["output_lengths"]
    # chatgpt_zh_lengths = json.load(open(chatgpt_zh_path))["output_lengths"]
    # chatgpt_es_lengths = json.load(open(chatgpt_es_path))["output_lengths"]
    # chatgpt_th_lengths = json.load(open(chatgpt_th_path))["output_lengths"]
    # deepseek_en_lengths = json.load(open(deepseek_en_path))["output_lengths"]
    # deepseek_zh_lengths = json.load(open(deepseek_zh_path))["output_lengths"]
    # deepseek_es_lengths = json.load(open(deepseek_es_path))["output_lengths"]
    # deepseek_th_lengths = json.load(open(deepseek_th_path))["output_lengths"]
    # gemini_en_lengths = json.load(open(gemini_en_path))["output_lengths"]
    # gemini_zh_lengths = json.load(open(gemini_zh_path))["output_lengths"]
    # gemini_es_lengths = json.load(open(gemini_es_path))["output_lengths"]
    # gemini_th_lengths = json.load(open(gemini_th_path))["output_lengths"]

    # plot_length_distribution_ridge(
    #     [gemini_en_lengths, gemini_zh_lengths, gemini_es_lengths, gemini_th_lengths],
    #     "results/mmath/length_distribution_gemini(2).pdf",
    #     xname="Output Lengths (Gemini-2.5-Flash)",
    #     legend_labels=["English", "Chinese", "Spanish", "Thai"],
    # )
    # plot_length_distribution_ridge(
    #     [deepseek_en_lengths, deepseek_zh_lengths, deepseek_es_lengths, deepseek_th_lengths],
    #     "results/mmath/length_distribution_deepseek(2).pdf",
    #     xname="Output Lengths (DeepSeek-V3.2)",
    #     legend_labels=["English", "Chinese", "Spanish", "Thai"],
    # )
    # plot_length_distribution_ridge(
    #     [chatgpt_en_lengths, chatgpt_zh_lengths, chatgpt_es_lengths, chatgpt_th_lengths],
    #     "results/mmath/length_distribution_chatgpt(2).pdf",
    #     xname="Output Lengths (ChatGPT-5.1)",
    #     legend_labels=["English", "Chinese", "Spanish", "Thai"],
    # )

    qwen_en_csv = "results/gsm8k/gsm8k_qwen3-8b_en_direct_maxtok4096.csv"
    qwen_zh_csv = "results/gsm8k/gsm8k_qwen3-8b_zh_direct_maxtok4096.csv"
    llama_en_csv = "results/gsm8k/gsm8k_llama-3.1-8b_en_direct_maxtok4096.csv"
    llama_zh_csv = "results/gsm8k/gsm8k_llama-3.1-8b_zh_direct_maxtok4096.csv"
    
    qwen_en_lengths = pd.read_csv(qwen_en_csv)["raw_output_length"].tolist()
    qwen_zh_lengths = pd.read_csv(qwen_zh_csv)["raw_output_length"].tolist()
    llama_en_lengths = pd.read_csv(llama_en_csv)["raw_output_length"].tolist()
    llama_zh_lengths = pd.read_csv(llama_zh_csv)["raw_output_length"].tolist()

    plot_length_distribution(
        [qwen_en_lengths, qwen_zh_lengths],
        "results/gsm8k/length_distribution_qwen.pdf",
        xname="Output Lengths (Qwen3-8B)",
        legend_labels=["English", "Chinese"],
    )
    plot_length_distribution(
        [llama_en_lengths, llama_zh_lengths],
        "results/gsm8k/length_distribution_llama.pdf",
        xname="Output Lengths (Llama-3.1-8B-Instruct)",
        legend_labels=["English", "Chinese"],
    )