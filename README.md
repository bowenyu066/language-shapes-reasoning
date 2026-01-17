# Representation Efficiency in Neural Reasoning: Evidence from Multilingual Mathematical Benchmarks

*MIT 6.7960 Deep Learning • Fall 2025*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **TL;DR**: This repository investigates whether representation choice (e.g., different languages with varying token densities) affects neural reasoning performance. Our findings show that well-trained models achieve **representation-invariant reasoning**—they can solve the same mathematical problems with near-identical accuracy across languages while using 5-10% fewer tokens in denser representations like Chinese.

![Overview](/figures/overview.png)

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
- [Configuration](#configuration)
- [Results & Analysis](#results--analysis)
- [Extending the Pipeline](#extending-the-pipeline)
- [Citation](#citation)

## Overview

### Research Motivation

The Transformer's quadratic self-attention complexity ($O(n^2)$) makes sequence length a critical bottleneck in deep learning. While most efficiency research focuses on processing fixed representations more efficiently (sparse attention, token pruning, etc.), this project asks a fundamental question:

**Can we achieve the same reasoning quality with inherently denser representations?**

Natural languages provide an ideal testbed: the same mathematical problem can be expressed in different languages with dramatically different token counts. Chinese, for instance, often requires 30-50% fewer tokens than English for equivalent content.

### Research Questions

1. **Do well-trained models show representation-invariant reasoning?** Can models solve problems equally well regardless of the linguistic representation?

2. **Is performance tied to architecture or training distribution?** When models struggle with certain representations, is it fundamental or just a training gap?

3. **Can representation bottlenecks be learned?** If a model performs poorly on a denser representation, can targeted fine-tuning close the gap?

### Datasets

- **GSM8K**: 1,319 grade-school math word problems (English + Chinese translation)
- **MMATH**: 374 competition-level math problems in 10 languages (English, Chinese, Spanish, Thai, etc.)

### Models Evaluated

**SOTA API Models:**
- ChatGPT-5.1
- Gemini-2.5-Flash
- DeepSeek-V3.2

**Open-Source Models (8B parameters):**
- Qwen3-8B (multilingual-trained)
- Llama-3.1-8B-Instruct (English-centric)

**Fine-tuned Models:**
- Llama-3.1-8B + LoRA (Chinese GSM8K)

## Key Findings

### 1. Representation-Invariant Reasoning (SOTA Models)

**Well-trained frontier models achieve near-identical accuracy across languages:**

| Model | English | Chinese | Gap |
|-------|---------|---------|-----|
| ChatGPT-5.1 | 80.0% | 78.3% | 1.7% |
| DeepSeek-V3.2 | 88.5% | 87.7% | 0.8% |
| Gemini-2.5-Flash | 81.2% | 80.8% | 0.4% |

**Token efficiency varies by model-tokenizer alignment:**
- **DeepSeek-V3.2**: Chinese uses **11% fewer tokens** (aligned)
- **ChatGPT-5.1**: Chinese uses **33% more tokens** (inverted)
- **Gemini-2.5**: Chinese uses **40% more tokens** (inverted)

**Key Insight**: Intrinsic density (information per character) doesn't automatically translate to computational savings. The **Encoder Gap**—misalignment between tokenizer and representation—can turn efficiency into a liability.

### 2. Training Distribution Determines Capability

**Qwen3-8B vs. Llama-3.1-8B on GSM8K:**

| Model | English | Chinese | Gap |
|-------|---------|---------|-----|
| Qwen3-8B | 88.7% | 89.4% | +0.7% |
| Llama-3.1-8B | 80.3% | 52.8% | −27.5% |

Llama answered **446 problems correctly in English but incorrectly in Chinese** (33.8% of test set)—clear evidence that the bottleneck is training coverage, not representation capability.

### 3. Representation Bottlenecks Are Learnable

**Fine-tuning Llama-3.1-8B on Chinese GSM8K (500 steps LoRA):**
- Chinese accuracy: **52.8% → 61.6%** (+8.8%)
- English accuracy: **80.3% → 81.1%** (maintained)
- Gap reduced: **27.5% → 19.5%**

Modest fine-tuning substantially improves performance without catastrophic forgetting, demonstrating that representation efficiency is a **learnable capability**.

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (for local models)
- 16GB+ VRAM recommended for 8B models

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/language-shapes-reasoning.git
cd language-shapes-reasoning

# 2. Create and activate virtual environment
conda create -n lang-reasoning python=3.12
conda activate lang-reasoning

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install API client libraries (if using API models)
pip install -q -U google-genai  # for Gemini

# 5. Install flash-attention for local models (optional but recommended)
pip install flash-attn --no-build-isolation
```

> **Note**: Flash Attention installation can be tricky. If it fails, see the [official guide](https://github.com/Dao-AILab/flash-attention) or run local models without it (slower).

### API Keys

Set environment variables for API models:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

For persistent setup, add these to your `~/.bashrc` or `~/.zshrc`.

---

## Quick Start

### Evaluate a Model (API)

```bash
# Evaluate ChatGPT-5.1 on MMATH (English)
python scripts/11_eval_mmath_api.py \
  --model chatgpt-5.1 \
  --language en \
  --max_tokens 8192

# Evaluate on Chinese with limited examples
python scripts/11_eval_mmath_api.py \
  --model deepseek-v3.2 \
  --language zh \
  --limit 50
```

### Evaluate a Local Model

```bash
# Evaluate Qwen3-8B on GSM8K (Chinese)
python scripts/10_eval_gsm8k_local.py \
  --model qwen3-8b \
  --language zh \
  --max_tokens 1024

# Compare across token budgets
python scripts/10_eval_gsm8k_local.py \
  --model llama-3.1-8b \
  --language en \
  --sweep_tokens 128,256,512,1024,4096
```

### Analyze Token Efficiency

```bash
# Compare output token lengths across languages
python scripts/12_compare_token_lengths.py \
  --results_dir results/gsm8k \
  --model qwen3-8b

# Analyze tokenizer efficiency
python scripts/13_compare_token_lengths_mmath.py \
  --results_dir results/mmath \
  --model deepseek-v3.2
```

---

## Repository Structure

```
language-shapes-reasoning/
├── configs/                    # Configuration files
│   ├── models.yaml            # Model definitions (API keys, paths)
│   ├── datasets.yaml          # Dataset configurations
│   └── experiments.yaml       # Experiment settings & prompts
│
├── data/                      # Datasets
│   ├── raw/                   # Original datasets
│   │   ├── gsm8k/            # GSM8K train/test splits
│   │   └── mmath/            # MMATH multilingual problems
│   └── processed/             # Standardized JSONL format
│       ├── gsm8k_en_test.jsonl
│       ├── gsm8k_zh_test.jsonl
│       └── mmath_*.jsonl
│
├── scripts/                   # Executable scripts
│   ├── 00_download_data.py           # Download datasets from HuggingFace
│   ├── 01_process_gsm8k.py           # Process & translate GSM8K
│   ├── 02_process_mmath.py           # Process MMATH to standard format
│   ├── 10_eval_gsm8k_local.py        # Evaluate local models (GSM8K)
│   ├── 11_eval_mmath_api.py          # Evaluate API models (MMATH)
│   ├── 12_compare_token_lengths.py   # Token efficiency analysis
│   ├── 13_compare_token_lengths_mmath.py
│   ├── 20_aggregate_results.py       # Aggregate experiment results
│   ├── 30_post_training.py           # Fine-tune models (LoRA)
│   ├── 31_eval_finetuned_gsm8k.py    # Evaluate fine-tuned models
│   └── 40_plot_results.py            # Generate figures
│
├── src/                       # Core library code
│   ├── __init__.py
│   ├── data_utils.py          # JSONL I/O, directory utilities
│   ├── model_interface.py     # Model abstraction (BaseModel, LocalModel, etc.)
│   ├── eval_runner.py         # Evaluation loop & prompt templating
│   ├── parsing.py             # Answer extraction & correctness checking
│   └── prompts.py             # Prompt templates for different languages
│
├── results/                   # Experiment outputs
│   ├── gsm8k/                # CSV files with per-example predictions
│   ├── mmath/
│   ├── gsm8k_finetuned/      # Fine-tuned model results
│   └── token_analysis/       # Token efficiency statistics
│
├── figures/                   # Generated plots & visualizations
│   ├── en_vs_zh_gsm8k.png
│   ├── token_distribution_*.png
│   └── ...
│
├── outputs/                   # Fine-tuning outputs
│   └── sft/
│       ├── llama-3.1-8b/     # LoRA checkpoints
│       └── qwen3-8b/
│
├── reports/                   # Documentation & analysis
│   ├── experimental_setup.md
│   ├── results_analysis.md
│   └── ...
│
├── requirements.txt           # Python dependencies
├── final_blog.md             # Full research writeup
└── README.md                 # This file
```

---

## Experiments

### 1. Download & Process Data

```bash
# Download raw datasets from HuggingFace
python scripts/00_download_data.py

# Process GSM8K (includes translation to Chinese via GPT-5.1-mini)
python scripts/01_process_gsm8k.py

# Process MMATH to standardized format
python scripts/02_process_mmath.py
```

### 2. Evaluate SOTA Models (MMATH)

```bash
# Run all API models on all languages
python scripts/11_eval_mmath_api.py --all

# Or run individually
python scripts/11_eval_mmath_api.py --model chatgpt-5.1 --language en
python scripts/11_eval_mmath_api.py --model deepseek-v3.2 --language zh
python scripts/11_eval_mmath_api.py --model gemini-2.5-flash --language es
```

**Supported languages**: `en`, `zh`, `es`, `th`, `de`, `fr`, `ru`, `ja`

### 3. Evaluate Open-Source Models (GSM8K)

```bash
# Evaluate Qwen3-8B (multilingual)
python scripts/10_eval_gsm8k_local.py \
  --model qwen3-8b \
  --languages en,zh \
  --max_tokens 1024,4096

# Evaluate Llama-3.1-8B (English-centric)
python scripts/10_eval_gsm8k_local.py \
  --model llama-3.1-8b \
  --languages en,zh \
  --max_tokens 128,256,512,1024,4096
```

### 4. Fine-Tune & Evaluate

```bash
# Fine-tune Llama-3.1-8B on Chinese GSM8K (LoRA)
python scripts/30_post_training.py \
  --model llama-3.1-8b \
  --language zh \
  --num_steps 500 \
  --output_dir outputs/sft/llama-3.1-8b-ft-zh

# Evaluate fine-tuned model
python scripts/31_eval_finetuned_gsm8k.py \
  --model outputs/sft/llama-3.1-8b-ft-zh \
  --languages en,zh
```

### 5. Analyze Results

```bash
# Aggregate all results into summary tables
python scripts/20_aggregate_results.py

# Generate token efficiency analysis
python scripts/12_compare_token_lengths.py --results_dir results/gsm8k
python scripts/13_compare_token_lengths_mmath.py --results_dir results/mmath

# Create visualizations
python scripts/40_plot_results.py --output_dir figures/
```

---

## Configuration

### Model Configuration (`configs/models.yaml`)

Define models and their parameters:

```yaml
api_models:
  chatgpt-5.1:
    type: openai
    model_name: "gpt-5.1"
    api_key_env: "OPENAI_API_KEY"
    max_context: 128000

local_models:
  qwen3-8b:
    type: local
    model_path: "Qwen/Qwen3-8B"
    backend: transformers
    max_context: 8192
```

### Dataset Configuration (`configs/datasets.yaml`)

```yaml
gsm8k:
  source: "openai/grade-school-math"
  languages: [en, zh]
  answer_type: float
  
mmath:
  source: "multilingual-math"
  languages: [en, zh, es, th, de, fr, ru, ja]
  level: hard
```

### Experiment Configuration (`configs/experiments.yaml`)

Define prompt templates and evaluation settings:

```yaml
prompts:
  gsm8k_en_direct: |
    You are a careful math tutor. Solve the following problem step by step.
    At the end, output ONLY the final numeric answer on a line starting with:
    #### <number>
```

---

## Results & Analysis

### Data Formats

**Processed Dataset Format (JSONL)**:
```json
{
  "id": "gsm8k_test_000123",
  "source": "gsm8k",
  "language": "en",
  "question": "John has 5 apples...",
  "answer": "15",
  "answer_type": "int"
}
```

**Results Format (CSV)**:
```csv
id,model,dataset,language,mode,max_tokens,gold,pred,correct,raw_output_length,raw_output
gsm8k_test_001,qwen3-8b,gsm8k,en,direct,1024,42,42,1,523,"Let's solve this step by step..."
```

### Key Metrics

- **Accuracy**: Percentage of correct answers
- **Token Efficiency**: Average output tokens (Chinese/English ratio)
- **Character Efficiency**: Average character count per correct solution
- **Encoder Gap**: Discrepancy between intrinsic and realized density

### Figures

Generated visualizations in `figures/`:
- Accuracy comparison across languages
- Token length distributions
- Character length distributions
- Performance vs. token budget curves

---

## Extending the Pipeline

### Add a New Model

1. **Implement model class** in `src/model_interface.py`:

```python
class MyCustomModel(BaseModel):
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        # Your inference code here
        return response_text
```

2. **Add configuration** to `configs/models.yaml`:

```yaml
my_model:
  type: custom
  model_path: "path/to/model"
  max_context: 8192
```

3. **Register in evaluation script** (`scripts/10_eval_*.py`)

### Add a New Dataset

1. **Create processing script** in `scripts/0X_process_mydataset.py`
2. **Add configuration** to `configs/datasets.yaml`
3. **Add prompt templates** to `configs/experiments.yaml`
4. **Update evaluation scripts** to handle new dataset format

### Add a New Language

1. **Translate dataset** (use `scripts/01_process_gsm8k.py` as template)
2. **Add prompt templates** in `configs/experiments.yaml`
3. **Run evaluations** with `--language <code>`

---

## Deliverables

**Paper**: See `final_blog.md` for the complete research writeup.

---

## Contributors

**Bowen Yu, Linrui Ma, and Yiwei Liang**

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- GSM8K dataset: [OpenAI](https://github.com/openai/grade-school-math)
- MMATH benchmark: Multilingual Math Competition Problems
- Model providers: OpenAI, Google, DeepSeek, Meta, Alibaba
- Computing resources: MIT Supercloud

---

## Contact

For questions or collaborations:
- Open an issue on GitHub
- Email: {bowenyu, linrui, liangyw}@mit.edu
