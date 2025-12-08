# Multilingual Math Reasoning Evaluation Pipeline

A modular evaluation pipeline for assessing multilingual math reasoning capabilities of language models.

## Overview

This repository evaluates:
- **Open-source models** (Qwen3-8B, Llama-3.1-8B, DeepSeekMath-7B) on GSM8K (English + Chinese)
- **Frontier API models** (ChatGPT-5.1, Gemini-3, DeepSeek-R1) on MMATH (multilingual hard math)

## Repository Structure

```
language-shapes-reasoning/
├── data/
│   ├── raw/                     # Original datasets
│   │   ├── gsm8k_train.jsonl
│   │   ├── gsm8k_test.jsonl
│   │   └── mmath_raw.jsonl
│   └── processed/               # Standardized format
│       ├── gsm8k_en_test.jsonl
│       ├── gsm8k_zh_test.jsonl
│       └── mmath_multilingual.jsonl
├── configs/
│   ├── models.yaml              # Model configurations
│   ├── datasets.yaml            # Dataset configurations
│   └── experiments.yaml         # Experiment settings & prompts
├── scripts/
│   ├── 00_download_data.py      # Download raw datasets
│   ├── 01_translate_gsm8k_zh.py # Translate GSM8K to Chinese
│   ├── 02_process_mmath.py      # Process MMATH to standard format
│   ├── 10_eval_gsm8k_local.py   # Evaluate local models on GSM8K
│   ├── 11_eval_mmath_api.py     # Evaluate API models on MMATH
│   └── 20_aggregate_results.py  # Aggregate all results
├── src/
│   ├── data_utils.py            # JSONL loading/saving utilities
│   ├── translate_utils.py       # Translation helper functions
│   ├── model_interface.py       # Model abstraction layer
│   ├── eval_runner.py           # Evaluation logic
│   └── parsing.py               # Answer extraction & comparison
├── results/
│   ├── gsm8k/                   # GSM8K evaluation results
│   └── mmath/                   # MMATH evaluation results
└── requirements.txt
```

## Installation

```bash
# Create virtual environment (recommended)
conda create -n lang-shapes-reasoning python=3.12
conda activate lang-shapes-reasoning

# Install dependencies
pip install -r requirements.txt

# Other than the dependencies in requirements.txt, several additional packages may be required for API models (e.g., Gemini-3) and local models
pip install -q -U google-genai                  # for Gemini-3
pip install flash-attn --no-build-isolation     # for local models
```

> Note: you may encounter some issues with the installation of flash-attn. If so, you can try to install it from source: https://github.com/Dao-AILab/flash-attention

## Configuration

### API Keys

Set environment variables for API models:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### Model Configuration

Edit `configs/models.yaml` to update model paths or API endpoints.

## Usage

### 1. Download Data

```bash
python scripts/00_download_data.py
```

This creates placeholder data. Replace with full datasets from:
- GSM8K: https://github.com/openai/grade-school-math
- MMATH: (your source)

### 2. Process Data

```bash
# Translate GSM8K to Chinese
python scripts/01_translate_gsm8k_zh.py

# Process MMATH to standard format
python scripts/02_process_mmath.py
```

### 3. Run Evaluations

#### Local Models (GSM8K)

```bash
# Run all experiments
python scripts/10_eval_gsm8k_local.py

# Run specific configuration
python scripts/10_eval_gsm8k_local.py --model qwen3-8b --language en --limit 10
```

#### API Models (MMATH)

```bash
# Run all experiments
python scripts/11_eval_mmath_api.py

# Run specific configuration
python scripts/11_eval_mmath_api.py --model chatgpt-5.1 --language zh --limit 10
```

### 4. Aggregate Results

```bash
python scripts/20_aggregate_results.py
```

Results are saved to `results/summary.csv`.

## Data Formats

### GSM8K Processed Format

```json
{
  "id": "gsm8k_test_000123",
  "source": "gsm8k",
  "split": "test",
  "level": "gsm8k",
  "language": "en",
  "question": "...",
  "answer": "4.5",
  "answer_type": "float",
  "meta": {"original_index": 123}
}
```

### MMATH Processed Format

```json
{
  "id": "mmath_000045_en",
  "problem_id": "000045",
  "source": "mmath",
  "level": "hard",
  "language": "en",
  "question": "...",
  "answer": "17",
  "answer_type": "int",
  "meta": {"subdomain": "algebra", "difficulty": "high"}
}
```

## Extending the Pipeline

### Adding a New Model

1. Implement a class inheriting from `BaseModel` in `src/model_interface.py`
2. Add configuration to `configs/models.yaml`
3. Register in the appropriate evaluation script

### Adding a New Dataset

1. Add configuration to `configs/datasets.yaml`
2. Create a processing script in `scripts/`
3. Add prompt templates to `src/eval_runner.py` or `configs/experiments.yaml`

### Adding New Prompt Variants

Edit `PROMPT_TEMPLATES` in `src/eval_runner.py` or `configs/experiments.yaml`.

## Output Format

Per-example results (CSV):
- `id`: Problem ID
- `model`: Model name
- `dataset`: Dataset name
- `language`: Language code
- `mode`: Evaluation mode (direct, zh_translate_then_solve)
- `max_tokens`: Max generation tokens
- `gold`: Gold answer
- `pred`: Predicted answer
- `correct`: 1 if correct, 0 otherwise
- `raw_output_length`: Length of raw model output
- `raw_output`: Full model response

## TODO

- [ ] Implement HuggingFace transformers integration for local models
- [ ] Add Ollama backend support
- [ ] Implement real translation API calls
- [ ] Add more languages for MMATH
- [ ] Add confidence intervals for accuracy metrics
- [ ] Implement caching for API calls

## License

MIT License
