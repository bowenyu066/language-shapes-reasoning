## Experimental Setup

We design three experiments to progressively investigate how representation density affects reasoning quality: evaluating state-of-the-art models for baseline representation invariance, comparing models with different training distributions to reveal potential bottlenecks, and fine-tuning to test whether such bottlenecks are learnable.

### Datasets

We use two mathematical reasoning benchmarks that provide semantically equivalent problems across languages. **MMATH** is a multilingual benchmark comprising 374 competition-level mathematics problems sourced from AIME and CNMO, professionally translated into 10 languages including English, Chinese, Japanese, Spanish, and Thai. The difficulty level ensures that problems require multi-step reasoning rather than pattern matching. **GSM8K** consists of 1,319 grade-school math word problems in its test set. Since no official Chinese version exists, we translated the dataset using GPT-5.1-mini and manually verified 100 randomly sampled examples to confirm translation fidelity. The training split (7,473 examples) serves as the basis for our fine-tuning experiments.

### Models

Our experiments span three categories of models. For state-of-the-art evaluation, we use frontier API models: ChatGPT-5.1, Gemini-2.5-Flash, and DeepSeek-V3.2. These models represent the current capability frontier and have been trained on massive multilingual corpora, making them suitable for testing whether well-trained models achieve representation-invariant reasoning.

For controlled comparison, we evaluate two open-source 8B-parameter models with contrasting training distributions: Qwen3-8B, which emphasizes multilingual capability, and Llama-3.1-8B-Instruct, which is predominantly English-centric. This pairing allows us to isolate the effect of training distribution while controlling for model scale.

For fine-tuning experiments, we adapt Llama-3.1-8B on the Chinese GSM8K training set using Low-Rank Adaptation (LoRA) with rank 32 and $\alpha=64$, targeting all attention and MLP projection matrices. Training proceeds for 500 steps with learning rate $10^{-5}$ and gradient accumulation over 2 steps. This lightweight adaptation tests whether representation bottlenecks can be overcome with modest additional training.

### Evaluation Protocol

All models receive problems formatted with language-appropriate prompts instructing step-by-step reasoning and a final numerical answer in the format `#### <number>`. We extract answers via regex matching and evaluate exact numerical correctness.

For SOTA models on MMATH, we use a maximum token budget of 8,192 and evaluate across five languages (English, Chinese, Japanese, Spanish, Thai). For open-source models on GSM8K, we sweep maximum token budgets from 128 to 4,096 to analyze accuracy as a function of generation length, revealing how quickly models converge to their peak performance in each representation.

### Token Length Analysis

To measure representation efficiency independent of any single tokenizer's biases, we tokenize model outputs using five different tokenizers: Qwen3-8B, DeepSeek-V3, Seed-Coder-8B, Llama-3.1-8B, and GPT-4o (via tiktoken). For fair comparison, we restrict analysis to problems where the model answered correctly in both English and Chinese, ensuring we compare token counts for successful reasoning chains rather than failed attempts. We report the ratio of Chinese to English token counts, where values below 1.0 indicate that Chinese uses fewer tokens for equivalent reasoning.
