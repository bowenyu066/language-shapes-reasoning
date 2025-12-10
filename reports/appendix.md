## Appendix

### A.1 Translation Quality Verification

The Chinese GSM8K dataset was created by translating the original English GSM8K using GPT-5.1-mini. To verify translation quality, we randomly sampled 100 examples and manually reviewed them for semantic fidelity, numerical accuracy, and grammatical correctness.

**Verification Criteria:**
- **Semantic fidelity**: The mathematical problem preserves its original meaning and all relevant details
- **Numerical accuracy**: All numbers, units, and mathematical relationships are correctly preserved
- **Grammatical correctness**: The Chinese translation is natural and grammatically correct

**Results:** Of the 100 sampled translations, 98 were deemed fully accurate, with 2 containing minor phrasing issues that did not affect problem solvability. No numerical errors were detected. We conclude that the translation quality is sufficient for our evaluation purposes.

---

### A.2 Full Accuracy Tables

#### Table A.1: MMATH Results by Model and Language

| Model | English | Chinese | Spanish | Japanese | Thai |
|-------|---------|---------|---------|----------|------|
| ChatGPT-5.1 | 79.9% (299/374) | 78.3% (293/374) | 77.8% (291/374) | 73.3% (274/374) | 70.9% (265/374) |
| DeepSeek-V3.2 | 88.5% (331/374) | 87.7% (328/374) | 88.5% (331/374) | 85.6% (320/374) | 85.0% (318/374) |
| Gemini-2.5 | 20.9% (78/374) | 80.7% (302/374) | 46.8% (175/374) | 46.5% (174/374) | 48.7% (182/374) |

#### Table A.2: GSM8K Results by Model, Language, and Max Tokens

**Llama-3.1-8B:**

| Max Tokens | English | Chinese | ZH (translate-then-solve) |
|------------|---------|---------|---------------------------|
| 128 | 4.6% (61/1319) | 6.8% (90/1319) | 0.2% (2/1319) |
| 256 | 59.9% (790/1319) | 49.8% (657/1319) | 39.0% (514/1319) |
| 512 | 80.4% (1060/1319) | 52.4% (691/1319) | 42.3% (558/1319) |
| 1024 | 79.6% (1050/1319) | 51.6% (680/1319) | 44.0% (580/1319) |
| 4096 | 80.3% (1059/1319) | 52.8% (697/1319) | 43.7% (576/1319) |

**Qwen3-8B:**

| Max Tokens | English | Chinese | ZH (translate-then-solve) |
|------------|---------|---------|---------------------------|
| 128 | 13.8% (182/1319) | 17.1% (226/1319) | 32.5% (429/1319) |
| 256 | 69.5% (917/1319) | 70.7% (932/1319) | 86.8% (1145/1319) |
| 512 | 88.1% (1162/1319) | 90.7% (1196/1319) | 90.4% (1193/1319) |
| 1024 | 88.5% (1167/1319) | 89.8% (1185/1319) | 90.1% (1188/1319) |
| 4096 | 88.7% (1170/1319) | 89.4% (1179/1319) | 89.8% (1185/1319) |

#### Table A.3: Fine-tuned Llama-3.1-8B Results

| Model | English | Chinese | ZH (translate-then-solve) |
|-------|---------|---------|---------------------------|
| Llama (base) | 80.3% (1059/1319) | 52.8% (697/1319) | 43.7% (576/1319) |
| Llama (ft-zh) | 81.1% (1069/1319) | 61.6% (812/1319) | 53.5% (705/1319) |

---

### A.3 Token Length Analysis

#### Table A.4: ZH/EN Token Ratio by Tokenizer (GSM8K, Both Correct Only)

| Tokenizer | Samples | Avg EN Tokens | Avg ZH Tokens | ZH/EN Ratio |
|-----------|---------|---------------|---------------|-------------|
| Qwen3-8B | 1,088 | 308.4 | 296.4 | 0.961 |
| DeepSeek-V3 | 1,088 | 288.4 | 258.9 | 0.898 |
| Seed-Coder-8B | 1,088 | 325.0 | 303.1 | 0.933 |
| Llama-3.1-8B | 1,088 | 294.1 | 303.7 | 1.033 |
| GPT-4o | 1,088 | 292.9 | 297.0 | 1.014 |

#### Table A.5: ZH/EN Token Ratio by Tokenizer (MMATH, Both Correct Only)

| Tokenizer | Samples | Avg EN Tokens | Avg ZH Tokens | ZH/EN Ratio |
|-----------|---------|---------------|---------------|-------------|
| Qwen3-8B | 321 | 904.9 | 847.9 | 0.937 |
| DeepSeek-V3 | 321 | 823.6 | 748.3 | 0.909 |
| Seed-Coder-8B | 321 | 941.5 | 858.6 | 0.912 |
| Llama-3.1-8B | 321 | 860.1 | 813.3 | 0.946 |
| GPT-4o | 321 | 866.2 | 811.5 | 0.937 |

*Note: Token ratios below 1.0 indicate Chinese uses fewer tokens than English for equivalent reasoning.*

---

### A.4 Fine-tuning Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | meta-llama/Llama-3.1-8B-Instruct |
| Adaptation Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank (r) | 32 |
| LoRA Alpha (α) | 64 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 1e-5 |
| Max Steps | 500 |
| Batch Size | 1 |
| Gradient Accumulation Steps | 2 |
| Effective Batch Size | 2 |
| Precision | bfloat16 |
| Max Prompt Length | 512 tokens |
| Max Completion Length | 4096 tokens |
| Training Data | Chinese GSM8K train split (7,473 examples) |

---

### A.5 Output Length Distributions

#### Figure A.1: MMATH Output Length Distribution by Language (ChatGPT-5.1)

[INSERT FIGURE: `results/mmath/length_distribution_chatgpt.pdf`]

*Distribution of raw output lengths (characters) for ChatGPT-5.1 across languages on MMATH. Chinese outputs (red) show a left-shifted distribution compared to English (blue), indicating more concise responses.*

---

#### Figure A.2: MMATH Output Length Distribution by Language (DeepSeek-V3.2)

[INSERT FIGURE: `results/mmath/length_distribution_deepseek.pdf`]

*Distribution of raw output lengths for DeepSeek-V3.2. The pattern is consistent with ChatGPT-5.1: Chinese outputs are more compact while maintaining equivalent accuracy.*

---

#### Figure A.3: GSM8K Output Length Distribution (Qwen3-8B)

[INSERT FIGURE: `results/gsm8k/length_distribution_qwen.pdf`]

*Token length distribution for Qwen3-8B on GSM8K. Chinese outputs (red) cluster at lower token counts than English (blue), demonstrating representation efficiency in a model with balanced multilingual training.*

---

#### Figure A.4: GSM8K Output Length Distribution (Llama-3.1-8B)

[INSERT FIGURE: `results/gsm8k/length_distribution_llama.pdf`]

*Token length distribution for Llama-3.1-8B on GSM8K. Unlike Qwen, the distributions overlap substantially, reflecting Llama's less efficient Chinese tokenization due to English-centric training.*

---

### A.6 Accuracy vs. Max Tokens

#### Figure A.5: Accuracy vs. Token Budget (GSM8K)

[INSERT FIGURE: `results/gsm8k/en_vs_zh_vs_zh_translate_then_solve.pdf`]

*Accuracy as a function of maximum token budget for Llama-3.1-8B and Qwen3-8B on GSM8K. Qwen achieves near-peak accuracy by 512 tokens in both languages, while Llama's Chinese accuracy plateaus well below its English ceiling regardless of token budget.*

---

### A.7 Cross-Language Accuracy Comparison

#### Figure A.6: MMATH Accuracy Across Languages

[INSERT FIGURE: `results/mmath/en_vs_zh_vs_es_vs_th.pdf`]

*Accuracy comparison across English, Chinese, Spanish, and Thai for SOTA models on MMATH. Well-trained models (ChatGPT-5.1, DeepSeek-V3.2) exhibit representation-invariant performance with gaps under 10% across languages.*

---

#### Figure A.7: MMATH English vs. Chinese Accuracy

[INSERT FIGURE: `results/mmath/en_vs_zh.pdf`]

*Direct comparison of English and Chinese accuracy on MMATH. Points near the diagonal indicate representation-invariant performance; deviation below the diagonal indicates Chinese underperformance.*
