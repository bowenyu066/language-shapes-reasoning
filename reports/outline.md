# Report Outline: Language as Representation — Token Efficiency and Reasoning in Multilingual LLMs

## Central Thesis

Language choice is a **representation design variable** that affects token efficiency without necessarily degrading reasoning quality—provided the model's architecture and training are aligned with the representation. We demonstrate this through multilingual math reasoning experiments, showing that well-trained models achieve representation-invariant accuracy while denser languages (e.g., Chinese) offer genuine token savings.

---

## 1. Introduction (Existing — ~500 words)

*Already written in `reports/introduction.md`*

- The quadratic cost of attention creates tension between sequence length and efficiency
- Existing work focuses on efficient processing of fixed representations
- Overlooked degree of freedom: **representation choice itself**
- Multilingual LLMs as a natural experimental proxy for representation density
- Research question: Can denser representations (fewer tokens) maintain reasoning quality?

---

## 2. Related Work (~300-400 words)

### 2.1 Efficient Transformers
- Sparse attention, learned compression tokens, efficient approximations
- Common assumption: representation is fixed, efficiency comes from processing

### 2.2 Multilingual Language Models
- Training on diverse languages enables cross-lingual transfer
- Prior work on multilingual reasoning benchmarks (MGSM, MMATH)
- Gap: representation efficiency perspective largely unexplored

### 2.3 Tokenization and Information Density
- Different tokenizers yield different token counts for equivalent semantics
- Language-specific tokenization strategies (BPE, character-level for CJK)
- Connection to our work: natural density variation across languages

---

## 3. Experimental Setup (~400-500 words)

### 3.1 Datasets
- **MMATH**: Multilingual math benchmark (English, Chinese, Spanish, Japanese, Thai)
- **GSM8K**: Grade school math, English original + GPT-5.1-mini Chinese translation
  - Translation quality: manually verified 100 examples for accuracy

### 3.2 Models
- **SOTA API Models**: ChatGPT-5.1, Gemini-2.5, DeepSeek-V3.2
- **Open-Source Models**: Llama-3.1-8B-Instruct, Qwen3-8B
- **Fine-tuned Models**: Llama-3.1-8B fine-tuned on Chinese GSM8K (LoRA)

### 3.3 Evaluation Metrics
- Accuracy (exact match on final numerical answer)
- Raw output length (characters)
- Token count (measured across 5 tokenizers: Qwen, DeepSeek, Seed, Llama, GPT-4o)
- Accuracy vs. max token budget curves

### 3.4 Tokenizers for Cross-Analysis
- Rationale: Different tokenizers have different multilingual efficiencies
- Measuring ZH/EN token ratio reveals representation density independent of model

---

## 4. Results and Analysis (~1200-1500 words)

### 4.1 SOTA Models: Representation-Invariant Accuracy (Experiment 1)

**Hypothesis**: Well-trained SOTA models should achieve similar accuracy across languages on equivalent mathematical content.

**Findings**:
- ChatGPT-5.1: EN 79.9%, ZH 78.3% — negligible gap (~1.6%)
- Similar patterns for Gemini-2.5 and DeepSeek-V3.2
- Accuracy is **representation-invariant** for sufficiently trained models

**Token Efficiency Analysis**:
- Average raw output length: EN ~1282 chars, ZH ~1155 chars (10% reduction)
- Token analysis (DeepSeek tokenizer): ZH/EN ratio ≈ 0.91 (9% token savings)
- **Key insight**: Same reasoning quality, fewer tokens

**Interpretation**: This confirms the introduction's premise—representation choice affects token budget without degrading task performance when models are adequately trained.

### 4.2 Open-Source Models: Revealing Architectural Bottlenecks (Experiment 2)

**Hypothesis**: Less comprehensively trained models may exhibit representation-dependent performance, revealing architectural or training bottlenecks.

**Findings — Qwen3-8B** (multilingual-focused training):
- EN accuracy: 88.7%, ZH accuracy: 89.4% (4096 max tokens)
- Representation-invariant, consistent with SOTA models
- ZH token distribution shifted left (shorter outputs)

**Findings — Llama-3.1-8B** (English-centric training):
- EN accuracy: 80.3%, ZH accuracy: 52.8% — **27.5% gap**
- Significant degradation on Chinese representation
- Reveals **representation-architecture misalignment**

**Token Length Analysis**:
- Qwen: ZH/EN token ratio ≈ 0.96 (Qwen tokenizer), ≈ 0.90 (DeepSeek tokenizer)
- Llama: ZH/EN ratio > 1.0 on Llama tokenizer (Chinese less efficient with English-centric tokenizer)

**Interpretation**: The performance gap is not about language difficulty (Qwen succeeds) but about training distribution. Llama's architecture isn't bottlenecked—its training is.

### 4.3 Accuracy vs. Token Budget Curves

**Analysis**: How does accuracy scale with max token allowance?

- Qwen (EN): 128 tok → 13.8%, 256 → 69.5%, 512 → 88.1%, 1024+ → 88.5%
- Qwen (ZH): 128 tok → 17.1%, 256 → 70.7%, 512 → 90.7%, 1024+ → 89.4%
- **Chinese reaches peak accuracy at lower token budgets** due to denser representation

**Implication**: Token efficiency translates to computational savings—Chinese reasoning completes in fewer tokens.

### 4.4 Fine-Tuning Bridges the Gap (Experiment 3)

**Hypothesis**: If the bottleneck is training distribution, fine-tuning on the target representation should recover performance.

**Setup**: Fine-tune Llama-3.1-8B on Chinese GSM8K using LoRA (500 steps)

**Results**:
| Model | EN Accuracy | ZH Accuracy |
|-------|-------------|-------------|
| Llama (base) | 80.3% | 52.8% |
| Llama (ft-zh) | 81.1% | 61.6% |

- Chinese accuracy improved by **+8.8%** (52.8% → 61.6%)
- English accuracy maintained (slight improvement)
- Gap reduced from 27.5% to 19.5%

**Interpretation**:
- The representation bottleneck is **learnable**, not architectural
- Modest fine-tuning enables reasoning in denser representations
- Suggests a path to efficient multilingual models: train on diverse representations

---

## 5. Discussion (~400-500 words)

### 5.1 Representation as a Design Variable
- Our findings support treating representation choice as a first-class design decision
- Token efficiency gains (5-10%) compound significantly at scale
- Well-trained models already exhibit representation invariance

### 5.2 Implications for Efficient Inference
- Chinese reasoning uses ~90-95% of English token count for equivalent accuracy
- Potential for language-aware routing: prefer denser representations when available
- Connection to compression research: natural languages as learned codebooks

### 5.3 Training Distribution vs. Architecture
- Llama's Chinese gap reflects training bias, not architectural limitation
- Fine-tuning demonstrates adaptability to new representations
- Suggests multilingual pretraining is crucial for representation-agnostic reasoning

### 5.4 Limitations
- Limited to mathematical reasoning (well-defined correctness criterion)
- Translation quality may introduce subtle biases
- Fine-tuning experiment limited to one model and one direction

---

## 6. Conclusion (~200-300 words)

- Revisit central thesis: representation choice matters for efficiency, not necessarily for quality
- Summary of key findings:
  1. SOTA models achieve representation-invariant accuracy
  2. Under-trained models reveal bottlenecks that are learnable, not fundamental
  3. Denser representations offer genuine token savings
- Future directions:
  - Extend to other reasoning domains (code, logical reasoning)
  - Study representation-aware training curricula
  - Investigate optimal tokenization strategies for efficiency

---

## Appendix

- A.1: Translation quality verification methodology
- A.2: Full accuracy tables across all configurations
- A.3: Token length distribution plots
- A.4: Fine-tuning hyperparameters and training curves

---

## Key Figures to Include

1. **Accuracy comparison bar chart**: EN vs ZH across all models (MMATH)
2. **Output length distributions**: Overlaid histograms by language (per model)
3. **Accuracy vs. max tokens curves**: Qwen and Llama on GSM8K
4. **Token ratio heatmap**: ZH/EN ratio across tokenizers and datasets
5. **Fine-tuning results**: Before/after accuracy comparison for Llama

---

## Suggested Word Distribution

| Section | Target Words |
|---------|-------------|
| Introduction | 500 (existing) |
| Related Work | 350 |
| Experimental Setup | 450 |
| Results and Analysis | 1400 |
| Discussion | 450 |
| Conclusion | 250 |
| **Total** | **~3400** |
