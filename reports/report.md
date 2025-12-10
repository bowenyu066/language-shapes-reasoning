# Language as Representation — Token Efficiency and Reasoning in Multilingual LLMs

## Introduction

Sequence length is a critical bottleneck in modern deep learning. Across domains—from autoregressive language models to vision transformers to protein structure predictors—the Transformer architecture has become the dominant paradigm, yet its core self-attention mechanism scales quadratically with sequence length, $O(n^2 d)$. This computational constraint forces practitioners into a fundamental trade-off: longer sequences enable richer representations and more complex reasoning, but at rapidly escalating cost in memory and compute.

This tension has motivated an extensive body of research aimed at a single objective: *processing representations more efficiently*. Sparse attention patterns reduce the number of pairwise interactions. Learned compression tokens distill long contexts into fixed-size summaries. Linear attention approximations replace quadratic operations with more tractable alternatives. In vision, patch pruning and latent space compression serve analogous goals. These methods share a common assumption—that the input representation is fixed, and efficiency must be achieved through algorithmic innovation in how we process it.

Yet this framing neglects a more fundamental degree of freedom: *the representation itself*. The same underlying information can be encoded in vastly different ways, each yielding different sequence lengths. An image can be tokenized into 256 patches or 1024. A protein sequence can be represented at the residue level or compressed into structural motifs. A paragraph of text might occupy 100 tokens in one encoding scheme and 60 in another. The choice of representation—not merely the method of processing—determines the token budget required for a given task.

This observation raises a central question for deep learning systems: **Does task performance depend on the representation, or only on the information content it encodes?** If a model can reason equally well over a 60-token representation as a 100-token one carrying the same information, then representation design becomes a lever for efficiency gains that compound with scale. Conversely, if compressed representations degrade performance, then the efficiency-quality trade-off must be carefully navigated.

This trade-off has been studied extensively in some domains. In computer vision, aggressive compression of latent spaces degrades reconstruction quality in predictable ways, and the rate-distortion curve is well characterized. In learned image compression, the relationship between bitrate and perceptual quality follows established information-theoretic principles. However, in the context of *reasoning*—where the goal is not reconstruction but inference over semantic content—this trade-off remains poorly understood. Can a model solve a math problem equally well when the problem statement is encoded in fewer tokens? Does the density of a representation impose limits on the complexity of reasoning it can support?

To investigate these questions empirically, we require a setting where (1) the same semantic content can be naturally expressed in representations of different lengths, (2) task performance can be precisely measured, and (3) we can isolate representation effects from confounding factors. Natural language provides an ideal experimental substrate. Human languages encode equivalent semantic content with dramatically different token efficiencies. Consider the sentence: "The quick brown fox jumps over the lazy dog." In English, a typical tokenizer produces roughly 10 tokens. The Chinese translation ("快速的棕色狐狸跳过懒狗") compresses to 4–6 tokens while preserving semantic content. This variation is not artificial—it reflects genuine differences in how information is encoded across writing systems and linguistic structures.

Crucially, we frame this not as a study of multilingualism per se, but as an investigation into **representation efficiency and its effect on model reasoning**. Languages serve as naturally occurring, semantically-aligned representations of varying density. By evaluating models on equivalent mathematical problems expressed in different languages, we can measure whether denser representations (fewer tokens) maintain reasoning quality—or whether they expose architectural bottlenecks that degrade performance.

Our experimental design targets three progressively refined questions:

1. **Do well-trained models exhibit representation-invariant reasoning?** We evaluate state-of-the-art models on multilingual mathematical benchmarks to test whether accuracy depends on the language of presentation—i.e., the representation—or only on the underlying mathematical content.

2. **What happens when models are less comprehensively trained?** By comparing models with different training distributions (multilingual vs. English-centric), we can identify whether performance gaps reflect fundamental architectural limitations or merely training data coverage.

3. **Are representation bottlenecks learnable?** If a model struggles with a particular representation, can targeted fine-tuning recover performance? This tests whether the bottleneck is intrinsic to the architecture or an artifact of training.

Our findings reveal a consistent pattern: representation choice affects token efficiency but need not degrade reasoning quality—provided the model has been adequately trained on the target representation. State-of-the-art models achieve near-identical accuracy across languages while using 5–10% fewer tokens in denser representations like Chinese. Models with narrower training distributions exhibit significant performance gaps, but these gaps can be substantially reduced through modest fine-tuning. Together, these results suggest that representation efficiency is not a trade-off to accept, but a capability to unlock through appropriate training.

The implications extend beyond multilingual modeling. If reasoning quality is invariant to representation density for well-trained models, then representation design becomes a first-class optimization target. Rather than engineering ever-more-efficient attention mechanisms to process fixed-length inputs, we might instead invest in representations that encode the same information in fewer tokens. Natural languages offer one such family of representations; learned compression schemes, domain-specific tokenizations, and structured encodings offer others. Our work provides empirical grounding for this perspective and a methodology for evaluating representation efficiency in reasoning tasks.

## Related Work

The challenge of efficient sequence processing in deep learning has attracted sustained attention, with two complementary research directions emerging: optimizing how models process fixed representations, and designing more compact representations that preserve task-relevant information.

### Efficient Processing of Fixed Representations

The quadratic complexity of self-attention has motivated numerous architectural innovations aimed at reducing computational cost while maintaining model expressiveness. Sparse attention mechanisms, exemplified by Longformer [^1] and BigBird [^2], replace full pairwise attention with structured patterns combining local windows, global tokens, and random connections. These approaches reduce complexity from $O(n^2)$ to $O(n)$ while preserving theoretical expressiveness—BigBird notably maintains Turing completeness despite its sparse structure. Empirically, such methods extend context length by 4–8× on equivalent hardware, enabling significant improvements on long-document tasks [^2].

Beyond static sparsity patterns, dynamic token reduction methods have gained traction, particularly in vision transformers. Token Merging (ToMe) [^3] progressively combines similar tokens across transformer layers, achieving substantial speedups without task-specific fine-tuning. Joint Token Pruning and Squeezing [^4] extends this idea by aggregating information from pruned tokens rather than discarding it entirely. These methods demonstrate that not all tokens contribute equally to final predictions—a 95% reduction in token count can degrade accuracy by less than 1% in some settings [^5]. Linear attention variants offer an alternative path, replacing the softmax operation with kernel approximations that permit $O(n)$ complexity through associative matrix multiplication [^6].

A common thread unites these approaches: they treat the input representation as given and seek algorithmic efficiency in processing it. While effective, this framing leaves unexplored the possibility that the representation itself might be redesigned for efficiency.

### Compact Representations Across Domains

An orthogonal line of research asks whether the same information can be encoded more compactly before processing begins. Autoencoders and their variational extensions (VAEs) [^7] learn to compress high-dimensional inputs into low-dimensional latent spaces, with reconstruction quality degrading gracefully as dimensionality decreases. This rate-distortion trade-off is well characterized: aggressive compression reduces bitrate but introduces reconstruction error following predictable information-theoretic curves [^8].

In learned image compression, neural networks now rival or exceed classical codecs by learning latent representations optimized for reconstruction fidelity [^9]. Recent work achieves compression rates of 10–20× while maintaining perceptual quality, substantially outperforming traditional methods like pruning and quantization [^10]. The key insight is that learned representations can exploit statistical structure in data more effectively than hand-designed encodings.

However, these compression studies primarily target reconstruction tasks, where the goal is faithful recovery of the original input. Whether compact representations preserve the information needed for *reasoning*—where models must perform inference rather than reconstruction—remains less understood. Our work addresses this gap by treating natural languages as a family of representations with inherently different information densities, allowing us to study how representation compactness affects reasoning performance without engineering artificial compression schemes.

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

## Discussion

Our findings offer a new perspective on efficiency in deep learning: rather than focusing exclusively on architectural innovations to process fixed representations more efficiently, we can achieve meaningful gains by choosing representations that encode the same information in fewer tokens. We discuss the implications of this perspective, its connections to broader research themes, and the limitations of our study.

### Representation as a First-Class Design Variable

The dominant paradigm for efficient deep learning treats the input representation as fixed and seeks savings through algorithmic innovation—sparse attention, token pruning, linear approximations. Our results suggest a complementary approach: optimizing the representation itself. When SOTA models achieve equivalent accuracy on Chinese and English while using 5–9% fewer tokens in Chinese, they demonstrate that representation choice can yield efficiency gains without quality degradation.

These savings are not trivial at scale. A 9% reduction in token count translates to approximately 17% reduction in attention computation due to quadratic scaling. For large-scale deployments processing billions of tokens daily, such gains compound into substantial resource savings. Moreover, unlike architectural modifications that may require retraining or introduce approximation errors, representation choice operates at the input level and is fully compatible with existing model infrastructure.

The key insight is that efficiency and quality need not trade off when the model has been adequately trained on the target representation. This reframes the question from "how do we process long sequences efficiently?" to "how do we encode information compactly while preserving what models need to reason?" Natural languages offer one answer—Chinese encodes mathematical content more densely than English—but the principle extends to any domain where multiple representations exist for the same underlying information.

### Implications for Efficient Inference

Our token efficiency analysis reveals consistent patterns across multiple tokenizers: Chinese reasoning chains require 90–95% of the tokens needed for equivalent English chains. This suggests practical strategies for efficient inference. When a problem can be expressed in multiple representations, preferring the denser one reduces computational cost without sacrificing accuracy—at least for well-trained models.

One might envision language-aware routing systems that translate inputs into denser representations before processing, then translate outputs back. Whether the overhead of translation outweighs the savings from reduced token counts is an empirical question that depends on translation quality, model size, and sequence length. Our results establish that the reasoning component itself benefits from denser representations; integrating translation into an end-to-end efficiency pipeline remains future work.

More broadly, natural languages can be viewed as learned codebooks that compress semantic content with varying efficiency. Chinese characters encode more information per token than English words—not through any magical property, but through different granularity choices in how meaning maps to symbols. This perspective connects our work to learned compression research, where neural networks discover efficient codes for specific data distributions. Languages represent humanity's millennia-long optimization of communication efficiency under cognitive constraints; modern tokenizers inherit and extend this optimization.

### Training Distribution vs. Architecture

A central finding of our work is that representation capability reflects training distribution rather than architectural limitation. Llama-3.1-8B's 27.5% accuracy gap on Chinese versus English could be mistaken for evidence that Chinese is inherently harder for transformer architectures—but Qwen3-8B's equivalent performance on both languages disproves this interpretation. The difference lies in training data composition, not model architecture.

This distinction matters for practitioners. An architectural limitation would require fundamental changes to model design; a training distribution gap can be addressed through data. Our fine-tuning experiment demonstrates this concretely: 500 steps of LoRA adaptation on Chinese data improves Llama's Chinese accuracy by 8.8 percentage points while maintaining English performance. The bottleneck is learnable.

This finding suggests that multilingual pretraining is valuable not merely for serving diverse user populations, but for unlocking representation-efficient inference. A model trained on diverse representations can exploit whichever encoding is most efficient for a given input. Conversely, models trained predominantly on one representation forfeit efficiency gains available in others.

### Limitations

Our study has several limitations that qualify the generality of our conclusions.

First, we focus exclusively on mathematical reasoning, where correctness can be precisely measured and semantic equivalence across languages is well-defined. Whether our findings extend to open-ended generation, subjective evaluation, or domains where translation introduces ambiguity remains to be tested. Mathematical problems have the virtue of clear ground truth; they may not represent the full complexity of language understanding.

Second, our Chinese datasets are either professionally translated (MMATH) or machine-translated with manual verification (GSM8K). Translation quality affects the validity of cross-lingual comparisons. While we verified 100 GSM8K translations manually and found high fidelity, subtle biases may persist. Perfect semantic equivalence across languages is an idealization that real translations approximate.

Third, our fine-tuning experiment examines only one model (Llama-3.1-8B) and one adaptation direction (English → Chinese). Whether similar gains obtain for other models or other language pairs is an empirical question. The 8.8% improvement we observe may reflect the specific characteristics of Llama's training distribution rather than a general principle. Broader experiments across models and languages would strengthen our conclusions.

Finally, we measure token counts using existing tokenizers rather than proposing new tokenization schemes. Our analysis reveals efficiency differences across languages given current tokenizers, but does not address whether better tokenizers could narrow or widen these gaps. Tokenizer design is itself a rich area for efficiency optimization that our work does not explore.

Despite these limitations, our core finding—that representation choice affects efficiency without necessarily degrading quality—rests on consistent evidence across multiple models, datasets, and tokenizers. The specific numbers may vary in other settings, but the qualitative conclusion appears robust.

## Conclusion

We set out to investigate whether representation choice affects reasoning quality, or only the token budget required to achieve it. Using natural languages as semantically-aligned representations of varying density, we evaluated mathematical reasoning across multiple models, languages, and experimental conditions. Our findings consistently support the thesis that **representation efficiency and reasoning quality are separable concerns**—denser representations reduce token counts without degrading accuracy, provided models have been adequately trained.

Three key results emerge from our experiments. First, state-of-the-art models exhibit representation-invariant reasoning: ChatGPT-5.1 and DeepSeek-V3.2 achieve near-identical accuracy across English and Chinese (gaps under 2%) while Chinese outputs require 5–9% fewer tokens. The same mathematical content, encoded more compactly, yields equivalent reasoning at reduced computational cost. Second, representation capability reflects training distribution rather than architectural limitation. Llama-3.1-8B's 27.5% accuracy gap on Chinese versus English disappears in Qwen3-8B, which was trained on more balanced multilingual data. The bottleneck is not intrinsic to the representation or the architecture—it is an artifact of training. Third, representation bottlenecks are learnable. Modest fine-tuning (500 steps of LoRA) improves Llama's Chinese accuracy by 8.8 percentage points while preserving English performance, demonstrating that models can acquire new representation capabilities without catastrophic forgetting.

These findings reframe efficiency in deep learning. The dominant approach—optimizing how we process fixed representations through sparse attention, token pruning, or linear approximations—addresses only half the problem. Our results suggest a complementary strategy: optimizing the representation itself. When denser encodings preserve reasoning quality, choosing them yields efficiency gains that compound with scale and require no architectural modification.

Several directions merit future investigation. Extending our analysis beyond mathematical reasoning to code generation, logical inference, and open-ended tasks would test the generality of representation invariance. Studying whether models can learn to translate inputs into more efficient representations before reasoning—and whether this end-to-end pipeline yields net efficiency gains—could yield practical inference optimizations. Finally, investigating tokenizer design through the lens of representation efficiency may reveal opportunities to learn encodings that are simultaneously compact and reasoning-friendly.

More broadly, our work contributes to a shift in perspective: from treating tokenization as a preprocessing detail to recognizing representation choice as a first-class design variable. Natural languages offer one family of representations with varying efficiency; learned compression schemes, structured encodings, and domain-specific tokenizations offer others. As models scale and inference costs grow, the question of how to encode information—not just how to process it—becomes increasingly central to efficient deep learning.

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


### References

[^1]: Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.

[^2]: Zaheer, M., Guruganesh, G., Dubey, A., et al. (2020). Big Bird: Transformers for Longer Sequences. NeurIPS 2020.

[^3]: Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2023). Token Merging: Your ViT But Faster. ICLR 2023.

[^4]: Wei, Y., et al. (2023). Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers. CVPR 2023.

[^5]: Efficient Vision Transformer via Token Merger. IEEE Transactions on Image Processing, 2023.

[^6]: Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). Efficient Transformers: A Survey. ACM Computing Surveys.

[^7]: Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.

[^8]: Ballé, J., Laparra, V., & Simoncelli, E. P. (2017). End-to-end Optimized Image Compression. ICLR 2017.

[^9]: Liu, J., et al. (2023). Learned Image Compression with Mixed Transformer-CNN Architectures. CVPR 2023.

[^10]: Chen, T., et al. (2024). Variational Autoencoder-based Neural Network Model Compression. arXiv:2408.14513.
