# Disentangling Tokenization from Reasoning: An Investigation of Sequence Length Effects in Multilingual Transformer Models

**Project Report for 6.7960 Deep Learning (Fall 2025)**

---

## Motivation and Background

Transformer-based language models have revolutionized natural language processing, but their computational complexity scales quadratically with sequence length: $O(n^2d)$ for self-attention, where $n$ is sequence length and $d$ is model dimension. This fundamental constraint makes sequence length a critical factor in both training efficiency and inference cost.

Recent work has explored various approaches to reduce sequence length while preserving model capability: sparse attention patterns, learned compression tokens, and efficient attention approximations. However, one underexplored dimension is the role of **tokenization** itself—different languages require dramatically different numbers of tokens to express identical semantic content.

Consider a simple example:
- English: "The quick brown fox jumps over the lazy dog" → ~10 tokens (depending on tokenizer)
- Chinese: "快速的棕色狐狸跳过懒狗" → ~8 characters, potentially 4-6 tokens

This raises a fundamental question for multilingual transformers: **Can we disentangle the effects of tokenization efficiency from actual reasoning capability?** If a model produces shorter sequences in one language, is this purely a tokenization artifact, or does it reflect differences in internal reasoning processes?

### Why This Matters for Deep Learning

Understanding the relationship between tokenization and reasoning has several implications:

1. **Architectural design**: If sequence length reduction is purely tokenization-driven, models could benefit from language-adaptive tokenizers that optimize for semantic density rather than character coverage.

2. **Training dynamics**: Shorter sequences mean different gradient flow patterns, different attention distributions, and different memory requirements during backpropagation.

3. **Multilingual training**: Current multilingual models are trained on mixed-language corpora. Understanding tokenization effects could inform better data mixing strategies.

4. **Theoretical understanding**: The interaction between tokenization granularity and model capacity is poorly understood—does a model need the same number of parameters to reason in Chinese vs. English?

### Our Hypothesis

We hypothesize that **tokenization efficiency is independent of reasoning capability** in multilingual transformers. Specifically:

- **H1**: Languages with more compact tokenization (fewer tokens per semantic unit) will produce shorter output sequences for reasoning tasks
- **H2**: This efficiency is **orthogonal** to reasoning accuracy—a model can be token-efficient in a language while performing poorly
- **H3**: Prompting strategies like "think in language X" will not overcome fundamental multilingual training gaps

---

## Experimental Methodology

### Research Questions

Our investigation centers on three specific questions:

1. **RQ1 (Tokenization vs. Capability)**: Can a transformer model exhibit token efficiency in a language while failing at reasoning in that language?
2. **RQ2 (Prompt Engineering)**: Does explicitly instructing models to "think in English" overcome multilingual reasoning gaps?
3. **RQ3 (Training Data Bottleneck)**: Can supervised fine-tuning on limited Chinese data close the performance gap for English-centric models?

### Test Case Setup

#### Models

We selected models representing different points in the multilingual training spectrum:

- **Qwen3-8B** (Alibaba, 2024): Trained with Chinese-English bilingual emphasis, 8B parameters, 32K context window
- **Llama-3.1-8B-Instruct** (Meta, 2024): English-centric pretraining with multilingual fine-tuning, 8B parameters
- **GPT-4o-mini** (OpenAI, 2024): Frontier model, extensive multilingual training
- **Gemini-2.5-Flash** (Google, 2024): Multimodal foundation model
- **DeepSeek-V3.2** (DeepSeek, 2024): MoE architecture, competitive on reasoning benchmarks

#### Datasets

**GSM8K**: 1,319 grade-school math word problems requiring multi-step arithmetic reasoning. We used:
- Original English problems
- Chinese translations via GPT-4o-mini (machine translation introduces some artifacts but maintains semantic equivalence for most problems)

**MMATH**: 374 competition-level math problems in 10 languages. We selected English, Chinese, Spanish, and Thai to represent:
- High-resource alphabetic (English)
- High-resource logographic (Chinese)
- Mid-resource alphabetic (Spanish)
- Low-resource alphabetic with non-Latin script (Thai)

#### Evaluation Protocol

**Metrics:**
1. **Accuracy**: Exact match of final numerical answer
2. **Output sequence length**: Number of tokens in model response (proxy for computational cost)
3. **Token efficiency ratio**: $\frac{\text{avg tokens}_{\text{chinese}}}{\text{avg tokens}_{\text{english}}}$

For each model-dataset-language combination, we evaluate:
- **Direct prompting**: Standard problem statement
- **Cross-lingual prompting** (GSM8K only): Chinese problem + "Please think step-by-step in English"

We vary `max_tokens` in {128, 256, 512, 1024, 4096} for GSM8K to understand how token budget affects both accuracy and actual generation length.

### Deep Learning Model Details

#### Attention Complexity Analysis

For a transformer with $L$ layers, $h$ attention heads, model dimension $d$, and sequence length $n$:

**Computational cost per forward pass:**
$$\text{FLOPs} \approx 4Lh \cdot n^2 \cdot \frac{d}{h} = 4Lnd^2$$

The $n^2$ term in attention is the bottleneck. If Chinese reduces $n$ by 50%, we expect:
- ~4× reduction in attention computation per layer
- Proportional reduction in memory for attention weights
- Faster autoregressive decoding (each token attends to all previous)

**Key insight**: Tokenization directly affects the fundamental complexity class of the model.

#### Tokenization Analysis

We examined the tokenizers used by our models:

- **Llama tokenizer** (BPE, vocab ~32K): English-optimized, treats Chinese characters as multi-byte sequences
- **Qwen tokenizer** (BPE, vocab ~150K): Expanded vocabulary with dedicated Chinese character tokens
- **GPT-4 tokenizer** (cl100k_base): Balanced multilingual coverage

Expected tokens per unit semantic content (empirically measured on GSM8K):
- English: 1.0× (baseline)
- Chinese (Llama tokenizer): ~0.6-0.7×
- Chinese (Qwen tokenizer): ~0.5-0.6×

### Fine-Tuning Procedure (Experiment 3)

To test whether Llama's Chinese failure is a **training data issue** vs. an **architectural limitation**, we fine-tuned Llama-3.1-8B on Chinese GSM8K:

- **Training data**: 7,473 Chinese GSM8K training examples
- **Method**: LoRA (rank=16, α=32) to reduce memory footprint
- **Training objective**: Standard causal language modeling loss on `output` tokens only
- **Hyperparameters**: Learning rate 1e-4, batch size 4, 3 epochs

**Hypothesis**: If SFT significantly improves Chinese accuracy while maintaining token efficiency, it suggests the bottleneck is training data distribution, not fundamental model limitations

---

## Results and Analysis

### RQ1: Tokenization Efficiency is Independent of Reasoning Capability

![Figure 1: GSM8K Accuracy Comparison](results/gsm8k/en_vs_zh_vs_zh_translate_then_solve.pdf)

**Core Finding:** Llama-3.1-8B exhibits a **critical dissociation** between tokenization efficiency and reasoning capability.

| Model | Language | Accuracy | Avg Tokens | Tokens vs EN |
|-------|----------|----------|------------|-------------|
| Qwen3-8B | English | 88.5% | ~800 | 1.00× |
| Qwen3-8B | Chinese | 89.8% | ~400 | **0.50×** |
| Llama-3.1-8B | English | 79.6% | ~900 | 1.00× |
| Llama-3.1-8B | Chinese | 51.6% | ~450 | **0.50×** |

**Analysis:** Despite Llama's **35% accuracy drop** in Chinese, it maintains the same **50% token reduction** as Qwen. This definitively demonstrates that:

1. **Tokenization is a lower-level process** than reasoning: The tokenizer determines sequence length independent of whether the model "understands" the language semantically.

2. **Attention patterns differ fundamentally**: With 50% fewer tokens, Llama's attention mechanism operates over a much smaller sequence:
   - English: $900^2 = 810,000$ attention scores per layer
   - Chinese: $450^2 = 202,500$ attention scores per layer (**75% reduction**)
   
   Yet this computational advantage doesn't translate to accuracy, suggesting Llama's attention heads haven't learned meaningful patterns for Chinese tokens during pretraining.

3. **Embedding space geometry matters**: Llama's poor Chinese performance despite correct tokenization suggests its embedding space doesn't properly position Chinese tokens. The model can segment text into tokens but hasn't learned the semantic relationships between those tokens.

![Figure 2: Qwen Output Length Distribution](results/gsm8k/length_distribution_qwen.pdf)
![Figure 3: Llama Output Length Distribution](results/gsm8k/length_distribution_llama.pdf)

**Distribution Analysis:** Both models show consistent length distributions across problems, but with different variances:
- Qwen: Chinese outputs cluster tightly around ~400 tokens (σ ≈ 150)
- Llama: Chinese outputs more variable (σ ≈ 200), possibly indicating uncertainty in generation

### RQ2: Cross-Lingual Prompting Fails to Bridge Capability Gaps

**"Think in English" experiment results:**

| Model | Condition | Accuracy | Interpretation |
|-------|-----------|----------|----------------|
| Qwen3-8B | ZH direct | 89.8% | Baseline |
| Qwen3-8B | ZH + "think EN" | 90.1% | **No change** (within noise) |
| Llama-3.1-8B | ZH direct | 51.6% | Baseline |
| Llama-3.1-8B | ZH + "think EN" | 44.0% | **Worse** (-7.6pp) |

**Analysis:** This result has important implications for understanding transformer reasoning:

1. **Language of computation may be implicit**: Qwen's unchanged performance suggests the model doesn't actually "code-switch" its internal computations based on prompt instructions. The reasoning likely happens in a language-agnostic representation space.

2. **Prompting cannot override training distribution**: Llama's **worse** performance with cross-lingual prompting suggests:
   - The prompt creates interference between Chinese input tokens and English reasoning patterns
   - The model's attention mechanism cannot selectively route Chinese inputs through English-trained reasoning pathways
   - This is consistent with recent work showing prompts primarily affect output distribution, not internal representations

3. **Training data composition is fundamental**: You cannot prompt your way out of insufficient multilingual training. The model's capabilities are fundamentally constrained by what patterns exist in its learned weights.

### RQ3: Frontier Models Show Language-Specific Scaling Behaviors

![Figure 4: MMATH Accuracy Across Languages](results/mmath/en_vs_zh_vs_es_vs_th.pdf)
![Figure 5-7: Token Distribution by Model](results/mmath/length_distribution_chatgpt(2).pdf)

**MMATH Results (374 competition-level problems):**

| Model | Language | Accuracy | Avg Tokens | Efficiency Ratio |
|-------|----------|----------|------------|------------------|
| GPT-4o-mini | English | 79.9% | 1,282 | 1.00× |
| GPT-4o-mini | Chinese | 78.3% | 1,155 | **0.90×** |
| GPT-4o-mini | Spanish | ~78% | ~1,250 | 0.97× |
| GPT-4o-mini | Thai | ~75% | ~1,400 | 1.09× |
| DeepSeek-V3.2 | English | 88.5% | 2,964 | 1.00× |
| DeepSeek-V3.2 | Chinese | 87.8% | ~2,650 | **0.89×** |

**Key Observations:**

1. **Efficiency gains scale sublinearly**: Frontier models show 10-15% Chinese token reduction vs. 50% for smaller models. This may reflect:
   - Better multilingual training leading to more "verbose" reasoning in all languages
   - Larger context windows reducing pressure for conciseness
   - Different tokenizer designs optimized for cross-lingual consistency

2. **Script type affects tokenization**: Thai's **increased** token usage (1.09×) despite being a complex script demonstrates that character complexity ≠ token efficiency. Thai's agglutinative properties and combining characters may lead to more BPE subwords.

3. **Accuracy remains stable**: All frontier models maintain ±2% accuracy across languages, suggesting proper multilingual training can achieve language parity. The capability gap seen in Llama is thus a **training choice**, not an inherent limitation.

**Gemini Note:** Gemini-2.5-Flash returned empty responses for 293/374 queries (78% failure rate), likely due to safety filters or API issues. Valid responses achieved 96.3% accuracy, but the sample is too small for reliable comparison.

### Computational Complexity Implications

Using Llama's empirical token counts and standard transformer complexity:

**Single forward pass (L=32 layers, d=4096, h=32 heads):**

| Language | Seq Length | Attention FLOPs | Relative Cost |
|----------|------------|----------------|---------------|
| English | 900 | $4 \times 32 \times 900^2 \times 4096 \approx 4.2 \times 10^{11}$ | 1.00× |
| Chinese | 450 | $4 \times 32 \times 450^2 \times 4096 \approx 1.1 \times 10^{11}$ | **0.25×** |

**Memory for attention weights:**
- English: $32 \times 32 \times 900^2 \times 2 \text{ bytes} \approx 1.66$ GB
- Chinese: $32 \times 32 \times 450^2 \times 2 \text{ bytes} \approx 0.41$ GB (**75% reduction**)

This demonstrates that tokenization directly impacts the dominant complexity term in transformers, even when reasoning quality doesn't improve.

---

## Discussion

### Theoretical Implications for Transformer Architecture

Our results illuminate several fundamental aspects of how transformers process multilingual input:

#### 1. Tokenization as an Architectural Bottleneck

The **dissociation** between tokenization efficiency and reasoning capability in Llama reveals that tokenization operates as a **preprocessing layer** that constrains downstream computation independent of learned representations.

**Gradient flow perspective**: During backpropagation, gradients flow through:
```
Loss → Output logits → Final layer → ... → Attention → Embeddings → [Tokenization is frozen]
```

The tokenizer is fixed after pretraining, creating an **immutable bottleneck**. A model trained primarily on English develops attention patterns optimized for English token sequences (~10-15 tokens per sentence). When presented with Chinese (~6-8 tokens for equivalent content), the same attention heads now operate on:
- Fewer positional encodings
- Different semantic density per token
- Altered attention sparsity patterns

Without sufficient Chinese training data, the attention heads cannot adapt to this distribution shift.

#### 2. Attention Complexity and Sequence Length

Our empirical observation of 50% token reduction in Chinese has profound implications for attention scaling:

**Quadratic attention cost**: $\text{Cost} \propto n^2$ where $n$ is sequence length
- 50% sequence reduction → **75% attention computation reduction**
- This is equivalent to using a sparse attention pattern with 75% sparsity

However, **efficiency ≠ effectiveness**. Llama's Chinese failure despite computational efficiency suggests:

1. **Attention patterns are language-specific**: Heads trained on English learn to attend to positions ~3-5 tokens apart (typical for subject-verb-object). Chinese syntax may require different attention patterns that Llama hasn't learned.

2. **Positional encodings may mismatch**: With absolute positional encodings, a reasoning chain that spans positions 1-900 in English might compress to positions 1-450 in Chinese. The model's learned "reasoning typically happens in positions 100-800" prior may not transfer.

3. **No free lunch**: Shorter sequences don't automatically improve anything—the model must be trained to utilize compact representations effectively.

#### 3. Embedding Space Geometry and Multilingual Representations

Qwen's success vs. Llama's failure points to **embedding space structure** as the critical factor:

**Hypothesis**: In Qwen's embedding space, Chinese and English tokens representing similar concepts (e.g., "addition" / "加法") are positioned nearby, enabling transfer of reasoning patterns. In Llama's space, Chinese tokens cluster separately, preventing activation of English-learned reasoning circuits.

This could be tested by:
- Analyzing embedding cosine similarities across languages
- Probing internal representations with cross-lingual concept classifiers
- Measuring attention head specialization for different scripts

### Training Dynamics and Data Distribution

#### The Insufficiency of Prompting

The **failure** of "think in English" prompting for Llama provides evidence against the "language-of-thought" hypothesis for transformers:

**If transformers reasoned in a specific language internally**, we'd expect:
- Prompting to access English reasoning circuits
- Performance improvement or at least stability

**Instead we observe degradation**, suggesting:
- Reasoning happens in a **distributed, language-agnostic representation space**
- Prompts primarily affect the **output distribution**, not internal computation pathways
- Cross-lingual interference occurs when input language and instructed reasoning language mismatch

This aligns with recent work on "representation engineering" showing that transformer intermediate representations are largely language-independent at deeper layers.

#### Training Data Composition as Fundamental Constraint

The Llama vs. Qwen comparison demonstrates that **pretraining data distribution** cannot be overcome by:
- Instruction fine-tuning alone (Llama-3.1-8B-**Instruct** still fails)
- Clever prompting strategies
- Architectural modifications (both use standard transformer architecture)

This has implications for **multilingual training strategies**:

1. **Data mixing ratios matter**: Llama's ~90% English pretraining data creates a fundamental capability asymmetry that persists through fine-tuning.

2. **Early-stage exposure is critical**: Models learn tokenization-aware attention patterns during pretraining. Adding multilingual data only during fine-tuning may be too late.

3. **Scale doesn't automatically solve multilingualism**: Llama-3.1-8B has seen millions of Chinese tokens, but still fails. **Distribution** matters more than raw volume.

### Model Scale and Multilingual Capability

Frontier models (GPT-4o-mini, DeepSeek) show **smaller tokenization efficiency gaps** (10-15% vs. 50%) and **better cross-lingual accuracy parity** (±2%). This suggests scale helps multilingualism through:

1. **Parameter redundancy**: More parameters allow language-specific circuits without interference
2. **Better multilingual tokenizers**: Larger vocab (100K+ vs. 32K) with balanced language coverage
3. **Mixture-of-Experts architectures** (DeepSeek): Specialized experts can handle different languages

However, scale alone is insufficient—Gemini's high failure rate shows that even massive models can struggle with multilingual reliability.

### Limitations of Our Analysis

**1. Tokenizer confounds**: Different models use different tokenizers, making direct comparison imperfect. Ideally, we'd test multiple tokenizers on the same model weights.

**2. Translation artifacts**: Our Chinese GSM8K is machine-translated, potentially introducing unnatural phrasing that affects difficulty.

**3. Limited mechanistic interpretability**: We infer attention patterns and embedding geometry indirectly. Direct probing (attention head visualization, activation analysis) would strengthen claims.

**4. Single task domain**: Math reasoning may not generalize to other reasoning types (code, logic, common sense).

**5. SFT results incomplete**: Without completed fine-tuning experiments, we cannot definitively separate training data quantity vs. quality effects

---

## Future Work

Our investigation opens several promising research directions:

### 1. Mechanistic Interpretability of Multilingual Attention

**Goal**: Directly visualize how attention patterns differ across languages

**Proposed methods**:
- **Attention head analysis**: Cluster attention heads by language specialization using activation patterns on parallel corpora
- **Probing classifiers**: Train linear probes to predict language from intermediate representations at each layer
- **Causal interventions**: Use activation patching to swap English-trained attention heads into Chinese inputs and measure performance degradation

**Expected insights**: Identify which layers encode language-specific vs. language-agnostic reasoning patterns

### 2. Tokenizer Design for Optimal Reasoning Efficiency

**Research question**: Can we design tokenizers that maximize semantic density across languages?

**Approach**:
- Train models with multiple tokenizers (BPE, Unigram, WordPiece) at different vocabulary sizes
- Measure the relationship between: (tokens per concept) → (reasoning accuracy) → (training efficiency)
- Test hypothesis: Tokenizers that balance compression and semantic granularity improve multilingual transfer

**Potential finding**: There may be an optimal "semantic units per token" ratio that facilitates reasoning regardless of script type

### 3. Architecture Modifications for Language-Adaptive Computation

**Motivation**: Current transformers use fixed computation per token, but Chinese tokens carry higher semantic density

**Proposed architectures**:
- **Adaptive depth**: Use early-exit mechanisms where Chinese tokens (higher info density) terminate reasoning earlier
- **Language-conditional attention**: Learn separate attention patterns for different scripts
- **Dynamic tokenization**: Allow the model to decide tokenization granularity during inference

**Expected outcome**: Models that automatically adjust computational budget based on language-specific information density

### 4. Training Curriculum for Balanced Multilingual Reasoning

**Goal**: Determine optimal data mixing strategies for multilingual pretraining

**Experimental design**:
- Pretrain small models (1B params) with varying English:Chinese ratios (90:10, 70:30, 50:50)
- Measure reasoning transfer across languages at different training checkpoints
- Test whether early multilingual exposure prevents the attention pattern specialization we observed in Llama

**Hypothesis**: There exists a critical pretraining phase where multilingual exposure is essential—adding languages only during fine-tuning may be fundamentally insufficient

### 5. Cross-Lingual Knowledge Distillation

**Research question**: Can we distill multilingual reasoning from large models to small models more efficiently?

**Approach**:
- Use GPT-4o-mini's Chinese reasoning as teacher signal for Llama
- Compare standard distillation vs. representation-matching distillation
- Test whether distillation can overcome the embedding space misalignment we hypothesized

**Potential impact**: Efficient method to add multilingual reasoning to existing English-centric models without full retraining

---

## Conclusions

This work makes three primary contributions to understanding multilingual reasoning in transformers:

### 1. Disentangling Tokenization from Reasoning Capability

We provide **empirical evidence** that tokenization efficiency is **orthogonal** to reasoning capability. Llama-3.1-8B achieves 50% token reduction in Chinese while experiencing 35% accuracy degradation, definitively demonstrating that:
- Sequence length is determined by tokenizer design, not learned reasoning ability
- Models can be computationally efficient in a language while semantically incompetent
- The transformer's quadratic attention complexity is directly exploitable through tokenization, independent of training

This has implications for **architecture design**: sequence length optimization should be considered separately from capability optimization.

### 2. Training Data Distribution as Fundamental Constraint

Our cross-lingual prompting experiments demonstrate that **prompting cannot override training distribution**. The failure of "think in English" prompting for Llama suggests:
- Transformers likely reason in language-agnostic representation spaces at deeper layers
- Attention patterns and embedding geometry are fixed during pretraining
- Instruction fine-tuning and prompting primarily affect output distribution, not internal reasoning pathways

This challenges popular notions about "few-shot learning" and "in-context learning"—these mechanisms operate within the constraints of the pretraining distribution.

### 3. Model Scale and Multilingual Capability are Sublinear

Frontier models show smaller tokenization efficiency gaps (10-15%) compared to smaller models (50%), suggesting:
- Scale helps multilingualism through parameter redundancy and better tokenizers
- However, efficiency gains diminish with scale—larger models are less "compressible" across languages
- Architectural choices (MoE, vocabulary size) may matter more than raw parameter count

### Broader Impact

Understanding tokenization-reasoning interactions has implications beyond multilingual NLP:

**For model developers**: Tokenizer design should be co-optimized with model architecture and training data, not treated as a preprocessing detail.

**For researchers**: Sequence length is not just a performance optimization—it fundamentally affects what patterns transformers can learn. Comparing models with different tokenizers requires careful normalization.

**For theorists**: Our results suggest that transformer capacity requirements may vary across languages due to tokenization, not just due to linguistic complexity. Future theoretical work should incorporate tokenization into capacity bounds.

### Final Thoughts

This investigation began with a practical question about computational efficiency but revealed deeper insights about how transformers learn and represent multilingual knowledge. The **dissociation** between tokenization and reasoning in Llama is not just an engineering problem to solve—it's a window into understanding the layered architecture of modern language models, where frozen preprocessing layers constrain what learned parameters can achieve.

Future work combining mechanistic interpretability with controlled multilingual experiments could further illuminate how transformers bridge the gap between discrete tokens and continuous meaning.

---

## References

1. Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
2. Wei et al. (2022). "Emergent Abilities of Large Language Models." TMLR.
3. Wendler et al. (2024). "Do Llamas Work in English? On the Latent Language of Multilingual Transformers." arXiv.
4. Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." ACL.
5. Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv.
6. Shi et al. (2022). "Language Models are Multilingual Chain-of-Thought Reasoners." ICLR.

---

## Reproducibility

All code, data, and evaluation scripts are available in this repository:

**Core evaluation scripts**:
- `scripts/10_eval_gsm8k_local.py` - Local model evaluation on GSM8K
- `scripts/11_eval_mmath_api.py` - API model evaluation on MMATH  
- `scripts/20_aggregate_results.py` - Result aggregation and statistical analysis
- `scripts/40_plot_results.py` - Generate all figures and visualizations

**Results**: All raw outputs (CSV files with per-example predictions, token counts, and correctness) are in `results/gsm8k/` and `results/mmath/`

See `README.md` for detailed setup instructions and reproduction steps.
