## Results and Analysis

We present findings from three experiments that progressively investigate the relationship between representation choice and reasoning quality. Our results support the thesis that representation density affects token efficiency but not necessarily task performance—provided the model has been adequately trained on the target representation. Crucially, we distinguish between *intrinsic density* (information per character) and *realized density* (information per token), showing that the gap between them—the **Encoder Gap**—determines whether theoretical efficiency translates to computational savings.

### 4.1 SOTA Models Exhibit Representation-Invariant Accuracy

Our first experiment evaluates whether well-trained frontier models achieve similar reasoning accuracy across representations encoding the same mathematical content. If reasoning quality depends only on information content rather than representation format, we would expect accuracy to remain stable across representations while output lengths vary with intrinsic density.

**Accuracy Results.** Table 1 presents accuracy on MMATH across four languages for three frontier models. ChatGPT-5.1 achieves 80.0% accuracy on English and 78.3% on Chinese—a gap of merely 1.7 percentage points. DeepSeek-V3.2 shows even stronger representation invariance, with English at 88.5% and Chinese at 87.7% (0.8% gap). Across all three models, the maximum accuracy gap between any two languages remains under 10 percentage points.

![Figure 1: MMATH accuracy across languages (English, Chinese, Spanish, and Thai)](/figures/en_vs_zh_vs_es_vs_th_mmath.png)

| Model | English | Chinese | Spanish | Thai |
|-------|---------|---------|---------|------|
| ChatGPT-5.1 | 80.0% | 78.3% | 77.8% | 70.9% |
| Gemini-2.5-Flash | 81.2% | 80.8% | 82.9% | 80.2% |
| DeepSeek-V3.2 | 88.5% | 87.7% | 88.5% | 85.0% |

These results confirm that sufficiently trained models achieve near-equivalent reasoning performance regardless of which linguistic representation encodes the problem. The small gaps that do exist likely reflect residual differences in training data coverage rather than fundamental representation limitations. This establishes a critical premise: for well-trained models, the "semantic latent space" is stable—the model has learned to decouple *meaning* from *encoding*.

**Intrinsic Density: Character-Level Analysis.** Having established reasoning invariance, we examine whether different representations exhibit different intrinsic densities. Figures 5–7 show the distribution of raw output character lengths across languages.

![Figure 5: Character-length distributions for ChatGPT-5.1 over all four languages](/figures/length_distribution_chatgpt.png)

![Figure 6: Character-length distributions for Gemini-2.5-Flash over all four languages](/figures/length_distribution_gemini.png)

![Figure 7: Character-length distributions for DeepSeek-V3.2 over all four languages](/figures/length_distribution_deepseek.png)

A notable pattern emerges: DeepSeek-V3.2 produces substantially shorter Chinese outputs at the character level compared to English, while ChatGPT-5.1 and Gemini-2.5-Flash show more similar distributions across languages. Chinese, as a logographic writing system, naturally encodes more information per character than alphabetic scripts. Yet whether models *exploit* this intrinsic density varies—DeepSeek generates concise Chinese outputs, while ChatGPT and Gemini do not leverage this compactness to the same degree. This difference in *generation verbosity* foreshadows the divergence we observe at the token level.

**Realized Density: Token-Level Analysis.** The central question is whether intrinsic density translates to computational savings. We define *realized density* as information per token—the actual sequence length processed by the Transformer, which determines computational cost via attention's $O(n^2)$ complexity.

![Figure 2: Token length distributions for ChatGPT-5.1 over all four languages](/figures/token_distribution_chatgpt_mmath.png)

![Figure 3: Token length distributions for Gemini-2.5-Flash over all four languages](/figures/token_distribution_gemini_mmath.png)

![Figure 4: Token length distributions for DeepSeek-V3.2 over all four languages](/figures/token_distribution_deepseek_mmath.png)

Here we observe a critical divergence:

1. **Alignment (DeepSeek-V3.2):** The model translates intrinsic density into realized density. Average output token length drops from 786.4 tokens in English to 700.7 tokens in Chinese, reduced by 11%. Chinese uses the *fewest* tokens among all four languages, directly reducing computational cost.

2. **Inversion (ChatGPT-5.1, Gemini-2.5-Flash):** These models exhibit "density inversion." Despite Chinese being intrinsically denser, Chinese outputs require 33% more tokens (ChatGPT) and 40% more tokens (Gemini) than English. The theoretical efficiency advantage becomes a practical liability.

| Model | EN Tokens | ZH Tokens | ZH/EN Ratio | Status |
|-------|-----------|-----------|-------------|--------|
| DeepSeek-V3.2 | 786.4 | 700.7 | 0.89 | Aligned |
| ChatGPT-5.1 | 282.6 | 375.8 | 1.33 | Inverted |
| Gemini-2.5-Flash | 360.5 | 503.8 | 1.40 | Inverted |

**The Encoder Gap.** This divergence isolates the **Encoder Gap**—the discrepancy between intrinsic and realized density. For DeepSeek, the tokenizer (and model training) preserves the density advantage: fewer characters → fewer tokens. For ChatGPT and Gemini, the encoding is inefficient: the tokenizer fragments the dense Chinese signal into more tokens, or the model generates more verbose outputs, negating the representation's inherent compactness.

To quantify this independent of any single model's tokenizer, we re-tokenized DeepSeek-V3.2's outputs (problems correct in both EN and ZH, $N=321$) using five different tokenizers:

| Tokenizer | Avg EN Tokens | Avg ZH Tokens | ZH/EN Ratio |
|-----------|---------------|---------------|-------------|
| DeepSeek-V3 | 823.6 | 748.3 | **0.909** |
| Seed-Coder-8B | 941.5 | 858.6 | **0.912** |
| Qwen3-8B | 904.9 | 847.9 | **0.937** |
| GPT-4o | 866.2 | 811.5 | **0.937** |
| Llama-3.1-8B | 860.1 | 813.3 | **0.946** |

All five tokenizers show ZH/EN ratios below 1.0 (5–9% savings), confirming that DeepSeek's Chinese outputs are genuinely more compact—the efficiency is real and tokenizer-independent. The variation in ratios (0.91–0.95) reflects differences in tokenizer vocabulary coverage for Chinese.

**Implications.** This finding has direct computational implications. When a model is aligned—producing concise outputs in dense representations and using a tokenizer that preserves that density—inference costs scale proportionally with representation choice. The same reasoning quality at 11% reduced token count yields savings in attention computation. However, when misaligned, choosing a "denser" representation can *increase* costs. Representation efficiency is not automatic; it requires encoder-representation alignment.

### 4.2 Training Distribution Determines Representation Capability

Our second experiment tests whether representation invariance is a property of model architecture or training distribution. We compare two 8B-parameter models with contrasting training emphases: Qwen3-8B and Llama-3.1-8B-Instruct. According to official documentation, Qwen3-8B was trained on 36 trillion tokens spanning 119 languages with explicit support for over 100 languages including Chinese [11]. In contrast, Meta's Llama-3.1-8B-Instruct officially supports only eight languages: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai. Notably, Meta explicitly stated that Llama 3.1 has been trained on a broader collection of languages than the 8 supported languages, but developers must fine-tune for unsupported languages, such as Chinese, for better performance [12]. Thus, the contrast between the two models allows us to isolate the effect of training distribution while controlling for model scale.

**Contrasting Performance Profiles.** On GSM8K, Qwen3-8B achieves 88.7% accuracy on English and 89.4% on Chinese (max tokens = 4096). These performances across representations are essentially identical—mirroring the representation invariance we observed in SOTA models in Section 4.1. However, the results diverge sharply for Llama-3.1-8B: 80.3% on English but only 52.8% on Chinese, a gap of 27.5 percentage points.

| Model | English | Chinese | Gap |
|-------|---------|---------|-----|
| Qwen3-8B | 88.7% | 89.4% | +0.7% |
| Llama-3.1-8B | 80.3% | 52.8% | −27.5% |

[Figure]

This stark contrast reveals that the performance gap is not intrinsic to the Chinese representation—if it were, Qwen would also struggle. Instead, Llama's performance decline reflects its insufficient exposure to Chinese during training. The representation itself supports effective reasoning; the model simply hasn't learned to exploit it.

**Per-Question Consistency Analysis.** To further examine this phenomenon, we analyzed per-question consistency across the two languages. For each of the 1,319 test problems, we recorded whether the model answered correctly in English, Chinese, both, or neither.

| Model | Both Correct | EN Only | ZH Only | Both Wrong |
|-------|--------------|---------|---------|------------|
| Qwen3-8B | 1,088 (82.5%) | 82 (6.2%) | 91 (6.9%) | 58 (4.4%) |
| Llama-3.1-8B | 613 (46.5%) | **446** (**33.8%**) | 84 (6.4%) | 176 (13.3%) |

The results are striking. Llama answers 446 questions (33.8%) correctly in English but incorrectly in Chinese. These are problems where the model demonstrably possesses the mathematical knowledge but fails to apply it when the problem is presented in Chinese. A representative instance:

> **Problem:** *There are four schools competing at a basketball tournament. Each school has sent a girls' basketball team and a boys' basketball team and each team has 5 players each. Each school has also sent a coach for each team. In total, how many people have all of the schools sent?*
>
> **Chinese Version:** "有四所学校参加篮球锦标赛。每所学校均派出一支女子篮球队和一支男子篮球队，每队有 5 名球员。每所学校还为每支队派出一名教练。总共这些学校一共派出了多少人？"
>
> **Gold Answer:** 48
>
> **English Output (Correct):** The model correctly reasons: "Each team has 5 players and 1 coach, so 6 people per team. Each school sends 2 teams (girls' and boys'), so 12 people per school. With 4 schools, the total is 4 × 12 = 48."
>
> **Chinese Output (Incorrect):** The model miscounts: "每所学校派出了两支球队，每支球队有5名球员，所以每所学校派出了10名球员。每所学校还有一名教练，总共 1 名。" (Each school sent 2 teams with 5 players each, so 10 players per school. Each school also has one coach, totaling 1.) The model treats "a coach for each team" as one coach per school rather than one per team, arriving at 11 people per school and 44 total.

This example illustrates a systematic failure mode: the model understands the mathematical structure when presented in English but misparses the same logical relationship ("a coach for each team" → 2 coaches per school) when the problem is presented in Chinese. The mathematical capability is conserved; only the representation-specific parsing fails.

**Accuracy vs. Token Budget.** We examined how accuracy scales with maximum token allowance. For Qwen3-8B, both languages reach near-peak accuracy by 512 tokens (EN: 88.1%, ZH: 90.7%), with Chinese actually converging slightly faster. This suggests that denser representations complete reasoning in fewer tokens without sacrificing quality.

Llama-3.1-8B shows a different pattern: English accuracy climbs steadily from 4.6% (128 tokens) to 80.3% (512+ tokens), while Chinese plateaus at 52.8% regardless of additional token budget. The bottleneck is not token availability but the model's inability to reason effectively in the Chinese representation.

**Intrinsic Density: Character-Level Analysis.** Paralleling our analysis in Section 4.1, we examine whether the intrinsic density advantage of Chinese manifests in the open-source models' outputs. For the 1,088 problems where Qwen3-8B answered correctly in both languages, Chinese outputs average 314 characters compared to 616 for English—a 49% reduction (ZH/EN ratio = 0.51). This confirms that Chinese is intrinsically denser: the model expresses equivalent reasoning in roughly half the characters.

[INSERT FIGURE: GSM8K character length distributions for Qwen3-8B]

This pattern is consistent with what we observed for DeepSeek-V3.2 on MMATH (Section 4.1), where Chinese outputs were ~35% shorter at the character level. Both well-trained models—Qwen and DeepSeek—produce concise Chinese outputs that exploit the representation's intrinsic density.

**Realized Density: Token-Level Analysis.** The critical question, as established in Section 4.1, is whether intrinsic density translates to realized efficiency. We re-tokenized Qwen3-8B's outputs using five different tokenizers:

| Tokenizer | Avg EN Tokens | Avg ZH Tokens | ZH/EN Ratio | Status |
|-----------|---------------|---------------|-------------|--------|
| DeepSeek-V3 | 288.4 | 258.9 | **0.898** | Aligned |
| Seed-Coder-8B | 325.0 | 303.1 | **0.933** | Aligned |
| Qwen3-8B | 308.4 | 296.4 | **0.961** | Aligned |
| GPT-4o | 292.9 | 297.0 | 1.014 | **Inverted** |
| Llama-3.1-8B | 294.1 | 303.7 | **1.033** | **Inverted** |

A striking pattern emerges that directly parallels our MMATH findings:

1. **Multilingual-optimized tokenizers** (DeepSeek, Seed-Coder, Qwen) preserve the density advantage: Chinese uses 4–10% fewer tokens than English.

2. **English-centric tokenizers** (GPT-4o, Llama) exhibit density inversion: despite Chinese being 49% shorter at the character level, it requires the *same or more* tokens after encoding.

This is the **Encoder Gap** in action. Llama's tokenizer, optimized for English, fragments Chinese text inefficiently—each Chinese character may require multiple tokens, destroying the intrinsic density advantage. The same reasoning, expressed more compactly in Chinese characters, balloons back to equivalent (or greater) token counts after encoding.

**The Compounding Effect.** For Llama-3.1-8B, the Encoder Gap compounds its training gap. Not only does the model struggle to *reason* in Chinese (accuracy: 52.8% vs 80.3%), but even when it succeeds, it gains no efficiency benefit—Chinese outputs require *more* tokens (ratio = 1.033) despite being intrinsically denser. The model pays a "Tokenizer Tax" on top of its capability gap.

In contrast, Qwen3-8B demonstrates what aligned representation efficiency looks like: near-identical accuracy across languages (88.7% vs 89.4%) combined with genuine token savings (4–10% depending on tokenizer). The model has learned to exploit the dense representation, and multilingual tokenizers preserve that advantage.

### 4.3 Representation Bottlenecks Are Learnable

Our third experiment asks whether Llama's Chinese performance gap can be closed through targeted fine-tuning. If the bottleneck reflects training distribution rather than architectural limitation, then additional exposure to Chinese reasoning should improve performance.

**Fine-tuning Setup.** We fine-tuned Llama-3.1-8B on the Chinese GSM8K training set (7,473 examples) using LoRA for 500 steps. This represents a modest intervention—approximately 0.1% of pretraining compute.

**Results.** Fine-tuning on Chinese data yields substantial improvements:

| Model | English | Chinese | Gap |
|-------|---------|---------|-----|
| Llama (base) | 80.3% | 52.8% | −27.5% |
| Llama (ft-zh) | 81.1% | 61.6% | −19.5% |

Chinese accuracy improves by 8.8 percentage points (52.8% → 61.6%) while English performance is maintained (slight improvement to 81.1%). The EN-ZH gap narrows from 27.5% to 19.5%. Notably, fine-tuning on Chinese does not degrade English performance, suggesting the model acquires new representation capabilities without forgetting existing ones.

**Interpretation.** These results demonstrate that representation bottlenecks are learnable rather than architectural. A model that initially struggles with a denser representation can acquire competence through targeted exposure. This has practical implications: rather than accepting representation-dependent performance as fixed, practitioners can adapt models to work effectively with more efficient representations.

The persistence of a 19.5% gap after fine-tuning suggests that 500 steps of LoRA adaptation captures much but not all of the capability gap. More extensive fine-tuning or full-parameter training would likely narrow the gap further. Nevertheless, the experiment establishes the key finding: representation capability is malleable.

### 4.4 Summary of Findings

Our experiments yield three key insights, unified by the framework of intrinsic versus realized density:

1. **Reasoning is representation-invariant for well-trained models.** SOTA models exhibit near-identical accuracy across languages (<2% gap for ChatGPT, DeepSeek; <1% for Qwen). The model's "semantic latent space" is stable—it has learned to decouple meaning from encoding.

2. **Intrinsic density does not guarantee realized efficiency.** Chinese is ~35–50% denser at the character level, but this advantage is preserved only when the encoder (tokenizer) and model training are aligned with the representation. DeepSeek and Qwen achieve 5–10% token savings with multilingual tokenizers; ChatGPT and Gemini show 33–40% token *penalties*; English-centric tokenizers (Llama, GPT-4o) exhibit density inversion even on well-formed outputs. The **Encoder Gap** determines whether theoretical efficiency becomes practical savings.

3. **Training distribution determines representation capability, and bottlenecks are learnable.** Models with narrow training distributions (Llama on Chinese) show degraded performance that models with broader training (Qwen) do not. However, modest fine-tuning substantially improves performance on underrepresented representations without degrading capability on well-supported ones.

Together, these findings support treating representation choice as a first-class design variable—but with an important qualification. Realizing efficiency gains from dense representations requires alignment across three layers: (1) choosing representations with high intrinsic density, (2) using tokenizers that preserve that density, and (3) training models that exploit rather than waste the density advantage. When all three layers are aligned, denser representations offer genuine efficiency gains at no accuracy cost. When misaligned, the theoretical advantage becomes a practical liability.
