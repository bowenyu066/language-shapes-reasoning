## Results and Analysis

We present findings from three experiments that progressively investigate the relationship between representation choice and reasoning quality. Our results support the thesis that representation density affects token efficiency but not necessarily task performance—provided the model has been adequately trained on the target representation.

### 4.1 SOTA Models Exhibit Representation-Invariant Accuracy

Our first experiment evaluates whether well-trained frontier models achieve similar reasoning accuracy across languages representing the same mathematical content. If reasoning quality depends only on information content rather than representation format, we would expect accuracy to remain stable across languages while token counts vary with representation density.

**Accuracy Results.** Table 1 presents accuracy on MMATH across five languages for three frontier models. ChatGPT-5.1 achieves 79.9% accuracy on English and 78.3% on Chinese—a gap of merely 1.6 percentage points. DeepSeek-V3.2 shows even stronger representation invariance, with English at 88.5% and Chinese at 87.7% (0.8% gap). Across both models, the maximum accuracy gap between any two languages remains under 10 percentage points for well-supported languages.

| Model | English | Chinese | Spanish | Japanese | Thai |
|-------|---------|---------|---------|----------|------|
| ChatGPT-5.1 | 79.9% | 78.3% | 77.8% | 73.3% | 70.9% |
| DeepSeek-V3.2 | 88.5% | 87.7% | 88.5% | 85.6% | 85.0% |

These results confirm that sufficiently trained models achieve near-equivalent reasoning performance regardless of which linguistic representation encodes the problem. The small gaps that do exist likely reflect residual differences in training data coverage rather than fundamental representation limitations.

**Token Efficiency.** While accuracy remains stable, output length varies substantially across representations. For DeepSeek-V3.2, average output length drops from 2,964 characters in English to 1,924 characters in Chinese—a 35% reduction. To quantify token efficiency independent of any single tokenizer's biases, we measured the same outputs using five different tokenizers. On problems where the model answered correctly in both languages, Chinese outputs consistently require fewer tokens: the ZH/EN ratio ranges from 0.91 (DeepSeek tokenizer) to 0.95 (Llama tokenizer), representing 5–9% token savings.

This finding has direct computational implications. If Chinese reasoning uses 9% fewer tokens while maintaining equivalent accuracy, then inference costs scale proportionally—the same reasoning quality at reduced computational expense. The savings compound with sequence length due to attention's quadratic complexity.

### 4.2 Training Distribution Determines Representation Capability

Our second experiment tests whether representation invariance is a property of model architecture or training distribution. We compare two 8B-parameter models with contrasting training emphases: Qwen3-8B (multilingual-focused) and Llama-3.1-8B (English-centric).

**Contrasting Performance Profiles.** On GSM8K, Qwen3-8B achieves 88.7% accuracy on English and 89.4% on Chinese (max tokens = 4096)—essentially identical performance across representations. Llama-3.1-8B tells a different story: 80.3% on English but only 52.8% on Chinese, a gap of 27.5 percentage points.

| Model | English | Chinese | Gap |
|-------|---------|---------|-----|
| Qwen3-8B | 88.7% | 89.4% | +0.7% |
| Llama-3.1-8B | 80.3% | 52.8% | −27.5% |

This stark contrast reveals that the performance gap is not intrinsic to the Chinese representation—if it were, Qwen would also struggle. Instead, Llama's degradation reflects insufficient exposure to Chinese during training. The representation itself supports effective reasoning; the model simply hasn't learned to exploit it.

**Accuracy vs. Token Budget.** Examining how accuracy scales with maximum token allowance reveals further insights. For Qwen3-8B, both languages reach near-peak accuracy by 512 tokens (EN: 88.1%, ZH: 90.7%), with Chinese actually converging slightly faster. This suggests that denser representations complete reasoning in fewer tokens without sacrificing quality.

Llama-3.1-8B shows a different pattern: English accuracy climbs steadily from 4.6% (128 tokens) to 80.3% (512+ tokens), while Chinese plateaus at 52.8% regardless of additional token budget. The bottleneck is not token availability but the model's inability to reason effectively in the Chinese representation.

**Token Length Distributions.** For the 1,088 problems where Qwen answered correctly in both languages, Chinese outputs average 296 tokens compared to 308 for English (ZH/EN ratio = 0.96 on the Qwen tokenizer, 0.90 on DeepSeek). The distribution of Chinese output lengths is shifted left, confirming that the model naturally produces more concise reasoning chains in the denser representation.

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

Our experiments yield three key insights:

1. **Representation invariance is achievable.** SOTA models exhibit near-identical accuracy across languages while benefiting from 5–9% token savings in denser representations like Chinese. Reasoning quality depends on information content, not representation format.

2. **Training distribution determines representation capability.** Models with narrow training distributions (Llama on Chinese) show degraded performance that models with broader training (Qwen) do not. The bottleneck lies in training, not architecture.

3. **Representation bottlenecks are learnable.** Modest fine-tuning substantially improves performance on underrepresented representations without degrading capability on well-supported ones. Representation efficiency is a capability to unlock, not a trade-off to accept.

Together, these findings support treating representation choice as a first-class design variable. For well-trained models, denser representations offer genuine efficiency gains at no accuracy cost. For models with capability gaps, targeted training can bridge the divide.
