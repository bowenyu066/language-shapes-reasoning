## Discussion
Our findings offer a new perspective on efficiency in deep learning sequences. Rather than focusing exclusively on architectural innovations to process fixed representations more efficiently, we demonstrate that meaningful gains may be achieved by selecting representations that encode the same information in fewer tokens. This shift moves the optimization perspective from the algorithmic level to the data level. Below, we discuss the implications of this perspective, its connection to broader research areas, and the limitations of our study.

### Beyond Architecture: Optimizing the Representation Itself

The dominant paradigm for efficient deep learning treats the input representation as fixed and seeks computational savings through algorithmic innovations such as sparse attention, token pruning, or linear approximations. Our results suggest a complementary and underutilized approach by optimizing the representation itself. When SOTA models achieve equivalent accuracy on Chinese and English while using 5–9% fewer tokens in Chinese, they demonstrate that representation choice can yield efficiency gains without quality degradation.

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
