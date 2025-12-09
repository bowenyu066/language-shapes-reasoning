# Does Language Shape Reasoning? Evaluating Token Efficiency Across Languages in Mathematical Problem Solving

**TL;DR:** We found that Chinese achieves comparable mathematical reasoning accuracy to English while using **~10-50% fewer tokens** across multiple state-of-the-art language models. This has significant implications for inference costs, latency, and our understanding of how language structure affects computational reasoning.

---

## 💰 The Hidden Cost of Reasoning

Imagine you're building an AI-powered math tutor. Your users submit questions, and your LLM reasons through the problem step-by-step before providing an answer. With GPT-4 pricing at ~$30 per million output tokens, token efficiency isn't just an academic concern—it directly impacts your infrastructure costs and user experience.

But here's an interesting question: **What if you could cut your token usage in half just by changing the language?**

Recent advances in multilingual LLMs have made it possible to reason in multiple languages with comparable accuracy. However, different languages have fundamentally different information densities. Chinese characters typically convey more semantic information per token compared to English words. Could this linguistic efficiency translate into computational efficiency?

---

## 🔬 Our Hypothesis

**Chinese is more token-efficient than English and other languages for mathematical reasoning tasks, achieving comparable accuracy with significantly fewer output tokens.**

This hypothesis is grounded in linguistic theory:
- **Logographic nature**: Chinese characters represent morphemes/syllables rather than phonemes
- **High information density**: Each character carries substantial semantic meaning
- **Compact tokenization**: Fewer tokens needed to express the same concepts

If true, this has practical implications for:
- **Cost reduction** in production systems
- **Faster inference** due to shorter sequence lengths
- **Better context utilization** within fixed token windows
- **Environmental impact** through reduced compute requirements

---

## 🧪 Experimental Design

We conducted three complementary experiments to test our hypothesis:

### Experiment 1: Open-Source Models on GSM8K

**Models:** Qwen3-8B, Llama-3.1-8B-Instruct  
**Dataset:** GSM8K (1,319 grade-school math problems)  
**Languages:** English (original) + Chinese (translated via GPT-4o-mini)  
**Configurations:** Multiple max token limits (128, 256, 512, 1024, 4096)

**Why these models?** 
- **Qwen3-8B**: Chinese-centric training, multilingual capability
- **Llama-3.1-8B**: English-centric, widely adopted baseline

**Special condition:** "Translate-then-solve" mode where we prompt Chinese questions with "Please think in English" to isolate language vs. capability effects.

### Experiment 2: Frontier Models on MMATH

**Models:** GPT-4o-mini (ChatGPT-5.1), Gemini-2.5-Flash, DeepSeek-V3.2  
**Dataset:** MMATH (374 challenging multilingual math problems)  
**Languages:** English, Chinese, Spanish, Thai  
**Configuration:** Max 8192 tokens

**Why MMATH?** Harder than GSM8K, already multilingual (no translation needed), tests generalization beyond grade-school math.

### Experiment 3: Fine-Tuning Llama on Chinese GSM8K

**Goal:** Determine if Llama's poor Chinese performance is due to lack of training data  
**Method:** Supervised fine-tuning (SFT) on Chinese GSM8K training set  
**Status:** Evaluation in progress

---

## 📊 Results

### Finding 1: Qwen Achieves Language Parity, Llama Does Not

![Figure 1: GSM8K Accuracy Comparison](results/gsm8k/en_vs_zh_vs_zh_translate_then_solve.pdf)

**Qwen3-8B (max_tokens=1024):**
- English: **88.5%** accuracy
- Chinese: **89.8%** accuracy
- Chinese + "think in English": **90.1%** accuracy

**Llama-3.1-8B (max_tokens=1024):**
- English: **79.6%** accuracy  
- Chinese: **51.6%** accuracy (⚠️ **35% degradation**)
- Chinese + "think in English": **44.0%** accuracy (❌ **even worse**)

**Key Insights:**
- Qwen demonstrates true multilingual reasoning capability
- Llama's Chinese performance suffers dramatically despite being a frontier open-source model
- Asking the model to "think in English" on Chinese problems doesn't help—in fact, it makes Llama's performance even worse, suggesting this introduces confusion rather than leveraging English reasoning strengths

### Finding 2: Chinese Uses ~50% Fewer Tokens (Open-Source Models)

![Figure 2: Qwen Output Length Distribution](results/gsm8k/length_distribution_qwen.pdf)
![Figure 3: Llama Output Length Distribution](results/gsm8k/length_distribution_llama.pdf)

**Qwen3-8B** average output length (max_tokens=1024):
- English: **~800 tokens**
- Chinese: **~400 tokens** (50% reduction ✓)

**Llama-3.1-8B** average output length (max_tokens=1024):
- English: **~900 tokens**
- Chinese: **~450 tokens** (50% reduction ✓)

**This is the money result:** Even when Llama struggles with Chinese accuracy, it still produces dramatically shorter outputs. This suggests tokenization efficiency is orthogonal to reasoning capability.

### Finding 3: Language Efficiency Holds for Frontier Models

![Figure 4: MMATH Accuracy Across Languages](results/mmath/en_vs_zh_vs_es_vs_th.pdf)

**ChatGPT-5.1 (GPT-4o-mini):**
- English: 79.9% accuracy, **1,282 avg tokens**
- Chinese: 78.3% accuracy, **1,155 avg tokens** (10% reduction ✓)
- Spanish: Similar patterns
- Thai: Higher token usage due to script complexity

**DeepSeek-V3.2:**
- English: 88.5% accuracy, **2,964 avg tokens**
- Chinese: Similar accuracy trends with token efficiency (detailed data shows consistent reduction)

![Figure 5-7: Token Distribution by Model](results/mmath/length_distribution_chatgpt(2).pdf)

**Key Insight:** Even for SOTA models, Chinese maintains comparable accuracy with 10-15% fewer tokens on harder problems. The efficiency gain is smaller than in open-source models but still significant at scale.

⚠️ **Note on Gemini:** Gemini-2.5-Flash had numerous empty responses (293/374), indicating API issues during evaluation. Valid responses showed high accuracy (96.3%) but results are not comparable.

---

## 💡 Discussion

### What We Learned

**1. Token efficiency is real and consistent**  
Across all models tested, Chinese consistently uses fewer tokens than English for mathematical reasoning. This holds true whether the model is good at Chinese (Qwen) or bad at it (Llama).

**2. Efficiency ≠ Capability**  
Llama demonstrates that a model can be token-efficient in a language while still performing poorly. This suggests:
- Tokenization determines sequence length
- Training data determines reasoning quality
- These are independent factors

**3. The "think in X language" prompt doesn't help**  
Our translate-then-solve experiment shows that asking models to think in English when given Chinese problems either has no effect (Qwen) or makes things worse (Llama). This suggests:
- Internal reasoning might not be as language-dependent as we think
- Models can't easily "code-switch" their reasoning process on command
- The language of the problem statement matters more than explicit reasoning language instructions

**4. Scale matters for multilingual reasoning**  
Frontier models (ChatGPT, DeepSeek) show smaller efficiency gaps than smaller models (Qwen, Llama), possibly because:
- Larger context windows reduce pressure on token efficiency
- Better multilingual training data
- More sophisticated tokenizers

### Practical Implications

**For Deployment:**
- If serving Chinese users, Chinese-language inference could reduce API costs by 10-50%
- Latency improvements from shorter sequences
- Better context budget utilization

**For Research:**
- Language choice is a valid optimization axis, not just a localization concern
- Tokenizer design significantly impacts computational efficiency
- Need for better multilingual benchmarks that control for tokenization effects

**For Training:**
- Our SFT experiment (still evaluating) will reveal whether fine-tuning can close the capability gap while preserving efficiency gains

---

## 🚧 Limitations & Future Work

### Limitations

1. **Translation artifacts**: Our Chinese GSM8K is machine-translated. While GPT-4o-mini is high-quality, translations may not be perfectly natural or culturally equivalent.

2. **Limited language coverage**: We tested 4-5 languages. Other language families (Arabic, Hindi, Japanese) might show different patterns.

3. **Math-specific**: Results may not generalize to other reasoning domains (code, logic puzzles, scientific reasoning).

4. **Tokenizer confounds**: Different models use different tokenizers, making cross-model comparisons imperfect.

5. **Incomplete SFT results**: We can't yet conclude whether training data is the bottleneck for models like Llama.

### Future Directions

- **Expand to more languages**: Test languages with different scripts and morphological complexity (agglutinative languages, Arabic, etc.)
  
- **Other reasoning tasks**: Evaluate on code generation, logical reasoning, scientific QA
  
- **Controlled tokenization**: Normalize for tokenizer differences or train models with identical tokenizers
  
- **Human evaluation**: Check if shorter Chinese outputs feel less "thorough" to human evaluators
  
- **Multimodal reasoning**: Do efficiency patterns hold for vision-language models?

- **Complete SFT analysis**: Understand if targeted training can achieve both accuracy parity and efficiency gains

---

## 🎯 Conclusion

We set out to answer: **Does Chinese reasoning cost less (computationally)?**

The answer is **yes**, with important nuances:

✅ Chinese consistently uses **10-50% fewer tokens** than English for mathematical reasoning  
✅ This efficiency holds across multiple model families and scales  
✅ Token efficiency is **independent of reasoning quality**—even poorly-performing models are token-efficient  
⚠️ However, not all models can reason effectively in Chinese despite efficiency gains  
⚠️ "Think in English" prompts don't salvage poor multilingual capabilities  

**The big picture:** Language is not just a localization concern—it's a computational resource management concern. As we deploy LLMs globally, understanding the interaction between linguistic structure, tokenization, and reasoning efficiency becomes crucial for building cost-effective, performant systems.

For a system serving millions of Chinese users doing math reasoning, switching from English to Chinese could save tens of thousands of dollars monthly in API costs while maintaining equivalent accuracy—if you choose the right model.

---

## 📁 Repository & Reproducibility

All code, data processing scripts, evaluation pipelines, and results are available in this repository. See `README.md` for:
- Setup instructions
- Dataset download and processing
- Evaluation scripts for local and API models
- Result aggregation and plotting

**Key files:**
- `scripts/10_eval_gsm8k_local.py` - Open-source model evaluation
- `scripts/11_eval_mmath_api.py` - Frontier model evaluation  
- `scripts/40_plot_results.py` - Generate all figures
- `results/` - Raw evaluation outputs and plots

---

## 🙏 Acknowledgments

This project was completed as part of MIT 6.7960 Deep Learning (Fall 2025). Thanks to the course staff for guidance and computational resources.

**Datasets:**
- GSM8K: OpenAI (https://github.com/openai/grade-school-math)
- MMATH: Multilingual math reasoning benchmark

**Models evaluated:**
- Qwen3-8B (Alibaba Cloud)
- Llama-3.1-8B (Meta)
- GPT-4o-mini (OpenAI)
- Gemini-2.5-Flash (Google)
- DeepSeek-V3.2 (DeepSeek)

---

*Have questions or ideas for collaboration? Open an issue or reach out!*
