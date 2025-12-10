## Related Work

Research into cost-effective sequence modeling generally fall into two catagories: speeding up how we process a fixed sequence of tokens, and changing the representation so that we start with fewer tokens in the first place.

### Efficient Processing of Fixed Representations

The quadratic cost of self-attention has led to many architectural tricks for cutting compute while keeping performance. A major line of work focuses on sparse attention, as in Longformer [1] and BigBird [2]. These models replace full attention with a mix of local windows, a few global tokens, and some random connections. This brings the complexity down from $O(n^2)$ toward $O(n)$ while still allowing information to move across the entire context. In practice, these ideas extend context length by about 4–8× on the same hardware and give noticeable gains on long-document tasks [2].

Other approaches reduce the number of tokens as the model runs. In vision transformers, Token Merging (ToMe) [3] repeatedly merges similar tokens across layers, speeding up inference without any task-specific finetuning. Joint Token Pruning and Squeezing [4] pushes this further by folding information from dropped tokens into the ones that remain, rather than discarding it. These results make a clear point: not all tokens matter equally. In some setups, cutting 95% of tokens only costs under 1% accuracy [5].

Linear-attention variants take a different route, approximating the softmax kernel so attention can be computed with associative matrix multiplications, again bringing complexity closer to $O(n)$ [6]. Across all of these methods, the input sequence is assumed fixed, and the goal is to process it more efficiently. This leaves open the question of whether we can instead redesign the representation itself to be more efficient.

### Compact Representations Across Domains
A separate line of work asks whether we can encode the same information more compactly before it ever reaches the transformer.

Autoencoders and variational autoencoders [7] are classical examples. They learn to compress high-dimensional inputs into low-dimensional latent spaces. Their behavior is well described by the rate-distortion trade-off, where aggressive compression reduces bitrate but introduces reconstruction error following predictable information-theoretic curves [8]. In learned image compression, neural networks now rival classical codecs by learning latent representations that are explicitly optimized for reconstruction quality [9], achieving 10–20× compression while keeping perceptual quality high [10]. The key insight is that learned encodings can exploit statistical structure in data far better than hand-designed schemes.

However, a key gap remains between these compression studies and the need of large reasoning models. Most compression work targets reconstruction, where the objective is to faithfully recover the input signal. It is still unclear to what extent compact representations preserve the information necessary for reasoning, rather than just for signal recovery.

Our work targets this gap by treating natural languages as a family of representations with different inherent information densities. Instead of designing a new compression model, we fix the underlying task and vary only the linguistic encoding, asking whether shrinking the token budget compromises reasoning performance or offers a viable path toward representation-level efficiency.

### References

[1] Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.

[2] Zaheer, M., Guruganesh, G., Dubey, A., et al. (2020). Big Bird: Transformers for Longer Sequences. NeurIPS 2020.

[3] Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2023). Token Merging: Your ViT But Faster. ICLR 2023.

[4] Wei, Y., et al. (2023). Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers. CVPR 2023.

[5] Efficient Vision Transformer via Token Merger. IEEE Transactions on Image Processing, 2023.

[6] Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. (2022). Efficient Transformers: A Survey. ACM Computing Surveys.

[7] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.

[8] Ballé, J., Laparra, V., & Simoncelli, E. P. (2017). End-to-end Optimized Image Compression. ICLR 2017.

[9] Liu, J., et al. (2023). Learned Image Compression with Mixed Transformer-CNN Architectures. CVPR 2023.

[10] Chen, T., et al. (2024). Variational Autoencoder-based Neural Network Model Compression. arXiv:2408.14513.

[11] Qwen Team. (2025). Qwen3-8B Model Card. Hugging Face. https://huggingface.co/Qwen/Qwen3-8B

[12] Meta AI. (2024). Llama-3.1-8B-Instruct Model Card. Hugging Face. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
