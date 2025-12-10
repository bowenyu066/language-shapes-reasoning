## Related Work

The challenge of efficient sequence processing in deep learning has attracted sustained attention, with two complementary research directions emerging: optimizing how models process fixed representations, and designing more compact representations that preserve task-relevant information.

### Efficient Processing of Fixed Representations

The quadratic complexity of self-attention has motivated numerous architectural innovations aimed at reducing computational cost while maintaining model expressiveness. Sparse attention mechanisms, exemplified by Longformer [1] and BigBird [2], replace full pairwise attention with structured patterns combining local windows, global tokens, and random connections. These approaches reduce complexity from $O(n^2)$ to $O(n)$ while preserving theoretical expressiveness—BigBird notably maintains Turing completeness despite its sparse structure. Empirically, such methods extend context length by 4–8× on equivalent hardware, enabling significant improvements on long-document tasks [2].

Beyond static sparsity patterns, dynamic token reduction methods have gained traction, particularly in vision transformers. Token Merging (ToMe) [3] progressively combines similar tokens across transformer layers, achieving substantial speedups without task-specific fine-tuning. Joint Token Pruning and Squeezing [4] extends this idea by aggregating information from pruned tokens rather than discarding it entirely. These methods demonstrate that not all tokens contribute equally to final predictions—a 95% reduction in token count can degrade accuracy by less than 1% in some settings [5]. Linear attention variants offer an alternative path, replacing the softmax operation with kernel approximations that permit $O(n)$ complexity through associative matrix multiplication [6].

A common thread unites these approaches: they treat the input representation as given and seek algorithmic efficiency in processing it. While effective, this framing leaves unexplored the possibility that the representation itself might be redesigned for efficiency.

### Compact Representations Across Domains

An orthogonal line of research asks whether the same information can be encoded more compactly before processing begins. Autoencoders and their variational extensions (VAEs) [7] learn to compress high-dimensional inputs into low-dimensional latent spaces, with reconstruction quality degrading gracefully as dimensionality decreases. This rate-distortion trade-off is well characterized: aggressive compression reduces bitrate but introduces reconstruction error following predictable information-theoretic curves [8].

In learned image compression, neural networks now rival or exceed classical codecs by learning latent representations optimized for reconstruction fidelity [9]. Recent work achieves compression rates of 10–20× while maintaining perceptual quality, substantially outperforming traditional methods like pruning and quantization [10]. The key insight is that learned representations can exploit statistical structure in data more effectively than hand-designed encodings.

However, these compression studies primarily target reconstruction tasks, where the goal is faithful recovery of the original input. Whether compact representations preserve the information needed for *reasoning*—where models must perform inference rather than reconstruction—remains less understood. Our work addresses this gap by treating natural languages as a family of representations with inherently different information densities, allowing us to study how representation compactness affects reasoning performance without engineering artificial compression schemes.

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
