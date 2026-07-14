import os
import random


VAULT_TOPICS = {
    "Attention Mechanisms": {
        "content": [
            "Attention computes a weighted sum of values where weights are derived from a query and key similarity function.",
            "The standard attention mechanism is defined as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.",
            "Multi-head attention runs multiple attention heads in parallel, each operating on a projected subspace.",
            "Self-attention allows each token to attend to every other token in the sequence, enabling long-range dependencies.",
            "The attention bottleneck arises when the softmax distribution becomes too diffuse over long sequences.",
        ],
        "links": ["Transformer Architecture", "Multi-Head Attention", "Softmax Function"],
    },
    "Transformer Architecture": {
        "content": [
            "The Transformer is a sequence-to-sequence model relying entirely on self-attention without recurrence.",
            "It consists of an encoder stack and a decoder stack, each composed of multi-head attention and feed-forward layers.",
            "Layer normalization and residual connections are applied after each sub-layer in the Transformer block.",
            "The original Transformer was introduced for machine translation and has since become the dominant architecture in NLP.",
            "Transformers process all tokens in parallel, making them highly parallelizable compared to RNNs.",
        ],
        "links": ["Attention Mechanisms", "Layer Normalization", "Residual Connections", "Feed-Forward Networks"],
    },
    "Multi-Head Attention": {
        "content": [
            "Multi-head attention projects queries, keys, and values h times with different learned linear projections.",
            "The h attention heads are computed in parallel, then concatenated and linearly projected to the output dimension.",
            "Each head potentially learns to focus on different aspects of the input, such as syntactic vs. semantic relationships.",
            "The computational cost is linear in the number of heads since each head operates on a fraction of the full dimension.",
            "Typical configurations use h = 8 or h = 16 heads with d_k = d_v = d_model / h.",
        ],
        "links": ["Attention Mechanisms", "Transformer Architecture"],
    },
    "Layer Normalization": {
        "content": [
            "Layer normalization computes the mean and variance across the feature dimension for each training example.",
            "It stabilizes training by ensuring that the activations have zero mean and unit variance regardless of input scale.",
            "Unlike batch normalization, layer normalization does not depend on batch size and behaves identically at train and test time.",
            "In Transformers, layer normalization is typically applied before or after each sub-layer (Pre-LN vs Post-LN).",
            "The learnable parameters gamma (scale) and beta (shift) allow the model to undo the normalization if needed.",
        ],
        "links": ["Transformer Architecture", "Residual Connections"],
    },
    "Residual Connections": {
        "content": [
            "Residual connections add the input of a layer to its output, allowing gradients to flow directly through the network.",
            "They mitigate the vanishing gradient problem in deep networks by providing a shortcut path for backpropagation.",
            "In Transformers, each sub-layer output is added to its input before layer normalization.",
            "Residual connections enable training of very deep models by preventing representation degradation.",
            "The formulation is: output = LayerNorm(x + Sublayer(x)).",
        ],
        "links": ["Transformer Architecture", "Layer Normalization"],
    },
    "Feed-Forward Networks": {
        "content": [
            "The feed-forward network in a Transformer consists of two linear transformations with a ReLU or GELU activation.",
            "The hidden dimension is typically 4 times the model dimension, creating a bottleneck then expansion structure.",
            "Each token is processed independently by the FFN, making it the most computationally expensive component per token.",
            "The FFN can be viewed as a memory unit that stores and retrieves knowledge from the model's parameters.",
            "SwiGLU and other gated variants have been shown to improve FFN performance over the standard ReLU activation.",
        ],
        "links": ["Transformer Architecture", "Attention Mechanisms"],
    },
    "Softmax Function": {
        "content": [
            "The softmax function converts a vector of raw scores into a probability distribution over classes.",
            "It is defined as softmax(x_i) = exp(x_i) / sum_j exp(x_j), ensuring outputs are positive and sum to one.",
            "The temperature parameter controls the sharpness of the distribution: higher temperature produces softer probabilities.",
            "Softmax is used in the attention mechanism to convert dot-product similarities into attention weights.",
            "Numerical stability is achieved by subtracting the maximum input value before exponentiation.",
        ],
        "links": ["Attention Mechanisms", "Multi-Head Attention"],
    },
    "Diffusion Models": {
        "content": [
            "Diffusion models learn to reverse a gradual noising process to generate data from pure noise.",
            "The forward process adds Gaussian noise over T timesteps following a variance schedule beta_1 through beta_T.",
            "The reverse process learns to denoise, parameterized by a neural network that predicts the added noise at each step.",
            "Training uses a simple mean-squared error loss between predicted and actual noise at randomly sampled timesteps.",
            "Sampling iteratively denoises from pure Gaussian noise through the learned reverse process.",
        ],
        "links": ["Denoising Objective", "Noise Scheduling", "Discrete Diffusion"],
    },
    "Denoising Objective": {
        "content": [
            "The denoising objective trains a model to predict the original clean data point from a corrupted observation.",
            "Score matching equivalently trains the model to predict the gradient of the log-density of the noisy data.",
            "The simplified objective L_simple = E_{t,x_0,epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ] is most commonly used.",
            "SNR-weighting adjusts the loss contribution per timestep based on the signal-to-noise ratio alpha_bar / (1 - alpha_bar).",
            "Continuous-time diffusion extends the objective to an integral over t in [0, 1] rather than discrete timesteps.",
        ],
        "links": ["Diffusion Models", "Noise Scheduling"],
    },
    "Noise Scheduling": {
        "content": [
            "The noise schedule determines how quickly information is destroyed during the forward diffusion process.",
            "Cosine scheduling alpha_bar = cos^2(pi * t / 2) provides a smooth transition that avoids abrupt information loss.",
            "Linear scheduling beta_t from 1e-4 to 0.02 was used in the original DDPM but can be suboptimal for image generation.",
            "The cumulative product alpha_bar_t = prod_{s=1}^t (1 - beta_s) controls the remaining signal at timestep t.",
            "Different schedules significantly impact sample quality, with cosine schedules generally outperforming linear ones.",
        ],
        "links": ["Diffusion Models", "Denoising Objective"],
    },
    "Discrete Diffusion": {
        "content": [
            "Discrete diffusion operates on categorical data by defining a transition matrix between discrete states.",
            "The absorbing-state diffusion gradually replaces tokens with a special [MASK] token over the forward process.",
            "The uniform-state diffusion transitions between all vocabulary tokens with uniform probability.",
            "The corruption process for text can be defined as a Markov chain on the vocabulary with an transition rate matrix Q.",
            "The reverse process predicts the original uncorrupted token given the noisy observation at timestep t.",
        ],
        "links": ["Diffusion Models", "Uniform-State Denoising"],
    },
    "Uniform-State Denoising": {
        "content": [
            "Uniform-state denoising corrupts tokens by replacing them with uniformly sampled random tokens from the vocabulary.",
            "The forward process has a uniform stationary distribution, ensuring the model learns from a fully noisy prior.",
            "The denoising network predicts the original clean token distribution from the uniformly corrupted input.",
            "This approach is particularly well-suited for non-autoregressive text generation on a fixed-length canvas.",
            "The corruption strength is controlled by a continuous timestep t, with t=0 being clean and t=1 being fully random.",
        ],
        "links": ["Discrete Diffusion", "Denoising Objective", "Diffusion Canvas"],
    },
    "Diffusion Canvas": {
        "content": [
            "The diffusion canvas is a fixed-length block of tokens that undergoes iterative denoising from random to coherent text.",
            "Each canvas block is processed in parallel using bidirectional attention, enabling simultaneous generation of all tokens.",
            "Tokens that have converged to stable, low-entropy predictions are locked and no longer perturbed in subsequent steps.",
            "Multiple canvas blocks can be concatenated to produce longer outputs without increasing the per-block computation.",
            "The canvas approach eliminates the sequential token-by-token bottleneck of autoregressive decoding.",
        ],
        "links": ["Uniform-State Denoising", "Bidirectional Attention", "Adaptive Early Stopping"],
    },
    "Bidirectional Attention": {
        "content": [
            "Bidirectional attention allows every token to attend to every other token in both directions simultaneously.",
            "Unlike causal attention (used in autoregressive models), there is no masking of future positions.",
            "This enables each token to condition its representation on the full surrounding context, improving coherence.",
            "In diffusion-based generation, bidirectional attention is natural since the entire canvas is generated at once.",
            "The computational cost is O(L^2) per layer where L is the canvas length, but the entire canvas is produced in one forward pass.",
        ],
        "links": ["Diffusion Canvas", "Attention Mechanisms"],
    },
    "Adaptive Early Stopping": {
        "content": [
            "Adaptive early stopping halts denoising iterations for individual tokens once they have stabilized.",
            "A token is considered stable when its predicted label has not changed for consecutive steps and its entropy is below a threshold.",
            "Locked tokens are preserved and no longer corrupted, reducing total compute for already-converged positions.",
            "The process terminates for the entire canvas when >95% of tokens are locked, avoiding unnecessary denoising steps.",
            "This creates a variable compute budget per sample, allocating more iterations to harder tokens.",
        ],
        "links": ["Diffusion Canvas", "Uniform-State Denoising"],
    },
    "NoProp Learning": {
        "content": [
            "NoProp (no backpropagation) learning trains each network block with an isolated local objective, not a global loss.",
            "Each block receives the same input and target, computing its own MSE loss and updating its parameters independently.",
            "The gradient for each block depends only on its own Jacobian, eliminating gradient dilution across deep stacks.",
            "Blocks are selected by a router based on cosine similarity between the input query and learned anchor embeddings.",
            "Since blocks do not share gradients, they can be trained in parallel and independently specialized to different data regimes.",
        ],
        "links": ["Mesh Router", "Localized Credit Assignment", "Feed-Forward Networks"],
    },
    "Mesh Router": {
        "content": [
            "The mesh router selects which expert blocks to activate for a given input based on cosine similarity routing.",
            "Each node in the mesh has a normalized anchor embedding that serves as its routing centroid.",
            "The top-k nodes with highest cosine similarity to the input query are activated for each forward pass.",
            "Quantile-based load balancing adjusts routing scores to prevent expert starvation across the mesh.",
            "The router enables conditional computation where different subnetworks specialize to different input patterns.",
        ],
        "links": ["NoProp Learning", "Localized Credit Assignment", "Node Mitosis"],
    },
    "Localized Credit Assignment": {
        "content": [
            "Localized credit assignment computes parameter updates using only local error signals, avoiding the global chain rule.",
            "Each block minimizes its own prediction error without receiving gradient information from downstream blocks.",
            "This eliminates the vanishing gradient problem because gradients do not traverse multiple layers.",
            "The convergence rate per block is independent of its depth in the network, unlike backprop where early layers converge slowly.",
            "Localized learning enables each block to specialize independently, creating an ensemble of experts rather than a monolithic model.",
        ],
        "links": ["NoProp Learning", "Mesh Router"],
    },
    "Node Mitosis": {
        "content": [
            "Node mitosis clones an existing mesh node when its sustained error exceeds a threshold, creating a specialized child.",
            "The child node inherits the parent's weights via state dict copy, then receives LoRA adapters for rapid adaptation.",
            "The child's anchor embedding is a perturbed version of the parent's, ensuring it routes to similar but distinct inputs.",
            "Mitosis enables the mesh to grow dynamically in response to underperforming regions of the input space.",
            "The mitosis threshold controls how quickly the mesh expands, balancing specialization against model complexity.",
        ],
        "links": ["Mesh Router", "NoProp Learning", "LoRA Adapters"],
    },
    "LoRA Adapters": {
        "content": [
            "Low-Rank Adaptation (LoRA) decomposes weight updates into low-rank matrices A and B, where the original weights stay frozen.",
            "The forward pass becomes Wx + BAx, where B in R^{d x r}, A in R^{r x d} with rank r << d.",
            "LoRA adds only 2 * r * d trainable parameters per layer, a tiny fraction of the full parameter count.",
            "In the mesh, LoRA adapters enable child nodes to specialize without overwriting the parent's general knowledge.",
            "The scaling factor alpha / r controls the magnitude of the LoRA update relative to the frozen base weights.",
        ],
        "links": ["Node Mitosis", "NoProp Learning"],
    },
}


def create_vault(output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    topic_names = list(VAULT_TOPICS.keys())
    rng.shuffle(topic_names)

    index_items = []
    for topic in topic_names:
        info = VAULT_TOPICS[topic]
        filename = topic.replace(" ", "_").replace("-", "_") + ".md"

        lines = []
        lines.append(f"# {topic}\n")
        lines.append("")

        content_sample = rng.sample(info["content"], min(len(info["content"]), 3 + rng.randint(0, 2)))
        for para in content_sample:
            lines.append(para)
            lines.append("")

        if info["links"]:
            num_links = rng.randint(1, len(info["links"]))
            link_sample = rng.sample(info["links"], num_links)
            lines.append("## See Also\n")
            for link in link_sample:
                link_file = link.replace(" ", "_").replace("-", "_")
                lines.append(f"- [[{link_file}|{link}]]")
            lines.append("")

        lines.append("## Tags\n")
        tags = ["deep-learning", "mesh", topic.lower().replace(" ", "-")]
        lines.append(" ".join(f"#{t}" for t in tags))
        lines.append("")

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        index_items.append((topic, filename))

    index_lines = ["# Knowledge Index\n", ""]
    index_lines.append("This vault contains the following topics:\n")
    for topic, filename in sorted(index_items, key=lambda x: x[0]):
        index_lines.append(f"- [[{filename.replace('.md', '')}|{topic}]]")
    index_lines.append("")
    index_lines.append("## Tags\n")
    index_lines.append("#index #deep-learning #mesh")
    index_lines.append("")

    with open(os.path.join(output_dir, "Index.md"), "w") as f:
        f.write("\n".join(index_lines))

    return len(topic_names) + 1
