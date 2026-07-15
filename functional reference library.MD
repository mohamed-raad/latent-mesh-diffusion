The Build Strategy: How to Use Both Tactically
Instead of using the adapter as a wrapper, treat Google's repository as a functional reference library and write your own kernel.

1. Copy Their Discrete Corruption and Canvas Math
Don't rewrite the fundamental text-diffusion math. From Google's code, copy the logic for:

Uniform State Diffusion: How they create the 256-token canvas using random placeholder tokens instead of hard BERT [MASK] tokens.

Self-Conditioning Gated MLPs: How they feed the previous step's softmax distribution back into the canvas embeddings to stabilize the next denoising pass.

Entropy-Bounded Sampling: Their exact logic for tracking token confidence thresholds to determine which tokens to lock in and which to fully re-noise.

2. Rebuild the Layer Mechanics into a Graph
Instead of loading a standard sequential Transformer class, write a custom execution hub:

Map your hidden states to an internal routing layer.

Let the routing layer interact with your local directory structure (the Obsidian markdown graph), determining which isolated, locally-trained NoProp blocks are streamed into VRAM to handle the current canvas state.

3. Layer in Your TurboQuant and MTP Heads Natively
Because you are writing the kernel from scratch, you can inject your custom attention mechanics directly:

Apply the PolarQuant rotation matrix right inside your local block's attention forward pass to compress the Key-Value (KV) cache.

Attach the parallel Multi-Token Prediction (MTP) projection heads directly to the output of your mesh nodes so they can learn to predict upcoming token blocks during the local training phase.

The Verdict
Use Google's repository to steal the Discrete Text Diffusion Sampler equations, but write your own Mesh Layered Kernel in PyTorch or clean JAX. It's the only way you'll get real-time dynamic growth and 100% independent NoProp training to run smoothly inside your 8GB VRAM budget.