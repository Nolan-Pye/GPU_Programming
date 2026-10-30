# Project Proposal: CUDA-Based LLM Inference Engine

**Name:** Nolan Pye

**Objective:** My goal is to build a small large language model (LLM) inference engine using CUDA. I will use pretrained TinyStories models from Karpathy's llama2.c project and implement the forward pass from scratch using custom CUDA kernels. The models will generate text on my laptop's NVIDIA GTX 1650 Ti GPU. I will compare inference performance against the original CPU implementation across different model sizes.

**Programming Language:** CUDA

**Hypothesis:** I expect the GPU implementation to generate text significantly faster than the CPU version, potentially achieving a 10× speedup for larger models. I anticipate smaller speedups for the smallest models because they have less computation to take advantage of GPU parallelism. I also expect that optimizing CUDA kernels and memory access patterns will further improve inference performance.
