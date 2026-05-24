# Road-to-LLMs-Scientist
## Env Settings (Windows 11)
- Conda
```bash
conda create --name llm-scientist python=3.12
conda activate llm-scientist
```
- pytorch (CUDA 13.0) and common dependencies
```bash
# https://pytorch.org/get-started/locally
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# https://huggingface.co/docs/transformers/installation
# The industry standard for loading model architectures (Llama, Qwen, Mistral) and their associated tokenizers
pip3 install transformers

# https://huggingface.co/docs/peft/install
# Parameter-Efficient Fine-Tuning: Essential for running techniques like LoRA (Low-Rank Adaptation), allowing you to train massive models on consumer or single-node GPUs by freezing the base weights and injecting trainable rank-decomposition matrices
pip3 install peft

# https://huggingface.co/docs/trl/index
# Transformers Reinforcement Learning
# The go-to library for alignment and safety research, containing out-of-the-box implementations for DPO (Direct Preference Optimization), RLHF, and reward modeling
pip3 install trl

# https://github.com/fla-org/flash-linear-attention
# [OPTIONAL] Flash Linear Attention
pip3 install flash-linear-attention

```

## Set HF_TOKEN
```bash
hf auth login
```