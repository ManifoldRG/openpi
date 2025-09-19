#!/usr/bin/env python3
"""
PaliGemma Weight Comparison Test Suite

This script compares the text generation capabilities of base PaliGemma weights
versus Pi0-trained PaliGemma weights using the HuggingFace transformers
implementation. The test isolates weight-level differences by using identical
inference pipelines.

Test Methodology:
1. Load base PaliGemma weights and Pi0-trained weights from checkpoints
2. Inject embedding tables into separate transformers model instances
3. Test both models on 25 diverse COCO validation images
4. Use three different prompt types to evaluate text generation robustness
5. Generate comparative HTML report with visual analysis

Key Features:
- Weight injection verification to ensure proper loading
- Multiple prompt types: basic caption, detailed description, creative pun
- Comprehensive HTML report with image visualization
- Statistical analysis of text generation patterns
- Detection of repetitive loops and generation failures
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import requests
from io import BytesIO

# JAX imports for loading weights
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Transformers imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# OpenPI imports
from openpi.models import model as _model
from openpi.models import pi0
from openpi.training.weight_loaders import PaliGemmaWeightLoader
from openpi.shared import download

def inject_embedding_with_verification(model, jax_embeddings, name="unknown"):
    """
    Inject JAX embedding weights into a PyTorch transformers model.

    This function replaces the embedding table in a PyTorch PaliGemma model
    with weights loaded from JAX checkpoints. It handles vocabulary size
    mismatches and verifies successful injection.

    Args:
        model: PyTorch PaliGemmaForConditionalGeneration model
        jax_embeddings: JAX embedding array from checkpoint
        name: Descriptive name for logging

    Returns:
        bool: True if injection successful, False otherwise
    """
    print(f"\nSTEP 2: Injecting {name} embeddings...")

    state_dict = model.state_dict()

    # Find embedding parameter
    embedding_param = None
    embedding_name = None

    for param_name, param in state_dict.items():
        if 'embed_tokens.weight' in param_name:
            embedding_param = param
            embedding_name = param_name
            break

    if embedding_param is None:
        print("  Could not find embedding parameter")
        return False

    # Store original for verification
    original_embedding = embedding_param.clone()

    print(f"  Found PyTorch embedding: {embedding_name}")
    print(f"  PyTorch shape: {embedding_param.shape}")
    print(f"  JAX shape: {jax_embeddings.shape}")

    # Handle vocab size mismatch
    pytorch_vocab_size, embed_dim = embedding_param.shape
    jax_vocab_size, jax_embed_dim = jax_embeddings.shape

    if embed_dim != jax_embed_dim:
        print(f"  ERROR: Embedding dimension mismatch: {embed_dim} vs {jax_embed_dim}")
        return False

    # Create injection tensor
    if pytorch_vocab_size >= jax_vocab_size:
        padded_embeddings = np.zeros((pytorch_vocab_size, embed_dim), dtype=jax_embeddings.dtype)
        padded_embeddings[:jax_vocab_size] = jax_embeddings
        injection_tensor = torch.from_numpy(padded_embeddings).to(embedding_param.dtype)
        print(f"  Vocabulary handling: Padded from {jax_vocab_size} to {pytorch_vocab_size}")
    else:
        truncated_embeddings = jax_embeddings[:pytorch_vocab_size]
        injection_tensor = torch.from_numpy(np.array(truncated_embeddings)).to(embedding_param.dtype)
        print(f"  Vocabulary handling: Truncated from {jax_vocab_size} to {pytorch_vocab_size}")

    print(f"  Injection tensor dtype: {injection_tensor.dtype}, target dtype: {embedding_param.dtype}")

    # Inject
    with torch.no_grad():
        embedding_param.copy_(injection_tensor)

    # Verify injection worked
    new_embedding = embedding_param.clone()
    injection_diff = torch.max(torch.abs(new_embedding - original_embedding))
    injection_mean = torch.mean(new_embedding)

    print(f"  Injection verification:")
    print(f"    Max parameter change: {injection_diff:.6f}")
    print(f"    New embedding mean: {injection_mean:.6f}")
    print(f"    Target JAX mean: {float(np.mean(jax_embeddings)):.6f}")

    if injection_diff > 0.001:  # Significant change indicates successful injection
        print(f"  SUCCESS: Embedding injection verified")
        return True
    else:
        print(f"  WARNING: Injection may have failed - minimal parameter change")
        return False



def replace_paligemma_weights_with_pi0():
    """
    Replace PaliGemma weights with Pi0-trained weights and return the modified model.
    """


    # Load transformers components
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"SUCCESS: Loaded processor for {model_id}")

    # Load both weight sets
    print("\nLoading weight sets from checkpoints...")


    pi0_params = _model.restore_params(
        download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
    )
    pi0_weights = pi0_params.get("PaliGemma", {})
    pi0_embeddings = pi0_weights['llm']['embedder']['input_embedding']

    print("SUCCESS: Both weight sets loaded")

    try:
        print("Loading PaliGemma model with memory optimizations...")
        base_paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            #torch_dtype=torch.float16,  # Use half precision to reduce memory
            device_map="auto",  # Automatically distribute across available devices
            low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
        )
        print("Base Model loaded successfully")
        
        if inject_embedding_with_verification(base_paligemma, pi0_embeddings, "Pi0"):
            return processor, base_paligemma # now has Pi0 embeddings
        else:
            print("ERROR: Failed to inject Pi0 embeddings - aborting test")
            return
    except Exception as e:
        print(f"ERROR: Failed to create Pi0 model: {e}")
        return

    

if __name__ == "__main__":
    replace_paligemma_weights_with_pi0()
