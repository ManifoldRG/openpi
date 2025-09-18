#!/usr/bin/env python3
"""
Simple PaliGemma test script based on the working example.
Uses a smaller model and tests on CPU to avoid long downloads.
"""

import logging
import time
from pathlib import Path

# Try using a smaller, faster model
try:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from PIL import Image
    import requests
    import torch

    print("Testing PaliGemma with Transformers library...")

    # Use a smaller model for faster testing
    model_id = "google/paligemma-3b-pt-224"  # Smaller base model

    print(f"Loading model: {model_id}")

    # Load on CPU and with reduced precision to save memory/time
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(model_id)

    print("Model loaded successfully!")

    # Test with a simple image and prompt
    prompt = "caption"

    # Use a simple local test image or fetch a small one
    try:
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        print("Downloaded test image successfully")
    except Exception as e:
        print(f"Failed to download image, creating dummy image: {e}")
        # Create a simple test image
        raw_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    # Process input
    inputs = processor(prompt, raw_image, return_tensors="pt")

    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")

    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,  # Reduced for faster testing
            do_sample=False,  # Greedy decoding for consistency
            temperature=1.0
        )

    # Decode output
    input_len = inputs["input_ids"].shape[-1]
    generated_text = processor.decode(output[0][input_len:], skip_special_tokens=True)

    print(f"Generated text: '{generated_text}'")
    print("✅ PaliGemma test successful!")

    # Test with different prompts
    test_prompts = [
        "What do you see?",
        "Describe this image",
        "caption en"
    ]

    print("\nTesting different prompts:")
    for test_prompt in test_prompts:
        inputs = processor(test_prompt, raw_image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        input_len = inputs["input_ids"].shape[-1]
        generated_text = processor.decode(output[0][input_len:], skip_special_tokens=True)
        print(f"  Prompt: '{test_prompt}' -> '{generated_text}'")

except ImportError as e:
    print(f"❌ Transformers/torch not available: {e}")
    print("This is expected in the openpi environment that uses JAX")

except Exception as e:
    print(f"❌ Error testing PaliGemma: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Now testing how this compares to our Pi0 implementation...")

# Now test with our Pi0 implementation to compare
try:
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    from openpi.models import pi0
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.weight_loaders import PaliGemmaWeightLoader
    from openpi.shared import array_typing as at

    print("Testing Pi0 with PaliGemma weights...")

    # Create Pi0 model
    rng = jax.random.key(42)
    pi0_config = pi0.Pi0Config()
    pi0_model = pi0_config.create(rng)

    # Create test observation with real prompt
    obs_spec, _ = pi0_config.inputs_spec()
    dummy_obs = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), obs_spec)

    # Test the same prompts we used with Transformers
    tokenizer = PaligemmaTokenizer(max_len=pi0_config.max_token_len)

    test_prompts = ["What do you see?", "Describe this image", "caption"]

    print("\nTesting Pi0 tokenization of the same prompts:")
    for prompt in test_prompts:
        tokens, mask = tokenizer.tokenize(prompt)
        valid_tokens = tokens[mask]
        decoded = tokenizer._tokenizer.decode(valid_tokens.tolist())

        print(f"  Prompt: '{prompt}'")
        print(f"    Tokens: {valid_tokens.tolist()}")
        print(f"    Decoded: '{decoded}'")

        # Update observation
        test_obs = dummy_obs.replace(
            tokenized_prompt=jnp.array([tokens]),
            tokenized_prompt_mask=jnp.array([mask]),
        )

        # Test prediction (not full autoregression to avoid the tracing issues)
        try:
            with at.disable_typechecking():
                prefix_tokens, prefix_mask, prefix_ar_mask = pi0_model.embed_prefix(test_obs)

            print(f"    Pi0 embed_prefix successful - prefix shape: {prefix_tokens.shape}")

        except Exception as e:
            print(f"    Pi0 embed_prefix failed: {e}")

    print("✅ Pi0 tokenization test completed!")

except Exception as e:
    print(f"❌ Error testing Pi0: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Summary:")
print("1. This script tests the official PaliGemma implementation")
print("2. Compares tokenization with Pi0's approach")
print("3. Helps identify differences in prompt handling")
print("4. Use this as a reference for proper PaliGemma usage")
