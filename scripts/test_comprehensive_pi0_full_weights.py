#!/usr/bin/env python3
"""
PaliGemma Weight Comparison Test Suite - Full Pi0 Weight Injection
This script compares the text generation capabilities of a base HuggingFace
PaliGemma model versus one with the full PaliGemma weights loaded from a
Pi0-trained checkpoint.
Test Methodology:
1. Load base PaliGemma model from HuggingFace.
2. Load Pi0-trained weights from a checkpoint.
3. Create a detailed mapping between Pi0 (JAX/Flax) and HuggingFace (PyTorch)
   parameter names.
4. Inject the complete set of Pi0 PaliGemma weights (vision encoder and LLM)
   into a separate HuggingFace model instance, including necessary transformations
   like transposing linear layer weights.
5. Test both models on 25 diverse COCO validation images using three different
   prompt types to evaluate text generation robustness.
6. Generate a comparative HTML report with visual analysis.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import requests
from io import BytesIO
import sys

# Add project root to path - same pattern as procgen_inference.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# JAX imports for loading weights
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Transformers imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# OpenPI imports - use full path like procgen_inference.py
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.models import pi0
from src.eval.profiling.openpi.src.openpi.shared import download

# Test dataset: 25 diverse COCO validation images
# Selected for variety across object categories, scenes, and visual complexity
COCO_TEST_IMAGES = [
    {"id": "000000397133", "url": "http://images.cocodataset.org/val2017/000000397133.jpg"},
    {"id": "000000037777", "url": "http://images.cocodataset.org/val2017/000000037777.jpg"},
    {"id": "000000252219", "url": "http://images.cocodataset.org/val2017/000000252219.jpg"},
    {"id": "000000087038", "url": "http://images.cocodataset.org/val2017/000000087038.jpg"},
    {"id": "000000174482", "url": "http://images.cocodataset.org/val2017/000000174482.jpg"},
    {"id": "000000403385", "url": "http://images.cocodataset.org/val2017/000000403385.jpg"},
    {"id": "000000006818", "url": "http://images.cocodataset.org/val2017/000000006818.jpg"},
    {"id": "000000480985", "url": "http://images.cocodataset.org/val2017/000000480985.jpg"},
    {"id": "000000458755", "url": "http://images.cocodataset.org/val2017/000000458755.jpg"},
    {"id": "000000331352", "url": "http://images.cocodataset.org/val2017/000000331352.jpg"},
    {"id": "000000579158", "url": "http://images.cocodataset.org/val2017/000000579158.jpg"},
    {"id": "000000578922", "url": "http://images.cocodataset.org/val2017/000000578922.jpg"},
    {"id": "000000472375", "url": "http://images.cocodataset.org/val2017/000000472375.jpg"},
    {"id": "000000013177", "url": "http://images.cocodataset.org/val2017/000000013177.jpg"},
    {"id": "000000544519", "url": "http://images.cocodataset.org/val2017/000000544519.jpg"},
    {"id": "000000334555", "url": "http://images.cocodataset.org/val2017/000000334555.jpg"},
    {"id": "000000502136", "url": "http://images.cocodataset.org/val2017/000000502136.jpg"},
    {"id": "000000520077", "url": "http://images.cocodataset.org/val2017/000000520077.jpg"},
    {"id": "000000184791", "url": "http://images.cocodataset.org/val2017/000000184791.jpg"},
    {"id": "000000050326", "url": "http://images.cocodataset.org/val2017/000000050326.jpg"},
    {"id": "000000315257", "url": "http://images.cocodataset.org/val2017/000000315257.jpg"},
    {"id": "000000226111", "url": "http://images.cocodataset.org/val2017/000000226111.jpg"},
    {"id": "000000017627", "url": "http://images.cocodataset.org/val2017/000000017627.jpg"},
    {"id": "000000025560", "url": "http://images.cocodataset.org/val2017/000000025560.jpg"},
    {"id": "000000036494", "url": "http://images.cocodataset.org/val2017/000000036494.jpg"}
]

# Three distinct prompt types for comprehensive evaluation
PROMPT_TYPES = {
    "basic_caption": {
        "template": "<image>caption en",
        "description": "Basic image captioning task",
        "max_tokens": 25
    },
    "detailed_description": {
        "template": "<image>Provide a detailed description of all objects, people, and activities visible in this image. Include spatial relationships, colors, and any notable details in a comprehensive paragraph.",
        "description": "Detailed object and scene description",
        "max_tokens": 75
    },
    "creative_pun": {
        "template": "<image>Create a clever pun or wordplay based on what you see in this image. Be creative and humorous.",
        "description": "Creative text generation requiring wordplay",
        "max_tokens": 40
    }
}

def verify_pi0_weights_available():
    """
    Verify that Pi0 weights can be loaded successfully and return them.
    
    Returns:
        tuple: (bool, dict) - (success, pi0_weights)
    """
    print("STEP 1: Verifying Pi0 weight availability...")
    
    try:
        # Load Pi0-trained weights from checkpoint
        print("  Loading Pi0-trained weights...")
        pi0_params = _model.restore_params(
            download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"),
            restore_type=np.ndarray # Use numpy for easier inspection and manipulation
        )
        pi0_weights = pi0_params.get("PaliGemma", {})
        
        if not pi0_weights:
            print("  ERROR: 'PaliGemma' key not found in Pi0 parameters.")
            return False, None

        # Check a sample weight to confirm loading
        sample_weight_path = 'llm.decoder.layers_0.attn.q_proj.kernel'
        current_level = pi0_weights
        for key in sample_weight_path.split('.'):
            current_level = current_level[key]
        
        print(f"  Sample Pi0 weight ({sample_weight_path}) shape: {current_level.shape}")
        print("  SUCCESS: Pi0 weights loaded successfully")
        return True, pi0_weights
        
    except Exception as e:
        print(f"  ERROR: Failed to load Pi0 weights: {e}")
        return False, None

def print_weight_structure(weights_dict, indent=0):
    """Utility to recursively print the structure of the weight dictionary."""
    for key, value in weights_dict.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_weight_structure(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value.shape} {value.dtype}")

def inject_paligemma_weights(hf_model, pi0_paligemma_weights, *, min_token_overlap: float = 0.4, verbose: bool = True):
    """
    Inject all PaliGemma weights from a Pi0 checkpoint into a Hugging Face model using a
    generic, structure-aware mapping (no hard-coded layer name list).

    Strategy:
    - Flatten JAX (Pi0) tree and HF state_dict
    - Build a shape index for JAX params; allow transpose match for 2D weights
    - Tokenize names into canonical tokens with light-weight synonyms (kernel~weight, scale~weight, embedding~embed)
    - For each HF param, find shape-compatible JAX candidates and score by token overlap; pick best unused
    - Copy weights (with transpose when needed), pad/truncate embeddings if vocab mismatch

    Args:
        hf_model: Hugging Face PaliGemmaForConditionalGeneration (torch) model instance
        pi0_paligemma_weights: dict of JAX/Flax weights under PaliGemma
        min_token_overlap: minimal token overlap score (0..1) to accept a match
        verbose: print detailed mapping decisions

    Returns:
        bool indicating whether mapping likely succeeded (>=90% parameters loaded)
    """
    import re

    def flatten_tree(tree, prefix=""):
        flat = {}
        for k, v in tree.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_tree(v, key))
            else:
                flat[key] = v
        return flat

    def tokenize(name: str) -> list[str]:
        # Split on separators and keep alnum tokens
        parts = re.split(r"[\./:_]+", name)
        tokens = []
        for p in parts:
            # split layer indices from names: layers_12 -> [layers, 12]
            tokens += re.split(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", p)
        tokens = [t.lower() for t in tokens if t and re.search(r"[a-zA-Z]", t)]
        # Synonyms to canonical tokens
        synonyms = {
            "kernel": "weight",
            "scale": "weight",
            "weights": "weight",
            "embedder": "embed",
            "embedding": "embed",
            "embeddings": "embed",
            "input": "in",
            "output": "out",
            "proj": "proj",
            "q": "q",
            "k": "k",
            "v": "v",
            "o": "o",
            "ln": "layernorm",
            "norm": "layernorm",
            "pos": "position",
        }
        tokens = [synonyms.get(t, t) for t in tokens]
        return tokens

    def jaccard(a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # Flatten JAX params under provided subtree (already 'PaliGemma')
    flat_jax = flatten_tree(pi0_paligemma_weights)

    # Build shape index for JAX params
    shape_index = {}
    transposed_shape_index = {}
    jax_shapes = {}
    for jk, jv in flat_jax.items():
        arr = np.array(jv)
        jax_shapes[jk] = tuple(arr.shape)
        shape_index.setdefault(tuple(arr.shape), []).append(jk)
        # For 2D weights, also index transposed shape
        if arr.ndim == 2:
            transposed_shape_index.setdefault((arr.shape[1], arr.shape[0]), []).append(jk)

    # Prepare HF state
    hf_state = hf_model.state_dict()
    used_jax = set()
    loaded = 0

    # Pre-tokenize JAX keys
    jax_key_tokens = {jk: tokenize(jk) for jk in flat_jax.keys()}

    for hf_name, hf_param in hf_state.items():
        target_shape = tuple(hf_param.shape)
        hf_tokens = tokenize(hf_name)

        # Special-case embeddings: allow vocab padding/truncation later
        is_embedding = ("embed_tokens" in hf_name) or (hf_name.endswith("lm_head.weight"))

        # Candidate JAX keys by exact shape or, for 2D, by transposed shape
        candidates = list(shape_index.get(target_shape, []))
        transpose_needed = {jk: False for jk in candidates}
        if not candidates and len(target_shape) == 2:
            t_cands = transposed_shape_index.get(target_shape, [])
            for jk in t_cands:
                candidates.append(jk)
                transpose_needed[jk] = True

        if not candidates:
            if verbose:
                print(f"  --> No shape-compatible candidate for: {hf_name} {target_shape}")
            continue

        # Score candidates by token overlap
        best_jk = None
        best_score = -1.0
        for jk in candidates:
            if jk in used_jax:
                continue
            score = jaccard(hf_tokens, jax_key_tokens[jk])
            if score > best_score:
                best_score = score
                best_jk = jk

        if best_jk is None or best_score < min_token_overlap:
            if verbose:
                print(f"  --> Low token match for: {hf_name} (best={best_score:.2f})")
            continue

        # Fetch JAX weight and adapt if needed
        jax_weight = np.array(flat_jax[best_jk])

        # Embedding vocab alignment
        if is_embedding and jax_weight.ndim == 2:
            hf_vocab, emb_dim = target_shape
            jax_vocab, jax_emb_dim = jax_weight.shape
            if emb_dim != jax_emb_dim and (jax_emb_dim, emb_dim) == target_shape:
                # extremely unlikely, skip
                pass
            if hf_vocab > jax_vocab:
                if verbose:
                    print(f"  Padding vocab for {hf_name} from {jax_vocab} to {hf_vocab}")
                padded = np.zeros((hf_vocab, jax_emb_dim), dtype=jax_weight.dtype)
                padded[:hf_vocab] = jax_weight
                jax_weight = padded
            elif jax_vocab > hf_vocab:
                if verbose:
                    print(f"  Truncating vocab for {hf_name} from {jax_vocab} to {hf_vocab}")
                jax_weight = jax_weight[:hf_vocab]

        # Transpose linear weights if needed
        if transpose_needed.get(best_jk, False) and jax_weight.ndim == 2:
            jax_weight = jax_weight.T

        # Final shape check
        if tuple(jax_weight.shape) != target_shape:
            if verbose:
                print(f"  --> SHAPE MISMATCH after adaption for {hf_name}: HF {target_shape} vs Pi0 {jax_weight.shape}")
            continue

        # Copy
        with torch.no_grad():
            hf_param.copy_(torch.from_numpy(jax_weight))

        used_jax.add(best_jk)
        loaded += 1
        if verbose:
            print(f"  MAPPED: {hf_name}  <=  {best_jk}  (score={best_score:.2f}{', T' if transpose_needed.get(best_jk, False) else ''})")

    total = len(hf_state)
    print(f"\n  Loaded {loaded} / {total} parameters with generic mapper.")
    if loaded < total * 0.9:
        print("  WARNING: Fewer than 90% parameters loaded; outputs may not be reliable.")
        return False
    return True

def test_model_on_all_images(model, processor, name, output_dir):
    """
    Test model on all images using multiple prompt types.
    This function evaluates the model's text generation capabilities across
    three different prompt types: basic captioning, detailed description,
    and creative pun generation. Each prompt type tests different aspects
    of the model's linguistic and creative abilities.
    Args:
        model: PyTorch PaliGemma model to test
        processor: HuggingFace processor for the model
        name: Descriptive name for this test run
        output_dir: Directory to save images and results
    Returns:
        dict: Results organized by prompt type and image
    """
    print(f"\nSTEP 3: Testing {name} on all 25 images with multiple prompts")
    print("=" * 80)

    # Create output directory for images
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    results = {}

    for i, img_info in enumerate(COCO_TEST_IMAGES):
        print(f"\nImage {i+1:2d}/25: {img_info['id']}")

        image_results = {}

        try:
            # Download and save image once
            response = requests.get(img_info['url'], timeout=15)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

            image_path = image_dir / f"{img_info['id']}.jpg"
            pil_image.save(image_path)

            # Test each prompt type
            for prompt_key, prompt_config in PROMPT_TYPES.items():
                print(f"  Testing {prompt_config['description']}...")

                # Generate text with this prompt
                inputs = processor(prompt_config['template'], pil_image, return_tensors="pt")

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=prompt_config['max_tokens'],
                        do_sample=False
                    )

                input_len = inputs["input_ids"].shape[-1]
                generated_text = processor.decode(output[0][input_len:], skip_special_tokens=True)

                # Analyze text quality
                words = generated_text.split()
                unique_words = len(set(words))
                is_repetitive = unique_words <= 2 and len(words) > 2
                is_single_char = all(len(word) == 1 for word in words[:5]) if words else False
                is_empty = len(generated_text.strip()) == 0

                # Detect repetitive patterns (same phrase repeated)
                has_loops = False
                if len(words) > 4:
                    # Check for immediate repetitions
                    for j in range(len(words) - 2):
                        if words[j] == words[j + 2] and words[j + 1] == words[j + 3]:
                            has_loops = True
                            break

                print(f"    Generated: '{generated_text}'")

                # Flag quality issues
                issues = []
                if is_repetitive:
                    issues.append(f"REPETITIVE ({unique_words} unique words)")
                if is_single_char:
                    issues.append("SINGLE CHARACTERS")
                if is_empty:
                    issues.append("EMPTY OUTPUT")
                if has_loops:
                    issues.append("REPETITIVE LOOPS")

                if issues:
                    print(f"    ISSUES: {', '.join(issues)}")

                image_results[prompt_key] = {
                    'generated_text': generated_text,
                    'word_count': len(words),
                    'unique_words': unique_words,
                    'is_repetitive': is_repetitive,
                    'is_single_char': is_single_char,
                    'is_empty': is_empty,
                    'has_loops': has_loops,
                    'prompt_template': prompt_config['template'],
                    'success': True
                }

            image_results['image_id'] = img_info['id']
            image_results['image_path'] = str(image_path)
            results[img_info['id']] = image_results

        except Exception as e:
            print(f"  ERROR: {e}")
            error_result = {
                'image_id': img_info['id'],
                'success': False,
                'error': str(e)
            }
            for prompt_key in PROMPT_TYPES.keys():
                error_result[prompt_key] = {
                    'generated_text': '',
                    'success': False,
                    'error': str(e)
                }
            results[img_info['id']] = error_result

    return results

def print_text_summary(base_results, pi0_results):
    """Print a concise text summary comparing base vs Pi0 results."""
    total_images = len(COCO_TEST_IMAGES)
    total_tests = total_images * len(PROMPT_TYPES)

    stats = {
        'base': {'success': 0, 'repetitive': 0, 'loops': 0, 'empty': 0},
        'pi0': {'success': 0, 'repetitive': 0, 'loops': 0, 'empty': 0}
    }
    for img_id in base_results.keys():
        for prompt_type in PROMPT_TYPES.keys():
            if img_id in base_results and prompt_type in base_results[img_id]:
                r = base_results[img_id][prompt_type]
                if r.get('success', False):
                    stats['base']['success'] += 1
                    stats['base']['repetitive'] += int(r.get('is_repetitive', False))
                    stats['base']['loops'] += int(r.get('has_loops', False))
                    stats['base']['empty'] += int(r.get('is_empty', False))
            if img_id in pi0_results and prompt_type in pi0_results[img_id]:
                r = pi0_results[img_id][prompt_type]
                if r.get('success', False):
                    stats['pi0']['success'] += 1
                    stats['pi0']['repetitive'] += int(r.get('is_repetitive', False))
                    stats['pi0']['loops'] += int(r.get('has_loops', False))
                    stats['pi0']['empty'] += int(r.get('is_empty', False))

    print("\nSummary (Base vs Pi0)")
    print("- Total tests:", total_tests)
    print(
        f"- Base success: {stats['base']['success']}/{total_tests} | "
        f"Repetitive: {stats['base']['repetitive']}, Loops: {stats['base']['loops']}, Empty: {stats['base']['empty']}"
    )
    print(
        f"- Pi0  success: {stats['pi0']['success']}/{total_tests} | "
        f"Repetitive: {stats['pi0']['repetitive']}, Loops: {stats['pi0']['loops']}, Empty: {stats['pi0']['empty']}"
    )

    return stats

def main():
    """
    Main test execution function - Full Pi0 Weight Injection
    This function orchestrates the complete test workflow:
    1. Verify Pi0 weights can be loaded.
    2. Load a base HuggingFace PaliGemma model as a baseline.
    3. Load a second HF model and inject the full Pi0 PaliGemma weights into it.
    4. Test both models on all images with multiple prompt types.
    5. Generate a comprehensive HTML report with visual comparisons.
    """
    print("PaliGemma Weight Comparison Test Suite - Full Pi0 Injection")
    print("=" * 80)
    print("Comparing HuggingFace base PaliGemma vs. full Pi0-trained weights")
    print("Testing text generation capabilities across multiple prompt types")
    print("=" * 80)

    # Create output directory
    timestamp = int(time.time())
    output_dir = Path(f"/tmp/comprehensive_verification_full_pi0_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # Verify Pi0 weights are available
    success, pi0_weights = verify_pi0_weights_available()
    if not success:
        print("ERROR: Pi0 weight verification failed - cannot proceed with test")
        return

    # Optional: Print weight structure for debugging the mapping
    # print_weight_structure(pi0_weights)

    # Load transformers components
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"SUCCESS: Loaded processor for {model_id}")

    # Test base model (HuggingFace default)
    print("\n" + "="*70)
    print("PHASE 1: Testing Base PaliGemma (HuggingFace Default)")
    print("="*70)

    model_base = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    base_results = test_model_on_all_images(model_base, processor, "Base PaliGemma (HF)", output_dir)

    # Test Pi0-injected model
    print("\n" + "="*70)
    print("PHASE 2: Testing Pi0-Trained Weights (Full Injection)")
    print("="*70)

    model_pi0_hf = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    if inject_paligemma_weights(model_pi0_hf, pi0_weights):
        pi0_results = test_model_on_all_images(model_pi0_hf, processor, "Pi0 (Full Injection)", output_dir)
    else:
        print("ERROR: Failed to inject Pi0 weights - aborting test")
        return

    # Print simple summary instead of HTML report
    summary_stats = print_text_summary(base_results, pi0_results)

    # Save comprehensive results
    all_results = {
        'test_info': {
            'timestamp': time.ctime(),
            'model_id': model_id,
            'num_images': len(COCO_TEST_IMAGES),
            'num_prompt_types': len(PROMPT_TYPES),
            'test_type': 'full_pi0_weight_injection',
        },
        'prompt_types': PROMPT_TYPES,
        'base_results': base_results,
        'pi0_results': pi0_results
    }

    json_file = output_dir / "detailed_results_full_pi0.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("TEST SUITE COMPLETED SUCCESSFULLY - FULL PI0 INJECTION")
    print("="*70)
    print(f"JSON Results: {json_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Total Tests Run: {len(COCO_TEST_IMAGES) * len(PROMPT_TYPES) * 2} individual generations")
    print("Summary:")
    print(
        f"  Base success {summary_stats['base']['success']}, Pi0 success {summary_stats['pi0']['success']}"
    )
    print("="*70)

if __name__ == "__main__":
    main()
