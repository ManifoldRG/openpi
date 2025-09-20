#!/usr/bin/env python3
"""
Layer-by-Layer Corruption Analysis for Pi0 PaliGemma

This script systematically replaces individual layers of the PaliGemma model
with Pi0-trained weights to identify exactly which layers are corrupted.

Based on test_comprehensive_25_image_verification.py
"""

import json
import os
import tempfile
import time
from pathlib import Path
import requests
from io import BytesIO
import urllib3

# Disable SSL warnings for COCO dataset
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import jax
import jax.numpy as jnp
import numpy as np
import torch
import gc
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# Import openpi modules
import sys
sys.path.append('src')

from openpi.models import pi0
from openpi.shared import download
from openpi.models import model as _model
from openpi.training.weight_loaders import PaliGemmaWeightLoader
from flax import nnx

# Test configuration
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

# Test images (using first 5 for faster iteration)
COCO_TEST_IMAGES = [
    {"id": "000000397133"},
    {"id": "000000037777"},
    {"id": "000000252219"},
    {"id": "000000087038"},
    {"id": "000000174482"}
]

def inject_single_language_layer(state_dict, jax_layers, target_layer_idx, component=None):
    """
    Inject weights for a single language model layer or component.

    Args:
        state_dict: PyTorch model state dict
        jax_layers: JAX language model layers
        target_layer_idx: Which layer to inject (0-17)
        component: Specific component to inject ('embeddings', 'attention', 'mlp', 'layernorm', 'final_norm')
                  If None, inject the entire layer

    Returns:
        int: Number of weight groups injected
    """
    injection_count = 0

    if component == 'embeddings':
        # Inject only embeddings
        if 'embedder' in jax_layers and 'input_embedding' in jax_layers['embedder']:
            jax_embeddings = jax_layers['embedder']['input_embedding']
            pytorch_vocab_size = 257216
            jax_vocab_size = jax_embeddings.shape[0]

            if pytorch_vocab_size >= jax_vocab_size:
                padded_embeddings = np.zeros((pytorch_vocab_size, 2048), dtype=jax_embeddings.dtype)
                padded_embeddings[:jax_vocab_size] = jax_embeddings
                inject_weight(state_dict, 'language_model.model.embed_tokens.weight', padded_embeddings)
            else:
                inject_weight(state_dict, 'language_model.model.embed_tokens.weight', jax_embeddings[:pytorch_vocab_size])

            injection_count += 1

    elif component == 'final_norm':
        # Inject final layer norm
        if 'final_norm' in jax_layers:
            inject_weight(state_dict, 'language_model.model.norm.weight', jax_layers['final_norm']['scale'])
            injection_count += 1

    elif component in ['attention', 'mlp', 'layernorm', None]:
        # Inject specific layer components
        if 'layers' in jax_layers and target_layer_idx < 18:
            layers = jax_layers['layers']
            prefix = f'language_model.model.layers.{target_layer_idx}'

            # Attention weights
            if component in ['attention', None] and 'attn' in layers:
                attn = layers['attn']

                # Query projection
                if 'q_einsum' in attn:
                    q_weight = attn['q_einsum']['w'][target_layer_idx]
                    q_weight_flat = np.reshape(q_weight.transpose(1, 0, 2), (2048, 2048))
                    inject_weight(state_dict, f'{prefix}.self_attn.q_proj.weight', q_weight_flat)
                    injection_count += 1

                # Key/Value projections
                if 'kv_einsum' in attn:
                    kv_weight = attn['kv_einsum']['w'][target_layer_idx]
                    k_weight = kv_weight[0, 0]
                    v_weight = kv_weight[1, 0]
                    inject_weight(state_dict, f'{prefix}.self_attn.k_proj.weight', k_weight.transpose(1, 0))
                    inject_weight(state_dict, f'{prefix}.self_attn.v_proj.weight', v_weight.transpose(1, 0))
                    injection_count += 2

                # Output projection
                if 'attn_vec_einsum' in attn:
                    out_weight = attn['attn_vec_einsum']['w'][target_layer_idx]
                    out_weight_flat = np.reshape(out_weight, (2048, 2048))
                    inject_weight(state_dict, f'{prefix}.self_attn.o_proj.weight', out_weight_flat)
                    injection_count += 1

            # MLP weights
            if component in ['mlp', None] and 'mlp' in layers:
                mlp = layers['mlp']

                # Gate and Up projections
                if 'gating_einsum' in mlp:
                    gating_weight = mlp['gating_einsum'][target_layer_idx]
                    gate_weight = gating_weight[0]
                    up_weight = gating_weight[1]
                    inject_weight(state_dict, f'{prefix}.mlp.gate_proj.weight', gate_weight.transpose(1, 0))
                    inject_weight(state_dict, f'{prefix}.mlp.up_proj.weight', up_weight.transpose(1, 0))
                    injection_count += 2

                # Down projection
                if 'linear' in mlp:
                    down_weight = mlp['linear'][target_layer_idx]
                    inject_weight(state_dict, f'{prefix}.mlp.down_proj.weight', down_weight.transpose(1, 0))
                    injection_count += 1

            # Layer norms
            if component in ['layernorm', None]:
                if 'pre_attention_norm' in layers:
                    inject_weight(state_dict, f'{prefix}.input_layernorm.weight', layers['pre_attention_norm']['scale'][target_layer_idx])
                    injection_count += 1

                if 'pre_ffw_norm' in layers:
                    inject_weight(state_dict, f'{prefix}.post_attention_layernorm.weight', layers['pre_ffw_norm']['scale'][target_layer_idx])
                    injection_count += 1

    return injection_count

def inject_weight(state_dict, param_name, jax_weight):
    """Helper function to inject a single weight with error handling."""
    if param_name not in state_dict:
        return False

    pytorch_param = state_dict[param_name]
    jax_tensor = torch.from_numpy(np.array(jax_weight))

    if pytorch_param.shape != jax_tensor.shape:
        print(f"    WARNING: Shape mismatch for {param_name}: {pytorch_param.shape} vs {jax_tensor.shape}")
        return False

    with torch.no_grad():
        pytorch_param.copy_(jax_tensor)

    return True

def test_model_quick(model, processor, name, output_dir):
    """Quick test on reduced image set with all prompt types."""
    results = {}

    for i, img_data in enumerate(COCO_TEST_IMAGES):
        img_id = img_data["id"]
        image_url = f"https://images.cocodataset.org/val2017/{img_id}.jpg"

        try:
            # Download image from URL (disable SSL verification for COCO)
            response = requests.get(image_url, timeout=10, verify=False)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')

            results[img_id] = {
                'image_path': image_url,
                'prompts': {}
            }

            print(f"Image {i+1}/{len(COCO_TEST_IMAGES)}: {img_id}")

            for prompt_key, prompt_config in PROMPT_TYPES.items():
                try:
                    prompt = prompt_config['template']
                    inputs = processor(prompt, image, return_tensors="pt")

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=prompt_config['max_tokens'],
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id
                        )

                    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
                    # Remove the prompt part
                    if "<image>" in generated_text:
                        generated_text = generated_text.split("<image>", 1)[1]
                        if prompt_config['template'].replace("<image>", "") in generated_text:
                            generated_text = generated_text.replace(prompt_config['template'].replace("<image>", ""), "").strip()

                    results[img_id]['prompts'][prompt_key] = {
                        'generated': generated_text,
                        'template': prompt_config['template'],
                        'description': prompt_config['description']
                    }

                    print(f"  {prompt_config['description']}: '{generated_text[:50]}{'...' if len(generated_text) > 50 else ''}'")

                except Exception as e:
                    print(f"  ERROR in {prompt_key}: {e}")
                    results[img_id]['prompts'][prompt_key] = {
                        'generated': f"ERROR: {str(e)}",
                        'template': prompt_config['template'],
                        'description': prompt_config['description']
                    }

        except Exception as e:
            print(f"  ERROR loading image {img_id}: {e}")
            continue

    return results

def analyze_corruption_patterns(results):
    """Analyze the results to identify corruption patterns."""
    corruption_indicators = {
        'repetitive_tokens': 0,
        'nonsense_words': 0,
        'empty_responses': 0,
        'error_responses': 0,
        'total_responses': 0
    }

    nonsense_patterns = ['increa', 'makita', 'palab', 'centrif', 'tanong', 'kani', 'accla']

    for img_id, img_results in results.items():
        for prompt_key, prompt_result in img_results.get('prompts', {}).items():
            generated = prompt_result.get('generated', '')
            corruption_indicators['total_responses'] += 1

            # Check for repetitive tokens
            words = generated.split()
            if len(words) > 3:
                unique_words = len(set(words))
                if unique_words / len(words) < 0.3:  # Less than 30% unique words
                    corruption_indicators['repetitive_tokens'] += 1

            # Check for nonsense words
            if any(pattern in generated.lower() for pattern in nonsense_patterns):
                corruption_indicators['nonsense_words'] += 1

            # Check for empty responses
            if len(generated.strip()) == 0:
                corruption_indicators['empty_responses'] += 1

            # Check for errors
            if generated.startswith('ERROR:'):
                corruption_indicators['error_responses'] += 1

    return corruption_indicators

def main():
    print("Layer-by-Layer Corruption Analysis")
    print("="*80)
    print("Systematically testing individual layer corruption in Pi0 PaliGemma")
    print("="*80)

    # Setup
    model_id = "google/paligemma-3b-pt-224"
    output_dir = Path(tempfile.mkdtemp(prefix="layer_corruption_"))
    print(f"Output directory: {output_dir}")

    # Load processor
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    print(f"SUCCESS: Loaded processor for {model_id}")

    # Load weight sets
    print("\nLoading weight sets from checkpoints...")

    # Load Pi0 weights
    pi0_params = _model.restore_params(
        download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")
    )
    pi0_llm = pi0_params.get("PaliGemma", {}).get("llm", {})

    print("SUCCESS: Pi0 weights loaded")

    # Test configurations - focused on key layers to manage memory
    test_cases = [
        {"name": "Base Model", "inject": None},
        {"name": "Pi0 Embeddings Only", "inject": "embeddings"},
        {"name": "Pi0 Final Norm Only", "inject": "final_norm"},
    ]

    # Test critical layers (0-5) with complete replacement to identify corruption spread
    critical_layers = [0, 1, 2, 3, 4, 5]
    for layer_idx in critical_layers:
        test_cases.extend([
            {"name": f"Pi0 Layer {layer_idx} Complete", "inject": f"layer_{layer_idx}_complete"},
        ])

    # Test middle layers (6, 9, 12, 15) to sample throughout the network
    sample_layers = [6, 9, 12, 15]
    for layer_idx in sample_layers:
        test_cases.extend([
            {"name": f"Pi0 Layer {layer_idx} Complete", "inject": f"layer_{layer_idx}_complete"},
        ])

    # Test final layers (16, 17) to see if corruption spreads
    final_layers = [16, 17]
    for layer_idx in final_layers:
        test_cases.extend([
            {"name": f"Pi0 Layer {layer_idx} Complete", "inject": f"layer_{layer_idx}_complete"},
        ])

    all_results = {}
    corruption_analysis = {}

    for i, test_case in enumerate(test_cases):
        print(f"\n" + "="*60)
        print(f"TEST {i+1}/{len(test_cases)}: {test_case['name']}")
        print("="*60)

        # Load fresh model
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        # Apply injection if specified
        if test_case['inject'] is not None:
            injection_count = 0

            if test_case['inject'] == "embeddings":
                injection_count = inject_single_language_layer(model.state_dict(), pi0_llm, 0, 'embeddings')
            elif test_case['inject'] == "final_norm":
                injection_count = inject_single_language_layer(model.state_dict(), pi0_llm, 0, 'final_norm')
            elif test_case['inject'].startswith("layer_"):
                parts = test_case['inject'].split('_')
                layer_idx = int(parts[1])
                component = parts[2] if parts[2] != 'complete' else None
                injection_count = inject_single_language_layer(model.state_dict(), pi0_llm, layer_idx, component)

            print(f"  Injected {injection_count} weight groups")
        else:
            print("  Using base model weights")

        # Run test
        results = test_model_quick(model, processor, test_case['name'], output_dir)
        all_results[test_case['name']] = results

        # Analyze corruption
        corruption_stats = analyze_corruption_patterns(results)
        corruption_analysis[test_case['name']] = corruption_stats

        print(f"  Corruption analysis:")
        print(f"    Repetitive: {corruption_stats['repetitive_tokens']}/{corruption_stats['total_responses']}")
        print(f"    Nonsense: {corruption_stats['nonsense_words']}/{corruption_stats['total_responses']}")
        print(f"    Empty: {corruption_stats['empty_responses']}/{corruption_stats['total_responses']}")
        print(f"    Errors: {corruption_stats['error_responses']}/{corruption_stats['total_responses']}")

        # Track severe corruption but continue testing all layers
        total_bad = (corruption_stats['repetitive_tokens'] +
                    corruption_stats['nonsense_words'] +
                    corruption_stats['empty_responses'] +
                    corruption_stats['error_responses'])

        if total_bad >= corruption_stats['total_responses'] * 0.8:  # 80% corruption
            print(f"  SEVERE CORRUPTION DETECTED - continuing to test other layers")

        # Clean up memory after each test
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        print(f"  Memory cleaned up")

    # Save comprehensive results
    final_results = {
        'test_info': {
            'timestamp': time.ctime(),
            'model_id': model_id,
            'num_images': len(COCO_TEST_IMAGES),
            'num_prompt_types': len(PROMPT_TYPES),
            'test_type': 'layer_by_layer_corruption_analysis'
        },
        'test_results': all_results,
        'corruption_analysis': corruption_analysis
    }

    json_file = output_dir / "layer_corruption_analysis.json"
    with open(json_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("LAYER CORRUPTION ANALYSIS COMPLETED")
    print("="*60)
    print(f"Results saved to: {json_file}")
    print(f"Output directory: {output_dir}")

    # Summary of corruption by layer
    print(f"\nCorruption Summary:")
    for test_name, stats in corruption_analysis.items():
        total_bad = (stats['repetitive_tokens'] + stats['nonsense_words'] +
                    stats['empty_responses'] + stats['error_responses'])
        corruption_pct = (total_bad / stats['total_responses'] * 100) if stats['total_responses'] > 0 else 0
        print(f"  {test_name}: {corruption_pct:.1f}% corrupted")

if __name__ == "__main__":
    main()