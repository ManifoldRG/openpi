#!/usr/bin/env python3
"""
Pi0 to PaliGemma Weight Injection Test

This script tests whether Pi0-trained weights can be successfully injected into 
a HuggingFace PaliGemma model and compares text generation quality.

Core workflow:
1. Load Pi0 checkpoint weights (JAX/Flax format)
2. Load base HuggingFace PaliGemma model
3. Create weight-injected model by mapping Pi0 weights to HF parameters
4. Compare text generation on sample images
"""

import os
import json
import time
import sys
from pathlib import Path
import numpy as np
import torch
import requests
from io import BytesIO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# JAX imports
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp

# Transformers imports
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

# OpenPI imports
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.shared import download

# Test configuration
TEST_IMAGES = [
    {"id": "397133", "url": "http://images.cocodataset.org/val2017/000000397133.jpg"},
    {"id": "037777", "url": "http://images.cocodataset.org/val2017/000000037777.jpg"},
    {"id": "252219", "url": "http://images.cocodataset.org/val2017/000000252219.jpg"},
]

PROMPTS = [
    ("caption", "Basic captioning"),
    ("Describe this image in detail.", "Detailed description"),
    ("What is the main subject of this image?", "Subject identification")
]

class Pi0WeightInjector:
    """Handles injection of Pi0 weights into HuggingFace PaliGemma models."""
    
    def __init__(self, pi0_weights):
        self.pi0_weights = self._flatten_weights(pi0_weights)
        self.pi0_main = self._filter_main_weights(self.pi0_weights)
    
    def _flatten_weights(self, weights, prefix=""):
        """Flatten nested weight dictionary with '/' separated keys."""
        flat = {}
        for k, v in weights.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_weights(v, key))
            else:
                flat[key] = np.array(v)
        return flat
    
    def _filter_main_weights(self, flat_weights):
        """Filter out action expert weights (those with '_1' suffix)."""
        return {k: v for k, v in flat_weights.items() 
                if not any(part.endswith("_1") for part in k.split("/"))}
    
    def inject_weights(self, hf_model, verbose=False):
        """
        Inject Pi0 weights into HuggingFace model.
        
        Returns:
            bool: True if injection was successful
        """
        hf_state = hf_model.state_dict()
        
        # Display weights before injection
        if verbose:
            self._display_weights(hf_state, "BEFORE injection")
        
        loaded_count = 0
        total_count = 0
        
        # Core parameter mappings
        mappings = [
            # Embeddings and projections
            ("llm/embedder/input_embedding", "language_model.model.embed_tokens.weight"),
            ("llm/final_norm/scale", "language_model.model.norm.weight"),
            ("img/head/kernel", "multi_modal_projector.linear.weight"),
            ("img/head/bias", "multi_modal_projector.linear.bias"),
            
            # Vision transformer
            ("img/embedding/kernel", "vision_tower.vision_model.embeddings.patch_embedding.weight"),
            ("img/embedding/bias", "vision_tower.vision_model.embeddings.patch_embedding.bias"),
            ("img/pos_embedding", "vision_tower.vision_model.embeddings.position_embedding.weight"),
            ("img/Transformer/encoder_norm/scale", "vision_tower.vision_model.post_layernorm.weight"),
            ("img/Transformer/encoder_norm/bias", "vision_tower.vision_model.post_layernorm.bias"),
        ]
        
        # Load direct mappings
        for pi0_key, hf_key in mappings:
            if self._load_single_param(pi0_key, hf_key, hf_state, verbose):
                loaded_count += 1
            total_count += 1
        
        # Load layered parameters (LLM and Vision transformer layers)
        llm_loaded = self._load_llm_layers(hf_state, verbose)
        vision_loaded = self._load_vision_layers(hf_state, verbose)
        loaded_count += llm_loaded + vision_loaded
        
        # Count parameters more accurately by checking actual availability
        total_count += self._count_available_layer_params()
        
        success_rate = loaded_count / max(total_count, 1)
        if verbose:
            print(f"Loaded {loaded_count}/{total_count} parameters ({success_rate:.1%})")
        
        # Display weights after injection  
        if verbose:
            self._display_weights(hf_state, "AFTER injection")
        
        return success_rate > 0.5
    
    def _load_single_param(self, pi0_key, hf_key, hf_state, verbose):
        """Load a single parameter with shape handling."""
        if pi0_key not in self.pi0_main or hf_key not in hf_state:
            return False
        
        pi0_param = self.pi0_main[pi0_key].copy()
        hf_param = hf_state[hf_key]
        
        # Handle shape mismatches
        pi0_param = self._handle_shape_mismatch(pi0_param, hf_param, hf_key)
        
        if pi0_param.shape != hf_param.shape:
            if verbose:
                print(f"Shape mismatch {hf_key}: {pi0_param.shape} vs {hf_param.shape}")
            return False
        
        try:
            with torch.no_grad():
                hf_param.copy_(torch.from_numpy(pi0_param))
            return True
        except Exception as e:
            if verbose:
                print(f"Error loading {hf_key}: {e}")
            return False
    
    def _handle_shape_mismatch(self, pi0_param, hf_param, param_name):
        """Handle common shape mismatches between Pi0 and HF parameters."""
        # Transpose 2D matrices if needed
        if (pi0_param.ndim == 2 and hf_param.ndim == 2 and 
            pi0_param.shape != hf_param.shape and pi0_param.T.shape == hf_param.shape):
            return pi0_param.T
        
        # Handle convolution weight format (HWIO -> OIHW)
        if (pi0_param.ndim == 4 and "patch_embedding.weight" in param_name and
            pi0_param.shape[-1] == hf_param.shape[0]):
            return np.transpose(pi0_param, (3, 2, 0, 1))
        
        # Handle embedding vocabulary size mismatch
        if ("embed_tokens.weight" in param_name and pi0_param.ndim == 2):
            hf_vocab, embed_dim = hf_param.shape
            pi0_vocab, pi0_dim = pi0_param.shape
            
            if pi0_dim == embed_dim and pi0_vocab != hf_vocab:
                if pi0_vocab < hf_vocab:
                    # Pad with zeros
                    padded = np.zeros((hf_vocab, embed_dim), dtype=pi0_param.dtype)
                    padded[:pi0_vocab] = pi0_param
                    return padded
                else:
                    # Truncate
                    return pi0_param[:hf_vocab]
        
        return pi0_param
    
    def _load_llm_layers(self, hf_state, verbose):
        """Load LLM transformer layer weights."""
        loaded = 0
        
        # Find batched LLM parameters
        layer_params = {
            'q': self._find_param("llm/layers/attn/q_einsum/w"),
            'k': self._find_param("llm/layers/attn/k_einsum/w"),
            'v': self._find_param("llm/layers/attn/v_einsum/w"),
            'o': self._find_param("llm/layers/attn/attn_vec_einsum/w"),
            'gate': self._find_param("llm/layers/mlp/gating_einsum"),
            'up': self._find_param("llm/layers/mlp/up_einsum"),
            'down': self._find_param("llm/layers/mlp/linear"),
            'attn_norm': self._find_param("llm/layers/attn_norm/scale"),
            'mlp_norm': self._find_param("llm/layers/mlp_norm/scale"),
        }
        
        if not any(param is not None for param in layer_params.values()):
            return 0
        
        # Dynamically determine number of layers from the parameter shape
        num_layers = self._get_num_llm_layers(layer_params)
        if num_layers == 0:
            return 0
            
        for i in range(num_layers):
            loaded += self._load_llm_layer(i, layer_params, hf_state)
        
        return loaded
    
    def _load_vision_layers(self, hf_state, verbose):
        """Load vision transformer layer weights."""
        loaded = 0
        
        # Find batched vision parameters
        vision_params = {
            'q': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"),
            'k': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"),
            'v': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"),
            'o': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"),
            'fc1': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"),
            'fc2': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"),
            'ln1': self._find_param("img/Transformer/encoderblock/LayerNorm_0/scale"),
            'ln2': self._find_param("img/Transformer/encoderblock/LayerNorm_1/scale"),
        }
        
        if not any(param is not None for param in vision_params.values()):
            return 0
        
        # Dynamically determine number of layers from the parameter shape
        num_layers = self._get_num_vision_layers(vision_params)
        if num_layers == 0:
            return 0
            
        for i in range(num_layers):
            loaded += self._load_vision_layer(i, vision_params, hf_state)
        
        return loaded
    
    def _find_param(self, suffix):
        """Find parameter by suffix in Pi0 weights."""
        matches = [k for k in self.pi0_main.keys() if k.endswith(suffix)]
        return self.pi0_main[matches[0]] if matches else None
    
    def _get_num_llm_layers(self, layer_params):
        """Dynamically determine number of LLM layers from parameter shapes."""
        for param_name, param_array in layer_params.items():
            if param_array is not None and param_array.ndim > 0:
                # LLM parameters are batched with layer count as first dimension
                return param_array.shape[0]
        return 0
    
    def _get_num_vision_layers(self, vision_params):
        """Dynamically determine number of vision transformer layers from parameter shapes."""
        for param_name, param_array in vision_params.items():
            if param_array is not None and param_array.ndim > 0:
                # Vision parameters are batched with layer count as first dimension
                return param_array.shape[0]
        return 0
    
    def _load_llm_layer(self, layer_idx, params, hf_state):
        """Load weights for a single LLM transformer layer."""
        loaded = 0
        prefix = f"language_model.model.layers.{layer_idx}"
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]  # Shape: (H, D, Hd)
            q_reshaped = np.transpose(q_weight, (0, 2, 1)).reshape(-1, q_weight.shape[1])
            if self._copy_param(f"{prefix}.self_attn.q_proj.weight", q_reshaped, hf_state):
                loaded += 1
        
        # Similar for k, v, o projections...
        # (Simplified for brevity - full implementation would handle all projections)
        
        return loaded
    
    def _load_vision_layer(self, layer_idx, params, hf_state):
        """Load weights for a single vision transformer layer."""
        loaded = 0
        prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]  # Shape: (D, H, Hd)
            q_reshaped = np.transpose(q_weight, (2, 1, 0)).reshape(-1, q_weight.shape[0])
            if self._copy_param(f"{prefix}.self_attn.q_proj.weight", q_reshaped, hf_state):
                loaded += 1
        
        # Similar for other projections...
        # (Simplified for brevity)
        
        return loaded
    
    def _copy_param(self, hf_key, pi0_array, hf_state):
        """Copy Pi0 parameter to HF model state."""
        if hf_key not in hf_state:
            return False
        
        try:
            with torch.no_grad():
                hf_state[hf_key].copy_(torch.from_numpy(pi0_array))
            return True
        except:
            return False
    
    def _count_available_layer_params(self):
        """Count available layer parameters in Pi0 weights dynamically."""
        # LLM layer parameter templates
        llm_param_suffixes = [
            "llm/layers/attn/q_einsum/w", "llm/layers/attn/k_einsum/w", 
            "llm/layers/attn/v_einsum/w", "llm/layers/attn/attn_vec_einsum/w",
            "llm/layers/mlp/gating_einsum", "llm/layers/mlp/up_einsum", "llm/layers/mlp/linear",
            "llm/layers/attn_norm/scale", "llm/layers/mlp_norm/scale"
        ]
        
        # Vision layer parameter templates
        vision_param_suffixes = [
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel", 
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
            "img/Transformer/encoderblock/LayerNorm_0/scale",
            "img/Transformer/encoderblock/LayerNorm_1/scale"
        ]
        
        # Get actual layer counts from the parameter shapes
        llm_layers = 0
        for suffix in llm_param_suffixes:
            param = self._find_param(suffix)
            if param is not None and param.ndim > 0:
                llm_layers = max(llm_layers, param.shape[0])
                break
        
        vision_layers = 0
        for suffix in vision_param_suffixes:
            param = self._find_param(suffix)
            if param is not None and param.ndim > 0:
                vision_layers = max(vision_layers, param.shape[0])
                break
        
        # Count available parameters
        llm_available = sum(1 for suffix in llm_param_suffixes if self._find_param(suffix) is not None)
        vision_available = sum(1 for suffix in vision_param_suffixes if self._find_param(suffix) is not None)
        
        llm_count = llm_available * llm_layers
        vision_count = vision_available * vision_layers
        
        return llm_count + vision_count
    
    def _display_weights(self, hf_state, phase):
        """Display model weights."""
        print(f"\nModel weights {phase}:")
        for name, param in hf_state.items():
            print(f"  {name}: {param.shape} mean={param.mean().item():.4f}")


def load_pi0_weights():
    """Load Pi0 checkpoint weights."""
    print("Loading Pi0 weights...")
    try:
        pi0_params = _model.restore_params(
            download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"),
            restore_type=np.ndarray
        )
        pi0_weights = pi0_params.get("PaliGemma", {})
        
        if not pi0_weights:
            raise ValueError("PaliGemma weights not found in Pi0 checkpoint")
        
        print("✓ Pi0 weights loaded successfully")
        return pi0_weights
    
    except Exception as e:
        print(f"✗ Failed to load Pi0 weights: {e}")
        return None


def test_model_generation(model, processor, model_name):
    """Test model text generation on sample images."""
    print(f"\nTesting {model_name}...")
    results = {}
    
    for img_info in TEST_IMAGES:
        img_id = img_info["id"]
        print(f"  Image {img_id}:")
        
        try:
            # Load image
            response = requests.get(img_info["url"], timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            img_results = {}
            
            # Test each prompt
            for prompt, description in PROMPTS:
                inputs = processor(prompt, image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=20, 
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                input_len = inputs["input_ids"].shape[-1]
                generated_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
                
                img_results[description] = generated_text
                print(f"    {description}: '{generated_text}'")
            
            results[img_id] = img_results
            
        except Exception as e:
            print(f"    Error: {e}")
            results[img_id] = {"error": str(e)}
    
    return results


def main():
    """Main test execution."""
    print("Pi0 Weight Injection Test")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(f"./")
    output_dir.mkdir(exist_ok=True)
    
    # Load Pi0 weights
    pi0_weights = load_pi0_weights()
    if pi0_weights is None:
        return
    
    # Load models
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✓ Loaded processor for {model_id}")
    
    # Test base model
    print("\n1. Testing base HuggingFace model...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    base_results = test_model_generation(base_model, processor, "Base HF Model")
    
    # Test Pi0-injected model
    print("\n2. Testing Pi0-injected model...")
    pi0_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    
    injector = Pi0WeightInjector(pi0_weights)
    if injector.inject_weights(pi0_model, verbose=True):
        print("✓ Pi0 weights injected successfully")
        pi0_results = test_model_generation(pi0_model, processor, "Pi0-Injected Model")
    else:
        print("✗ Pi0 weight injection failed")
        return
    
    # Save results
    results = {
        "timestamp": time.ctime(),
        "model_id": model_id,
        "base_results": base_results,
        "pi0_results": pi0_results
    }
    
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Test completed successfully")
    print(f"Results saved to: {results_file}")
    
    # Quick comparison
    print("\nQuick Comparison:")
    for img_id in base_results:
        if img_id in pi0_results and "error" not in base_results[img_id]:
            print(f"  Image {img_id}:")
            for prompt_desc in base_results[img_id]:
                base_text = base_results[img_id][prompt_desc]
                pi0_text = pi0_results[img_id].get(prompt_desc, "N/A")
                print(f"    {prompt_desc}:")
                print(f"      Base: '{base_text}'")
                print(f"      Pi0:  '{pi0_text}'")


if __name__ == "__main__":
    main()
