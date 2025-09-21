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
COCO_TEST_IMAGES = [
    {"id": "000000397133"},
    {"id": "000000037777"},
    {"id": "000000252219"},
    {"id": "000000087038"},
    {"id": "000000174482"}
]

# Generate URLs for the test images
TEST_IMAGES = [
    {"id": img["id"], "url": f"http://images.cocodataset.org/val2017/{img['id']}.jpg"}
    for img in COCO_TEST_IMAGES
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
        self.weight_changes = {}  # Track weight changes during injection
        self.replacement_log = []  # Track all replacements with detailed info
        self.failed_replacements = []  # Track failed replacement attempts
        self.replacement_summary = {
            'direct_mappings': [],
            'llm_layers': [],
            'vision_layers': [],
            'skipped_components': []
        }
    
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
            print(f"\nAvailable Pi0 parameters:")
            for i, (key, value) in enumerate(sorted(self.pi0_main.items())):
                if i < 20:  # Show first 20 for debugging
                    print(f"  {key}: {value.shape}")
                elif i == 20:
                    print(f"  ... and {len(self.pi0_main) - 20} more parameters")
                    break
        
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
            success = self._load_single_param(pi0_key, hf_key, hf_state, verbose)
            
            # Track replacement attempt
            self.replacement_log.append({
                'type': 'direct_mapping',
                'pi0_key': pi0_key,
                'hf_key': hf_key,
                'success': success,
                'pi0_shape': self.pi0_main.get(pi0_key, {}).shape if pi0_key in self.pi0_main else None,
                'hf_shape': hf_state.get(hf_key, {}).shape if hf_key in hf_state else None
            })
            
            if success:
                loaded_count += 1
                self.replacement_summary['direct_mappings'].append({
                    'pi0_key': pi0_key,
                    'hf_key': hf_key,
                    'pi0_shape': self.pi0_main[pi0_key].shape,
                    'hf_shape': hf_state[hf_key].shape
                })
            else:
                self.failed_replacements.append({
                    'type': 'direct_mapping',
                    'pi0_key': pi0_key,
                    'hf_key': hf_key,
                    'reason': 'Parameter not found or shape mismatch'
                })
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
        # Handle position embedding shape mismatch (squeeze extra dimension)
        if ("position_embedding.weight" in param_name and 
            pi0_param.ndim == 3 and hf_param.ndim == 2 and
            pi0_param.shape[0] == 1 and pi0_param.shape[1:] == hf_param.shape):
            return np.squeeze(pi0_param, axis=0)
        
        # Handle MLP gate/up projection shape mismatch: (2, D, H) -> (H, D)
        if (("gate_proj.weight" in param_name or "up_proj.weight" in param_name) and 
            pi0_param.ndim == 3 and hf_param.ndim == 2 and
            pi0_param.shape[0] == 2 and pi0_param.shape[1:] == hf_param.shape[::-1]):
            # Take first slice and transpose: (2, 2048, 16384) -> (2048, 16384) -> (16384, 2048)
            return pi0_param[0].T
        
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
            loaded += self._load_llm_layer(i, layer_params, hf_state, verbose)
        
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
            'q_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"),
            'k_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"),
            'v_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"),
            'o_bias': self._find_param("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"),
            'fc1': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"),
            'fc2': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"),
            'fc1_bias': self._find_param("img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"),
            'ln1': self._find_param("img/Transformer/encoderblock/LayerNorm_0/scale"),
            'ln2': self._find_param("img/Transformer/encoderblock/LayerNorm_1/scale"),
            'ln1_bias': self._find_param("img/Transformer/encoderblock/LayerNorm_0/bias"),
        }
        
        if not any(param is not None for param in vision_params.values()):
            return 0
        
        # Dynamically determine number of layers from the parameter shape
        num_layers = self._get_num_vision_layers(vision_params)
        if num_layers == 0:
            return 0
            
        for i in range(num_layers):
            loaded += self._load_vision_layer(i, vision_params, hf_state, verbose)
        
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
    
    def _load_llm_layer(self, layer_idx, params, hf_state, verbose=False):
        """Load weights for a single LLM transformer layer."""
        loaded = 0
        prefix = f"language_model.model.layers.{layer_idx}"
        layer_replacements = []
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]  # Shape: (H, D, Hd)
            q_reshaped = np.transpose(q_weight, (0, 2, 1)).reshape(-1, q_weight.shape[1])
            hf_key = f"{prefix}.self_attn.q_proj.weight"
            success = self._copy_param(hf_key, q_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'q_proj',
                'pi0_shape': q_weight.shape,
                'hf_shape': q_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['k'] is not None:
            k_weight = params['k'][layer_idx]
            k_reshaped = np.transpose(k_weight, (0, 2, 1)).reshape(-1, k_weight.shape[1])
            hf_key = f"{prefix}.self_attn.k_proj.weight"
            success = self._copy_param(hf_key, k_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'k_proj',
                'pi0_shape': k_weight.shape,
                'hf_shape': k_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['v'] is not None:
            v_weight = params['v'][layer_idx]
            v_reshaped = np.transpose(v_weight, (0, 2, 1)).reshape(-1, v_weight.shape[1])
            hf_key = f"{prefix}.self_attn.v_proj.weight"
            success = self._copy_param(hf_key, v_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'v_proj',
                'pi0_shape': v_weight.shape,
                'hf_shape': v_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['o'] is not None:
            o_weight = params['o'][layer_idx]
            o_reshaped = o_weight.reshape(-1, o_weight.shape[-1])
            hf_key = f"{prefix}.self_attn.o_proj.weight"
            success = self._copy_param(hf_key, o_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'o_proj',
                'pi0_shape': o_weight.shape,
                'hf_shape': o_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # MLP projections
        if params['gate'] is not None:
            gate_weight = params['gate'][layer_idx]
            hf_key = f"{prefix}.mlp.gate_proj.weight"
            success = self._copy_param(hf_key, gate_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'gate_proj',
                'pi0_shape': gate_weight.shape,
                'hf_shape': gate_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['up'] is not None:
            up_weight = params['up'][layer_idx]
            hf_key = f"{prefix}.mlp.up_proj.weight"
            success = self._copy_param(hf_key, up_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'up_proj',
                'pi0_shape': up_weight.shape,
                'hf_shape': up_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['down'] is not None:
            down_weight = params['down'][layer_idx]
            hf_key = f"{prefix}.mlp.down_proj.weight"
            success = self._copy_param(hf_key, down_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'down_proj',
                'pi0_shape': down_weight.shape,
                'hf_shape': down_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Layer norms
        if params['attn_norm'] is not None:
            attn_norm_weight = params['attn_norm'][layer_idx]
            hf_key = f"{prefix}.input_layernorm.weight"
            success = self._copy_param(hf_key, attn_norm_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'input_layernorm',
                'pi0_shape': attn_norm_weight.shape,
                'hf_shape': attn_norm_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['mlp_norm'] is not None:
            mlp_norm_weight = params['mlp_norm'][layer_idx]
            hf_key = f"{prefix}.post_attention_layernorm.weight"
            success = self._copy_param(hf_key, mlp_norm_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'post_attention_layernorm',
                'pi0_shape': mlp_norm_weight.shape,
                'hf_shape': mlp_norm_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Record layer summary
        self.replacement_summary['llm_layers'].append({
            'layer_idx': layer_idx,
            'components': layer_replacements,
            'total_components': len(layer_replacements),
            'successful_components': sum(1 for r in layer_replacements if r['success'])
        })
        
        return loaded
    
    def _load_vision_layer(self, layer_idx, params, hf_state, verbose=False):
        """Load weights for a single vision transformer layer."""
        loaded = 0
        prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
        layer_replacements = []
        
        # Attention projections
        if params['q'] is not None:
            q_weight = params['q'][layer_idx]  # Shape: (D, H, Hd)
            q_reshaped = np.transpose(q_weight, (2, 1, 0)).reshape(-1, q_weight.shape[0])
            hf_key = f"{prefix}.self_attn.q_proj.weight"
            success = self._copy_param(hf_key, q_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'q_proj',
                'pi0_shape': q_weight.shape,
                'hf_shape': q_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['k'] is not None:
            k_weight = params['k'][layer_idx]
            k_reshaped = np.transpose(k_weight, (2, 1, 0)).reshape(-1, k_weight.shape[0])
            hf_key = f"{prefix}.self_attn.k_proj.weight"
            success = self._copy_param(hf_key, k_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'k_proj',
                'pi0_shape': k_weight.shape,
                'hf_shape': k_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['v'] is not None:
            v_weight = params['v'][layer_idx]
            v_reshaped = np.transpose(v_weight, (2, 1, 0)).reshape(-1, v_weight.shape[0])
            hf_key = f"{prefix}.self_attn.v_proj.weight"
            success = self._copy_param(hf_key, v_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'v_proj',
                'pi0_shape': v_weight.shape,
                'hf_shape': v_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['o'] is not None:
            o_weight = params['o'][layer_idx]
            o_reshaped = o_weight.reshape(-1, o_weight.shape[-1])
            hf_key = f"{prefix}.self_attn.out_proj.weight"
            success = self._copy_param(hf_key, o_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'out_proj',
                'pi0_shape': o_weight.shape,
                'hf_shape': o_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Attention biases
        if params['q_bias'] is not None:
            q_bias = params['q_bias'][layer_idx]  # Shape: (H, Hd)
            q_bias_reshaped = q_bias.reshape(-1)
            hf_key = f"{prefix}.self_attn.q_proj.bias"
            success = self._copy_param(hf_key, q_bias_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'q_proj_bias',
                'pi0_shape': q_bias.shape,
                'hf_shape': q_bias_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['k_bias'] is not None:
            k_bias = params['k_bias'][layer_idx]
            k_bias_reshaped = k_bias.reshape(-1)
            hf_key = f"{prefix}.self_attn.k_proj.bias"
            success = self._copy_param(hf_key, k_bias_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'k_proj_bias',
                'pi0_shape': k_bias.shape,
                'hf_shape': k_bias_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['v_bias'] is not None:
            v_bias = params['v_bias'][layer_idx]
            v_bias_reshaped = v_bias.reshape(-1)
            hf_key = f"{prefix}.self_attn.v_proj.bias"
            success = self._copy_param(hf_key, v_bias_reshaped, hf_state, verbose)
            layer_replacements.append({
                'component': 'v_proj_bias',
                'pi0_shape': v_bias.shape,
                'hf_shape': v_bias_reshaped.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['o_bias'] is not None:
            o_bias = params['o_bias'][layer_idx]
            hf_key = f"{prefix}.self_attn.out_proj.bias"
            success = self._copy_param(hf_key, o_bias, hf_state, verbose)
            layer_replacements.append({
                'component': 'out_proj_bias',
                'pi0_shape': o_bias.shape,
                'hf_shape': o_bias.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # MLP projections
        if params['fc1'] is not None:
            fc1_weight = params['fc1'][layer_idx]
            hf_key = f"{prefix}.mlp.fc1.weight"
            success = self._copy_param(hf_key, fc1_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'fc1',
                'pi0_shape': fc1_weight.shape,
                'hf_shape': fc1_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['fc2'] is not None:
            fc2_weight = params['fc2'][layer_idx]
            hf_key = f"{prefix}.mlp.fc2.weight"
            success = self._copy_param(hf_key, fc2_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'fc2',
                'pi0_shape': fc2_weight.shape,
                'hf_shape': fc2_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # MLP bias
        if params['fc1_bias'] is not None:
            fc1_bias = params['fc1_bias'][layer_idx]
            hf_key = f"{prefix}.mlp.fc1.bias"
            success = self._copy_param(hf_key, fc1_bias, hf_state, verbose)
            layer_replacements.append({
                'component': 'fc1_bias',
                'pi0_shape': fc1_bias.shape,
                'hf_shape': fc1_bias.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Layer norms
        if params['ln1'] is not None:
            ln1_weight = params['ln1'][layer_idx]
            hf_key = f"{prefix}.layer_norm1.weight"
            success = self._copy_param(hf_key, ln1_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'layer_norm1',
                'pi0_shape': ln1_weight.shape,
                'hf_shape': ln1_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        if params['ln2'] is not None:
            ln2_weight = params['ln2'][layer_idx]
            hf_key = f"{prefix}.layer_norm2.weight"
            success = self._copy_param(hf_key, ln2_weight, hf_state, verbose)
            layer_replacements.append({
                'component': 'layer_norm2',
                'pi0_shape': ln2_weight.shape,
                'hf_shape': ln2_weight.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Layer norm bias
        if params['ln1_bias'] is not None:
            ln1_bias = params['ln1_bias'][layer_idx]
            hf_key = f"{prefix}.layer_norm1.bias"
            success = self._copy_param(hf_key, ln1_bias, hf_state, verbose)
            layer_replacements.append({
                'component': 'layer_norm1_bias',
                'pi0_shape': ln1_bias.shape,
                'hf_shape': ln1_bias.shape,
                'hf_key': hf_key,
                'success': success
            })
            if success:
                loaded += 1
        
        # Record layer summary
        self.replacement_summary['vision_layers'].append({
            'layer_idx': layer_idx,
            'components': layer_replacements,
            'total_components': len(layer_replacements),
            'successful_components': sum(1 for r in layer_replacements if r['success'])
        })
        
        return loaded
    
    def _copy_param(self, hf_key, pi0_array, hf_state, verbose=False):
        """Copy Pi0 parameter to HF model state and track changes."""
        if hf_key not in hf_state:
            if verbose:
                print(f"    Missing HF key: {hf_key}")
            return False
        
        hf_param = hf_state[hf_key]
        
        # Store original weights for comparison
        original_weights = hf_param.clone().detach().cpu().numpy()
        
        # Handle shape mismatches
        pi0_array = self._handle_shape_mismatch(pi0_array, hf_param, hf_key)
        
        if pi0_array.shape != hf_param.shape:
            if verbose:
                print(f"    Shape mismatch {hf_key}: Pi0 {pi0_array.shape} vs HF {hf_param.shape}")
            return False
        
        try:
            with torch.no_grad():
                hf_param.copy_(torch.from_numpy(pi0_array))
            
            # Calculate and store weight differences
            weight_diff = pi0_array - original_weights
            self.weight_changes[hf_key] = {
                'original_mean': float(np.mean(original_weights)),
                'original_std': float(np.std(original_weights)),
                'new_mean': float(np.mean(pi0_array)),
                'new_std': float(np.std(pi0_array)),
                'diff_mean': float(np.mean(weight_diff)),
                'diff_std': float(np.std(weight_diff)),
                'diff_magnitude': float(np.linalg.norm(weight_diff)),
                'relative_change': float(np.linalg.norm(weight_diff) / (np.linalg.norm(original_weights) + 1e-8)),
                'shape': pi0_array.shape
            }
            
            if verbose:
                print(f"    ✓ Loaded {hf_key}: {pi0_array.shape}")
            return True
        except Exception as e:
            if verbose:
                print(f"    ✗ Error loading {hf_key}: {e}")
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
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias",
            "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel",
            "img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias",
            "img/Transformer/encoderblock/LayerNorm_0/scale",
            "img/Transformer/encoderblock/LayerNorm_1/scale",
            "img/Transformer/encoderblock/LayerNorm_0/bias"
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
    
    def analyze_weight_changes(self, verbose=True):
        """Analyze and display weight change statistics."""
        if not self.weight_changes:
            print("No weight changes recorded.")
            return {}
        
        # Calculate overall statistics
        all_relative_changes = [change['relative_change'] for change in self.weight_changes.values()]
        all_diff_magnitudes = [change['diff_magnitude'] for change in self.weight_changes.values()]
        
        stats = {
            'total_parameters_changed': len(self.weight_changes),
            'mean_relative_change': float(np.mean(all_relative_changes)),
            'std_relative_change': float(np.std(all_relative_changes)),
            'max_relative_change': float(np.max(all_relative_changes)),
            'min_relative_change': float(np.min(all_relative_changes)),
            'mean_diff_magnitude': float(np.mean(all_diff_magnitudes)),
            'total_diff_magnitude': float(np.sum(all_diff_magnitudes))
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("WEIGHT CHANGE ANALYSIS")
            print(f"{'='*60}")
            print(f"Total parameters changed: {stats['total_parameters_changed']}")
            print(f"Mean relative change: {stats['mean_relative_change']:.6f}")
            print(f"Std relative change: {stats['std_relative_change']:.6f}")
            print(f"Max relative change: {stats['max_relative_change']:.6f}")
            print(f"Min relative change: {stats['min_relative_change']:.6f}")
            print(f"Mean diff magnitude: {stats['mean_diff_magnitude']:.6f}")
            print(f"Total diff magnitude: {stats['total_diff_magnitude']:.6f}")
            
            # Show top 10 largest changes
            sorted_changes = sorted(self.weight_changes.items(), 
                                  key=lambda x: x[1]['relative_change'], reverse=True)
            
            print(f"\nTop 10 Largest Relative Changes:")
            print(f"{'Parameter':<50} {'Rel Change':<12} {'Diff Mag':<12} {'Shape'}")
            print("-" * 90)
            for param_name, change in sorted_changes[:10]:
                short_name = param_name.split('.')[-2:] if '.' in param_name else [param_name]
                short_name = '.'.join(short_name)
                print(f"{short_name:<50} {change['relative_change']:<12.6f} "
                      f"{change['diff_magnitude']:<12.6f} {str(change['shape'])}")
            
            # Show parameters with smallest changes (likely unchanged)
            print(f"\nTop 10 Smallest Relative Changes:")
            print(f"{'Parameter':<50} {'Rel Change':<12} {'Diff Mag':<12} {'Shape'}")
            print("-" * 90)
            for param_name, change in sorted_changes[-10:]:
                short_name = param_name.split('.')[-2:] if '.' in param_name else [param_name]
                short_name = '.'.join(short_name)
                print(f"{short_name:<50} {change['relative_change']:<12.6f} "
                      f"{change['diff_magnitude']:<12.6f} {str(change['shape'])}")
        
        return stats
    
    def get_layer_change_summary(self):
        """Get a summary of changes by layer type."""
        layer_stats = {}
        
        for param_name, change in self.weight_changes.items():
            # Extract layer type
            if 'language_model' in param_name:
                if 'layers.' in param_name:
                    layer_type = 'llm_layer'
                else:
                    layer_type = 'llm_other'
            elif 'vision_tower' in param_name:
                if 'layers.' in param_name:
                    layer_type = 'vision_layer'
                else:
                    layer_type = 'vision_other'
            elif 'multi_modal_projector' in param_name:
                layer_type = 'projector'
            else:
                layer_type = 'other'
            
            if layer_type not in layer_stats:
                layer_stats[layer_type] = {
                    'count': 0,
                    'total_relative_change': 0,
                    'max_relative_change': 0,
                    'total_diff_magnitude': 0
                }
            
            layer_stats[layer_type]['count'] += 1
            layer_stats[layer_type]['total_relative_change'] += change['relative_change']
            layer_stats[layer_type]['max_relative_change'] = max(
                layer_stats[layer_type]['max_relative_change'], 
                change['relative_change']
            )
            layer_stats[layer_type]['total_diff_magnitude'] += change['diff_magnitude']
        
        # Calculate averages
        for layer_type, stats in layer_stats.items():
            if stats['count'] > 0:
                stats['avg_relative_change'] = stats['total_relative_change'] / stats['count']
                stats['avg_diff_magnitude'] = stats['total_diff_magnitude'] / stats['count']
        
        return layer_stats
    
    def generate_replacement_report(self, save_to_file=True, output_dir=None):
        """Generate a comprehensive report of all weight replacements."""
        if output_dir is None:
            output_dir = Path("./")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PI0 TO HUGGINGFACE PALIGEMMA WEIGHT REPLACEMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {time.ctime()}")
        report_lines.append("")
        
        # Summary statistics
        total_attempted = len(self.replacement_log)
        total_successful = sum(1 for log in self.replacement_log if log['success'])
        total_failed = len(self.failed_replacements)
        
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total replacement attempts: {total_attempted}")
        report_lines.append(f"Successful replacements: {total_successful}")
        report_lines.append(f"Failed replacements: {total_failed}")
        report_lines.append(f"Success rate: {total_successful/max(total_attempted,1)*100:.1f}%")
        report_lines.append("")
        
        # Direct mappings section
        report_lines.append("1. DIRECT PARAMETER MAPPINGS")
        report_lines.append("-" * 50)
        if self.replacement_summary['direct_mappings']:
            for mapping in self.replacement_summary['direct_mappings']:
                report_lines.append(f"✓ {mapping['pi0_key']}")
                report_lines.append(f"  -> {mapping['hf_key']}")
                report_lines.append(f"  Pi0 shape: {mapping['pi0_shape']}")
                report_lines.append(f"  HF shape:  {mapping['hf_shape']}")
                report_lines.append("")
        else:
            report_lines.append("No successful direct mappings found.")
            report_lines.append("")
        
        # LLM layers section
        report_lines.append("2. LANGUAGE MODEL TRANSFORMER LAYERS")
        report_lines.append("-" * 50)
        if self.replacement_summary['llm_layers']:
            for layer in self.replacement_summary['llm_layers']:
                layer_idx = layer['layer_idx']
                success_count = layer['successful_components']
                total_count = layer['total_components']
                
                report_lines.append(f"Layer {layer_idx}: {success_count}/{total_count} components replaced")
                
                for component in layer['components']:
                    status = "✓" if component['success'] else "✗"
                    report_lines.append(f"  {status} {component['component']}: {component['pi0_shape']} -> {component['hf_shape']}")
                    report_lines.append(f"    HF key: {component['hf_key']}")
                
                report_lines.append("")
        else:
            report_lines.append("No LLM layers processed.")
            report_lines.append("")
        
        # Vision layers section  
        report_lines.append("3. VISION TRANSFORMER LAYERS")
        report_lines.append("-" * 50)
        if self.replacement_summary['vision_layers']:
            for layer in self.replacement_summary['vision_layers']:
                layer_idx = layer['layer_idx']
                success_count = layer['successful_components']
                total_count = layer['total_components']
                
                report_lines.append(f"Vision Layer {layer_idx}: {success_count}/{total_count} components replaced")
                
                for component in layer['components']:
                    status = "✓" if component['success'] else "✗"
                    report_lines.append(f"  {status} {component['component']}: {component['pi0_shape']} -> {component['hf_shape']}")
                    report_lines.append(f"    HF key: {component['hf_key']}")
                
                report_lines.append("")
        else:
            report_lines.append("No vision layers processed.")
            report_lines.append("")
        
        # Failed replacements section
        if self.failed_replacements:
            report_lines.append("4. FAILED REPLACEMENTS")
            report_lines.append("-" * 50)
            for failure in self.failed_replacements:
                report_lines.append(f"✗ {failure['pi0_key']} -> {failure['hf_key']}")
                report_lines.append(f"  Reason: {failure['reason']}")
                report_lines.append("")
        
        # Excluded components section
        report_lines.append("5. EXCLUDED COMPONENTS (Action Expert)")
        report_lines.append("-" * 50)
        excluded_count = 0
        for key in self.pi0_weights.keys():
            if any(part.endswith("_1") for part in key.split("/")):
                excluded_count += 1
                if excluded_count <= 10:  # Show first 10
                    report_lines.append(f"  {key}")
        
        if excluded_count > 10:
            report_lines.append(f"  ... and {excluded_count - 10} more action expert components")
        
        report_lines.append(f"\nTotal excluded components: {excluded_count}")
        report_lines.append("")
        
        # Weight change statistics
        if self.weight_changes:
            report_lines.append("6. WEIGHT CHANGE STATISTICS")
            report_lines.append("-" * 50)
            
            all_relative_changes = [change['relative_change'] for change in self.weight_changes.values()]
            all_diff_magnitudes = [change['diff_magnitude'] for change in self.weight_changes.values()]
            
            report_lines.append(f"Parameters changed: {len(self.weight_changes)}")
            report_lines.append(f"Mean relative change: {np.mean(all_relative_changes):.6f}")
            report_lines.append(f"Std relative change: {np.std(all_relative_changes):.6f}")
            report_lines.append(f"Max relative change: {np.max(all_relative_changes):.6f}")
            report_lines.append(f"Min relative change: {np.min(all_relative_changes):.6f}")
            report_lines.append(f"Mean diff magnitude: {np.mean(all_diff_magnitudes):.6f}")
            report_lines.append("")
            
            # Top 5 largest changes
            sorted_changes = sorted(self.weight_changes.items(), 
                                  key=lambda x: x[1]['relative_change'], reverse=True)
            
            report_lines.append("Top 5 Largest Relative Changes:")
            for i, (param_name, change) in enumerate(sorted_changes[:5]):
                short_name = param_name.split('.')[-2:] if '.' in param_name else [param_name]
                short_name = '.'.join(short_name)
                report_lines.append(f"  {i+1}. {short_name}: {change['relative_change']:.6f}")
                report_lines.append(f"     Shape: {change['shape']}, Diff magnitude: {change['diff_magnitude']:.6f}")
            
            report_lines.append("")
        
        # Component type summary
        report_lines.append("7. COMPONENT TYPE SUMMARY")
        report_lines.append("-" * 50)
        
        # Count by component type
        component_types = {}
        for log in self.replacement_log:
            if log['success']:
                if log['type'] == 'direct_mapping':
                    comp_type = log['pi0_key'].split('/')[-1]
                else:
                    comp_type = log.get('component', 'unknown')
                
                if comp_type not in component_types:
                    component_types[comp_type] = 0
                component_types[comp_type] += 1
        
        for comp_type, count in sorted(component_types.items()):
            report_lines.append(f"  {comp_type}: {count} replacements")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Join all lines
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            timestamp = int(time.time())
            report_file = output_dir / f"weight_replacement_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            print(f"\n📊 Weight replacement report saved to: {report_file}")
        
        return report_text


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
                # Prepare inputs properly for PaliGemma
                inputs = processor(images=image, text=prompt, return_tensors="pt")

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

def get_weight_injected_model():
    """Load and return a Pi0 weight-injected Paligemma model."""
    model_id = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✓ Loaded processor for {model_id}")
    
    
    # Test Pi0-injected model
    print("\n2. Testing Pi0-injected model...")
    pi0_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    pi0_weights = load_pi0_weights()
    if pi0_weights is None:
        print("✗ Failed to load Pi0 weights")
        return None, None
    
    injector = Pi0WeightInjector(pi0_weights)
    if injector.inject_weights(pi0_model, verbose=True):
        print("✓ Pi0 weights injected successfully")
        return pi0_model, processor
    else:
        print("✗ Pi0 weight injection failed")
        return None, None



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
        
        # Analyze weight changes
        weight_stats = injector.analyze_weight_changes(verbose=True)
        layer_summary = injector.get_layer_change_summary()
        
        # Generate comprehensive replacement report
        replacement_report = injector.generate_replacement_report(save_to_file=True, output_dir=output_dir)
        
        print(f"\n{'='*60}")
        print("LAYER-WISE CHANGE SUMMARY")
        print(f"{'='*60}")
        for layer_type, stats in layer_summary.items():
            print(f"{layer_type.upper()}:")
            print(f"  Parameters changed: {stats['count']}")
            print(f"  Avg relative change: {stats['avg_relative_change']:.6f}")
            print(f"  Max relative change: {stats['max_relative_change']:.6f}")
            print(f"  Avg diff magnitude: {stats['avg_diff_magnitude']:.6f}")
        
        pi0_results = test_model_generation(pi0_model, processor, "Pi0-Injected Model")
    else:
        print("✗ Pi0 weight injection failed")
        return
    
    # Save results
    results = {
        "timestamp": time.ctime(),
        "model_id": model_id,
        "base_results": base_results,
        "pi0_results": pi0_results,
        "weight_analysis": {
            "overall_stats": weight_stats,
            "layer_summary": layer_summary,
            "detailed_changes": injector.weight_changes,
            "replacement_summary": injector.replacement_summary,
            "replacement_log": injector.replacement_log,
            "failed_replacements": injector.failed_replacements
        }
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
