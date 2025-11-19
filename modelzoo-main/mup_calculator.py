#!/usr/bin/env python3
"""
Kyle's muP Configuration Calculator

Usage:
    python kyle_mup_calculator.py --hidden-size 1024 --base-hidden-size 256 \\
                                  --num-heads 16 --num-kv-groups 4 \\
                                  --num-layers 24 --base-init-std 0.02
"""

import argparse
import math
import json


class MuPCalculator:
    """Calculator for Kyle's muP scaling factors."""
    
    def __init__(
        self,
        hidden_size: int,
        base_hidden_size: float,
        filter_size: int,
        base_filter_size: float,
        num_layers: int,
        num_heads: int,
        num_kv_groups: int = None,
        base_init_std: float = 0.02,
    ):
        """
        Initialize calculator with model configuration.
        
        Args:
            hidden_size: Target model hidden dimension
            base_hidden_size: Base model hidden dimension
            filter_size: Target FFN intermediate dimension
            base_filter_size: Base FFN intermediate dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_kv_groups: Number of KV groups (None for MHA)
            base_init_std: Base initialization std dev
        """
        self.hidden_size = hidden_size
        self.base_hidden_size = base_hidden_size
        self.filter_size = filter_size
        self.base_filter_size = base_filter_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups or num_heads
        self.base_init_std = base_init_std
        
        # Calculate core scaling factors
        self.m = hidden_size / base_hidden_size  # Width multiplier
        self.r = num_heads / self.num_kv_groups  # KV ratio
        self.d = hidden_size // num_heads  # Head dimension
        self.L = num_layers  # Depth
        
    def _truncate(self, value: float, decimals: int = 6) -> float:
        """Truncate float to specified decimals."""
        return round(value, decimals)
    
    def calculate_embedding_scales(self):
        """Calculate embedding layer scaling factors."""
        return {
            'init_std': self._truncate(self.base_init_std / self.m),
            'lr_scale': self._truncate(1.0 / self.m),
            'wd_scale': self._truncate(self.m),
            'output_multiplier': self._truncate(self.m),
        }
    
    def calculate_hidden_scales(self):
        """Calculate hidden layer (Q, FFN input) scaling factors."""
        return {
            'init_std': self._truncate(self.base_init_std / math.sqrt(self.m)),
            'lr_scale': self._truncate(1.0 / self.m),
            'wd_scale': self._truncate(self.m),
            'output_multiplier': 1.0,
        }
    
    def calculate_kv_scales(self):
        """Calculate K/V projection scaling factors."""
        numerator = 1.0 + math.sqrt(self.r)
        denominator = 2.0 * math.sqrt(self.m)
        
        return {
            'init_std': self._truncate(self.base_init_std * numerator / denominator),
            'lr_scale': self._truncate(1.0 / self.m),
            'wd_scale': self._truncate(self.m),
            'output_multiplier': self._truncate(2.0 / numerator),
        }
    
    def calculate_output_scales(self):
        """Calculate output layer scaling factors (with depth scaling)."""
        depth_scale = 1.0 / self.L
        
        return {
            'init_std': self._truncate(self.base_init_std / self.m * depth_scale),
            'lr_scale': self._truncate(1.0 / self.m),
            'wd_scale': self._truncate(self.m),
            'output_multiplier': 1.0,
        }
    
    def calculate_ffn_output_scales(self):
        """Calculate FFN output layer scaling factors (with depth scaling)."""
        depth_scale = 1.0 / self.L
        
        return {
            'init_std': self._truncate(self.base_init_std / math.sqrt(self.m) * depth_scale),
            'lr_scale': self._truncate(1.0 / self.m),
            'wd_scale': self._truncate(self.m),
            'output_multiplier': 1.0,
        }
    
    def calculate_norm_scales(self):
        """Calculate normalization layer scaling factors."""
        return {
            'lr_scale': self._truncate(1.0 / self.m),
        }
    
    def calculate_attention_scale(self):
        """Calculate attention scaling factor."""
        return self._truncate(1.0 / self.d)
    
    def calculate_depth_scale(self):
        """Calculate depth scaling factor."""
        return self._truncate(1.0 / self.L)
    
    def generate_yaml_config(self, base_lr: float = 6e-4):
        """
        Generate YAML configuration snippet.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            Dictionary with YAML configuration
        """
        emb = self.calculate_embedding_scales()
        hid = self.calculate_hidden_scales()
        kv = self.calculate_kv_scales()
        out = self.calculate_output_scales()
        ffn_out = self.calculate_ffn_output_scales()
        
        config = {
            'model': {
                'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_layers,
                'num_heads': self.num_heads,
                'filter_size': self.filter_size,
                
                # muP base dimensions
                'mup_base_hidden_size': self.base_hidden_size,
                'mup_base_filter_size': self.base_filter_size,
                
                # Scaling factors
                'embeddings_scale': emb['output_multiplier'],
                'scale_qk_dot_by_d': True,
                'attention_logits_alpha': 1.0,
                'scale_output_logits_by_d': True,
                'output_logits_alpha': 1.0,
                'depth_scaling_factor': self.calculate_depth_scale(),
                
                # Initializers
                'embedding_initializer': {
                    'name': 'truncated_normal',
                    'mean': 0.0,
                    'std': emb['init_std'],
                    'a': -2.0 * emb['init_std'],
                    'b': 2.0 * emb['init_std'],
                },
                'initializer': {
                    'name': 'truncated_normal',
                    'mean': 0.0,
                    'std': hid['init_std'],
                    'a': -2.0 * hid['init_std'],
                    'b': 2.0 * hid['init_std'],
                },
                'attention_q_initializer': {
                    'name': 'truncated_normal',
                    'mean': 0.0,
                    'std': hid['init_std'],
                    'a': -2.0 * hid['init_std'],
                    'b': 2.0 * hid['init_std'],
                },
                'output_layer_initializer': {
                    'name': 'truncated_normal',
                    'mean': 0.0,
                    'std': out['init_std'],
                    'a': -2.0 * out['init_std'],
                    'b': 2.0 * out['init_std'],
                },
                'ffn_output_layer_initializer': {
                    'name': 'truncated_normal',
                    'mean': 0.0,
                    'std': ffn_out['init_std'],
                    'a': -2.0 * ffn_out['init_std'],
                    'b': 2.0 * ffn_out['init_std'],
                },
            },
            'optimizer': {
                'learning_rate': base_lr,
                'adjust_learning_rate': {
                    'embedding': emb['lr_scale'],
                    'q_decoder_attention': hid['lr_scale'],
                    'k_decoder_attention': kv['lr_scale'],
                    'v_decoder_attention': kv['lr_scale'],
                    'decoder_kernel': hid['lr_scale'],
                    'decoder_input_ffn': hid['lr_scale'],
                    'decoder_output_ffn': ffn_out['lr_scale'],
                },
            },
        }
        
        # Add GQA config if applicable
        if self.num_kv_groups < self.num_heads:
            config['model']['attention_module'] = 'multiquery_attention'
            config['model']['extra_attention_params'] = {
                'num_kv_groups': self.num_kv_groups
            }
        
        return config
    
    def print_summary(self):
        """Print a summary of all scaling factors."""
        print("=" * 80)
        print("Kyle's muP Configuration Summary")
        print("=" * 80)
        print()
        
        print("Model Configuration:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Base hidden size: {self.base_hidden_size}")
        print(f"  Filter size: {self.filter_size}")
        print(f"  Base filter size: {self.base_filter_size}")
        print(f"  Number of layers: {self.num_layers}")
        print(f"  Number of heads: {self.num_heads}")
        print(f"  Number of KV groups: {self.num_kv_groups}")
        
        if self.num_kv_groups < self.num_heads:
            if self.num_kv_groups == 1:
                attn_type = "Multi-Query Attention (MQA)"
            else:
                attn_type = "Grouped-Query Attention (GQA)"
        else:
            attn_type = "Multi-Head Attention (MHA)"
        print(f"  Attention type: {attn_type}")
        print()
        
        print("Scaling Factors:")
        print(f"  Width multiplier (m): {self.m:.4f}")
        print(f"  KV ratio (r): {self.r:.4f}")
        print(f"  Head dimension (d): {self.d}")
        print(f"  Depth (L): {self.L}")
        print()
        
        print("Embedding Layer:")
        emb = self.calculate_embedding_scales()
        for key, value in emb.items():
            print(f"  {key}: {value}")
        print()
        
        print("Hidden Layers (Q projection, FFN input):")
        hid = self.calculate_hidden_scales()
        for key, value in hid.items():
            print(f"  {key}: {value}")
        print()
        
        print("K/V Projections:")
        kv = self.calculate_kv_scales()
        for key, value in kv.items():
            print(f"  {key}: {value}")
        print()
        
        print("Output Layer (with depth scaling):")
        out = self.calculate_output_scales()
        for key, value in out.items():
            print(f"  {key}: {value}")
        print()
        
        print("FFN Output Layer (with depth scaling):")
        ffn_out = self.calculate_ffn_output_scales()
        for key, value in ffn_out.items():
            print(f"  {key}: {value}")
        print()
        
        print("Normalization Layers:")
        norm = self.calculate_norm_scales()
        for key, value in norm.items():
            print(f"  {key}: {value}")
        print()
        
        print("Additional Scales:")
        print(f"  Attention scale: {self.calculate_attention_scale()}")
        print(f"  Depth scale: {self.calculate_depth_scale()}")
        print()
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Kyle's muP scaling factors for GPT2"
    )
    
    parser.add_argument(
        '--hidden-size',
        type=int,
        required=True,
        help='Target model hidden dimension'
    )
    parser.add_argument(
        '--base-hidden-size',
        type=float,
        required=True,
        help='Base model hidden dimension'
    )
    parser.add_argument(
        '--filter-size',
        type=int,
        required=True,
        help='Target FFN intermediate dimension'
    )
    parser.add_argument(
        '--base-filter-size',
        type=float,
        required=True,
        help='Base FFN intermediate dimension'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        required=True,
        help='Number of transformer layers'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        required=True,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--num-kv-groups',
        type=int,
        default=None,
        help='Number of KV groups (defaults to num_heads for MHA)'
    )
    parser.add_argument(
        '--base-init-std',
        type=float,
        default=0.02,
        help='Base initialization standard deviation'
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=6e-4,
        help='Base learning rate'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output file for JSON configuration'
    )
    parser.add_argument(
        '--output-yaml',
        type=str,
        help='Output file for YAML configuration'
    )
    
    args = parser.parse_args()
    
    # Create calculator
    calc = MuPCalculator(
        hidden_size=args.hidden_size,
        base_hidden_size=args.base_hidden_size,
        filter_size=args.filter_size,
        base_filter_size=args.base_filter_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_groups=args.num_kv_groups,
        base_init_std=args.base_init_std,
    )
    
    # Print summary
    calc.print_summary()
    
    # Generate config
    config = calc.generate_yaml_config(base_lr=args.base_lr)
    
    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"JSON configuration saved to: {args.output_json}")
        print()
    
    # Save YAML if requested
    if args.output_yaml:
        try:
            import yaml
            with open(args.output_yaml, 'w') as f:
                yaml.dump(config, f, sort_keys=False, default_flow_style=False)
            print(f"YAML configuration saved to: {args.output_yaml}")
            print()
        except ImportError:
            print("Warning: PyYAML not installed, cannot save YAML file")
            print("Install with: pip install pyyaml")
            print()


if __name__ == '__main__':
    main()
