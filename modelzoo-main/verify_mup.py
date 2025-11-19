"""
muP Implementation Verification Script
======================================

This script verifies that Kyle's muP implementation is correctly applied to the GPT2 model.

It checks:
1. Initialization standard deviations for each layer type
2. Learning rate adjustments for each layer type
3. Output multipliers
4. Attention scaling

Expected muP Scaling Rules:
--------------------------
Layer Type           | Init Std            | LR Scale     | Output Mult
---------------------|---------------------|--------------|-------------
Embedding            | 1/m                 | 1/m          | m
Hidden (Q, FFN_in)   | 1/√m                | 1/m          | 1
K/V projections      | (1+√r)/(2√m)        | 1/m          | 2/(1+√r)
Unembedding          | 1/m                 | 1/m          | 1
Normalization        | -                   | 1/m          | -

Where:
- m = hidden_size / base_hidden_size
- r = num_heads / num_kv_groups
- d = hidden_size / num_heads
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
from tabulate import tabulate

# Add ModelZoo to path
modelzoo_path = Path(__file__).parent / "src"
sys.path.insert(0, str(modelzoo_path))

from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import Gpt2Model, GPT2LMHeadModelConfig
from mup_implementation import apply_mup_to_config, MuPConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MuPVerifier:
    """Verify muP implementation correctness."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        base_hidden_size: int = 256,
        num_heads: int = 12,
        num_kv_groups: int = 1,
        num_layers: int = 2,
        base_init_std: float = 0.02,
    ):
        """
        Initialize verifier.
        
        Args:
            hidden_size: Model hidden dimension
            base_hidden_size: Base hidden dimension for muP
            num_heads: Number of attention heads
            num_kv_groups: Number of KV groups
            num_layers: Number of transformer layers
            base_init_std: Base initialization std
        """
        self.hidden_size = hidden_size
        self.base_hidden_size = base_hidden_size
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.num_layers = num_layers
        self.base_init_std = base_init_std
        
        # Calculate scaling factors
        self.m = hidden_size / base_hidden_size
        self.r = num_heads / num_kv_groups if num_kv_groups > 0 else 1.0
        self.d = hidden_size // num_heads
        self.L = num_layers
        
        logger.info("=" * 80)
        logger.info("muP Verification Configuration")
        logger.info("=" * 80)
        logger.info(f"Hidden size: {hidden_size}")
        logger.info(f"Base hidden size: {base_hidden_size}")
        logger.info(f"Num heads: {num_heads}")
        logger.info(f"Num KV groups: {num_kv_groups}")
        logger.info(f"Num layers: {num_layers}")
        logger.info(f"")
        logger.info(f"Calculated muP parameters:")
        logger.info(f"  m (width multiplier): {self.m:.4f}")
        logger.info(f"  r (KV ratio): {self.r:.4f}")
        logger.info(f"  d (head dim): {self.d}")
        logger.info(f"  L (num layers): {self.L}")
        logger.info("=" * 80)
        
    def create_model(self) -> Tuple[Gpt2Model, GPT2LMHeadModelConfig]:
        """Create a muP-scaled GPT2 model."""
        # Create base config
        config = GPT2LMHeadModelConfig(
            name="GPT2LMHeadModel",
            vocab_size=50257,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_heads=self.num_heads,
            filter_size=self.hidden_size * 4,
            max_position_embeddings=1024,
            attention_type="scaled_dot_product",
            extra_attention_params={"num_kv_groups": self.num_kv_groups},
            embedding_dropout_rate=0.1,
            share_embedding_weights=False,  # Don't share to verify separately
            position_embedding_type="learned",
            use_ffn_bias=True,
            nonlinearity="gelu",
            layer_norm_epsilon=1e-5,
            use_bias_in_output=False,
            dropout_rate=0.1,
        )
        
        # Apply muP
        config = apply_mup_to_config(
            config=config,
            base_hidden_size=self.base_hidden_size,
            base_filter_size=self.base_hidden_size * 4,
            base_init_std=self.base_init_std,
        )
        
        # Create model
        model = Gpt2Model(config)
        
        return model, config
    
    def calculate_expected_init_std(self, layer_type: str) -> float:
        """
        Calculate expected initialization std for a layer type.
        
        Args:
            layer_type: One of ['embedding', 'hidden', 'kv', 'unembedding', 
                                'ffn_output', 'attention_output']
        
        Returns:
            Expected initialization standard deviation
        """
        if layer_type == 'embedding':
            return self.base_init_std * (1.0 / self.m)
        
        elif layer_type == 'hidden':  # Q, FFN input
            return self.base_init_std * (1.0 / (self.m ** 0.5))
        
        elif layer_type == 'kv':  # K, V projections
            return self.base_init_std * ((1 + self.r ** 0.5) / (2 * self.m ** 0.5))
        
        elif layer_type == 'unembedding':
            return self.base_init_std * (1.0 / self.m)
        
        elif layer_type == 'ffn_output':  # FFN output with depth scaling
            return self.base_init_std * (1.0 / (self.m ** 0.5)) * (1.0 / self.L)
        
        elif layer_type == 'attention_output':  # Attention output with depth scaling
            return self.base_init_std * (1.0 / (self.m ** 0.5)) * (1.0 / self.L)
        
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def calculate_expected_lr_scale(self, layer_type: str) -> float:
        """
        Calculate expected learning rate scale for a layer type.
        
        Args:
            layer_type: Layer type identifier
        
        Returns:
            Expected LR scale factor
        """
        if layer_type in ['embedding', 'hidden', 'kv', 'unembedding', 'normalization']:
            return 1.0 / self.m
        else:
            return 1.0  # Default
    
    def measure_layer_init_std(self, param: torch.Tensor) -> float:
        """Measure actual initialization std of a parameter."""
        return param.std().item()
    
    def verify_initialization(self, model: Gpt2Model) -> List[Dict]:
        """
        Verify initialization of all layers.
        
        Returns:
            List of verification results
        """
        logger.info("\n" + "=" * 80)
        logger.info("INITIALIZATION VERIFICATION")
        logger.info("=" * 80)
        
        results = []
        
        # Get all named parameters
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue  # Skip biases
            
            # Measure actual std
            actual_std = self.measure_layer_init_std(param)
            
            # Determine layer type and expected std
            expected_std = None
            layer_type = None
            
            if 'embedding_layer' in name and 'position' not in name:
                layer_type = 'embedding'
                expected_std = self.calculate_expected_init_std('embedding')
            
            elif 'position_embedding' in name:
                layer_type = 'position_embedding'
                expected_std = self.calculate_expected_init_std('embedding')
            
            elif 'proj_q_dense_layer' in name or 'q_dense' in name:
                layer_type = 'Q_projection'
                expected_std = self.calculate_expected_init_std('hidden')
            
            elif 'proj_k_dense_layer' in name or 'k_dense' in name:
                layer_type = 'K_projection'
                expected_std = self.calculate_expected_init_std('kv')
            
            elif 'proj_v_dense_layer' in name or 'v_dense' in name:
                layer_type = 'V_projection'
                expected_std = self.calculate_expected_init_std('kv')
            
            elif 'proj_output_dense_layer' in name or ('attn' in name and 'dense' in name):
                layer_type = 'attention_output'
                expected_std = self.calculate_expected_init_std('attention_output')
            
            elif 'ffn' in name and '.0.' in name:  # FFN input layer
                layer_type = 'FFN_input'
                expected_std = self.calculate_expected_init_std('hidden')
            
            elif 'ffn' in name and '.1.' in name:  # FFN output layer
                layer_type = 'FFN_output'
                expected_std = self.calculate_expected_init_std('ffn_output')
            
            elif 'lm_head' in name or 'output' in name.lower():
                layer_type = 'unembedding'
                expected_std = self.calculate_expected_init_std('unembedding')
            
            else:
                layer_type = 'unknown'
                expected_std = None
            
            if expected_std is not None:
                # Calculate error
                rel_error = abs(actual_std - expected_std) / expected_std
                passed = rel_error < 0.15  # 15% tolerance
                
                results.append({
                    'layer_name': name,
                    'layer_type': layer_type,
                    'expected_std': expected_std,
                    'actual_std': actual_std,
                    'rel_error': rel_error,
                    'passed': passed,
                })
        
        return results
    
    def verify_lr_adjustments(self, config: GPT2LMHeadModelConfig) -> List[Dict]:
        """
        Verify learning rate adjustment groups.
        
        Returns:
            List of verification results
        """
        logger.info("\n" + "=" * 80)
        logger.info("LEARNING RATE ADJUSTMENT VERIFICATION")
        logger.info("=" * 80)
        
        results = []
        
        if not hasattr(config, 'lr_adjustment_groups'):
            logger.warning("No lr_adjustment_groups found in config!")
            return results
        
        lr_groups = config.lr_adjustment_groups
        
        # Expected LR scales for each group
        expected_scales = {
            'embedding': 1.0 / self.m,
            'q_decoder_attention': 1.0 / self.m,
            'k_decoder_attention': 1.0 / self.m,
            'v_decoder_attention': 1.0 / self.m,
            'decoder_kernel': 1.0 / self.m,
            'decoder_input_ffn': 1.0 / self.m,
            'decoder_output_ffn': 1.0 / self.m,
            'normalization': 1.0 / self.m,
        }
        
        for group_name, group_config in lr_groups.items():
            expected_scale = expected_scales.get(group_name, 1.0)
            
            # Extract actual scale from LRAdjustmentGroup
            if hasattr(group_config, 'scale'):
                actual_scale = group_config.scale
            else:
                actual_scale = None
            
            if actual_scale is not None:
                rel_error = abs(actual_scale - expected_scale) / expected_scale
                passed = rel_error < 0.01  # 1% tolerance for LR scales
                
                results.append({
                    'group_name': group_name,
                    'expected_scale': expected_scale,
                    'actual_scale': actual_scale,
                    'rel_error': rel_error,
                    'passed': passed,
                })
        
        return results
    
    def verify_output_multipliers(self, config: GPT2LMHeadModelConfig) -> List[Dict]:
        """
        Verify output multipliers (embeddings_scale, attention_logits_alpha, etc.).
        
        Returns:
            List of verification results
        """
        logger.info("\n" + "=" * 80)
        logger.info("OUTPUT MULTIPLIER VERIFICATION")
        logger.info("=" * 80)
        
        results = []
        
        # Embeddings scale (should be m)
        if hasattr(config, 'embeddings_scale'):
            expected = self.m
            actual = config.embeddings_scale
            rel_error = abs(actual - expected) / expected
            passed = rel_error < 0.01
            
            results.append({
                'parameter': 'embeddings_scale',
                'expected': expected,
                'actual': actual,
                'rel_error': rel_error,
                'passed': passed,
            })
        
        # Attention logits alpha (should be 1/d * 1/d = 1/d^2 when scale_qk_dot_by_d=True)
        if hasattr(config, 'attention_logits_alpha'):
            expected = (1.0 / self.d) / self.d  # Because scale_qk_dot_by_d divides by d
            actual = config.attention_logits_alpha
            rel_error = abs(actual - expected) / expected if expected != 0 else 0
            passed = rel_error < 0.01
            
            results.append({
                'parameter': 'attention_logits_alpha',
                'expected': expected,
                'actual': actual,
                'rel_error': rel_error,
                'passed': passed,
            })
        
        # Output logits alpha (should be 1.0 for unembedding)
        if hasattr(config, 'output_logits_alpha'):
            expected = 1.0
            actual = config.output_logits_alpha
            rel_error = abs(actual - expected) / expected
            passed = rel_error < 0.01
            
            results.append({
                'parameter': 'output_logits_alpha',
                'expected': expected,
                'actual': actual,
                'rel_error': rel_error,
                'passed': passed,
            })
        
        # Depth scaling factor (should be 1/L)
        if hasattr(config, 'depth_scaling_factor'):
            expected = 1.0 / self.L
            actual = config.depth_scaling_factor
            rel_error = abs(actual - expected) / expected
            passed = rel_error < 0.01
            
            results.append({
                'parameter': 'depth_scaling_factor',
                'expected': expected,
                'actual': actual,
                'rel_error': rel_error,
                'passed': passed,
            })
        
        return results
    
    def print_results(
        self,
        init_results: List[Dict],
        lr_results: List[Dict],
        mult_results: List[Dict],
    ):
        """Print verification results in a nice table format."""
        
        # Initialization results
        print("\n" + "=" * 100)
        print("INITIALIZATION VERIFICATION RESULTS")
        print("=" * 100)
        
        init_table = []
        for r in init_results:
            init_table.append([
                r['layer_type'],
                f"{r['expected_std']:.6f}",
                f"{r['actual_std']:.6f}",
                f"{r['rel_error']*100:.2f}%",
                "✓ PASS" if r['passed'] else "✗ FAIL",
            ])
        
        print(tabulate(
            init_table,
            headers=['Layer Type', 'Expected Std', 'Actual Std', 'Rel Error', 'Status'],
            tablefmt='grid',
        ))
        
        # Summary
        total_init = len(init_results)
        passed_init = sum(1 for r in init_results if r['passed'])
        print(f"\nInitialization Summary: {passed_init}/{total_init} checks passed")
        
        # Learning rate results
        print("\n" + "=" * 100)
        print("LEARNING RATE ADJUSTMENT VERIFICATION RESULTS")
        print("=" * 100)
        
        lr_table = []
        for r in lr_results:
            lr_table.append([
                r['group_name'],
                f"{r['expected_scale']:.6f}",
                f"{r['actual_scale']:.6f}",
                f"{r['rel_error']*100:.2f}%",
                "✓ PASS" if r['passed'] else "✗ FAIL",
            ])
        
        print(tabulate(
            lr_table,
            headers=['Group Name', 'Expected Scale', 'Actual Scale', 'Rel Error', 'Status'],
            tablefmt='grid',
        ))
        
        # Summary
        total_lr = len(lr_results)
        passed_lr = sum(1 for r in lr_results if r['passed'])
        print(f"\nLR Adjustment Summary: {passed_lr}/{total_lr} checks passed")
        
        # Output multiplier results
        print("\n" + "=" * 100)
        print("OUTPUT MULTIPLIER VERIFICATION RESULTS")
        print("=" * 100)
        
        mult_table = []
        for r in mult_results:
            mult_table.append([
                r['parameter'],
                f"{r['expected']:.6f}",
                f"{r['actual']:.6f}",
                f"{r['rel_error']*100:.2f}%",
                "✓ PASS" if r['passed'] else "✗ FAIL",
            ])
        
        print(tabulate(
            mult_table,
            headers=['Parameter', 'Expected', 'Actual', 'Rel Error', 'Status'],
            tablefmt='grid',
        ))
        
        # Summary
        total_mult = len(mult_results)
        passed_mult = sum(1 for r in mult_results if r['passed'])
        print(f"\nOutput Multiplier Summary: {passed_mult}/{total_mult} checks passed")
        
        # Overall summary
        print("\n" + "=" * 100)
        print("OVERALL VERIFICATION SUMMARY")
        print("=" * 100)
        total_checks = total_init + total_lr + total_mult
        total_passed = passed_init + passed_lr + passed_mult
        
        print(f"Total checks: {total_checks}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_checks - total_passed}")
        print(f"Success rate: {total_passed/total_checks*100:.1f}%")
        
        if total_passed == total_checks:
            print("\n🎉 ALL CHECKS PASSED! muP implementation is correct!")
        else:
            print(f"\n⚠️  {total_checks - total_passed} checks failed. Please review the implementation.")
        
        print("=" * 100)
    
    def run_verification(self):
        """Run all verification checks."""
        logger.info("Starting muP verification...")
        
        # Create model
        model, config = self.create_model()
        
        # Run verification checks
        init_results = self.verify_initialization(model)
        lr_results = self.verify_lr_adjustments(config)
        mult_results = self.verify_output_multipliers(config)
        
        # Print results
        self.print_results(init_results, lr_results, mult_results)
        
        return init_results, lr_results, mult_results


def verify_mup_for_different_configs():
    """Test muP with different configurations."""
    
    configs = [
        {
            'name': 'MHA (num_kv_groups=1)',
            'hidden_size': 768,
            'num_heads': 12,
            'num_kv_groups': 1,
        },
        {
            'name': 'GQA (num_kv_groups=4)',
            'hidden_size': 768,
            'num_heads': 12,
            'num_kv_groups': 4,
        },
        {
            'name': 'MQA (num_kv_groups=12)',
            'hidden_size': 768,
            'num_heads': 12,
            'num_kv_groups': 12,
        },
    ]
    
    for config in configs:
        print("\n\n" + "#" * 100)
        print(f"# TESTING: {config['name']}")
        print("#" * 100)
        
        verifier = MuPVerifier(
            hidden_size=config['hidden_size'],
            base_hidden_size=256,
            num_heads=config['num_heads'],
            num_kv_groups=config['num_kv_groups'],
            num_layers=2,
            base_init_std=0.02,
        )
        
        verifier.run_verification()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify muP implementation")
    
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size'
    )
    
    parser.add_argument(
        '--base_hidden_size',
        type=int,
        default=256,
        help='Base hidden size'
    )
    
    parser.add_argument(
        '--num_heads',
        type=int,
        default=12,
        help='Number of attention heads'
    )
    
    parser.add_argument(
        '--num_kv_groups',
        type=int,
        default=1,
        help='Number of KV groups'
    )
    
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of layers'
    )
    
    parser.add_argument(
        '--test_all',
        action='store_true',
        help='Test all configurations (MHA, GQA, MQA)'
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        verify_mup_for_different_configs()
    else:
        verifier = MuPVerifier(
            hidden_size=args.hidden_size,
            base_hidden_size=args.base_hidden_size,
            num_heads=args.num_heads,
            num_kv_groups=args.num_kv_groups,
            num_layers=args.num_layers,
        )
        
        verifier.run_verification()
