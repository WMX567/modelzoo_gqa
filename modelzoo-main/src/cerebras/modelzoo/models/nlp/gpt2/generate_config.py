#!/usr/bin/env python3
"""
Generate parameterized config files for learning rate sweeps.
"""

import yaml
import sys
import argparse


def generate_config(base_config_path, learning_rate, output_path, num_kv_groups=None):
    """Generate a config file with specified learning rate and num_kv_groups."""

    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Calculate end learning rate (lr / 10)
    end_lr = learning_rate / 10.0

    # Update learning rates in the scheduler
    schedulers = config['trainer']['init']['schedulers'][0]['SequentialLR']['schedulers']
    schedulers[0]['LinearLR']['end_learning_rate'] = learning_rate
    schedulers[1]['LinearLR']['initial_learning_rate'] = learning_rate
    schedulers[1]['LinearLR']['end_learning_rate'] = end_lr

    # Update num_kv_groups if provided
    if num_kv_groups is not None:
        try:
            config['trainer']['init']['model']['extra_attention_params']['num_kv_groups'] = int(num_kv_groups)
        except Exception as e:
            print(f"Warning: Could not set num_kv_groups: {e}")

    # Save parameterized config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Generated config with LR={learning_rate}, end_LR={end_lr}, num_kv_groups={num_kv_groups} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate parameterized config files')
    parser.add_argument('base_config', help='Path to base config file')
    parser.add_argument('learning_rate', type=float, help='Learning rate value')
    parser.add_argument('output_config', help='Path for output config file')
    parser.add_argument('num_kv_groups', nargs='?', default=None, help='num_kv_groups value (optional)')

    args = parser.parse_args()
    generate_config(args.base_config, args.learning_rate, args.output_config, args.num_kv_groups)
