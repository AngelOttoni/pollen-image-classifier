#!/usr/bin/env python3
'''
CLI para pipeline BoVW+SIFT.

Uso:
    python src/cli.py --config configs/baseline_k100.yaml --mode run-all
'''
import argparse
import yaml
from .run_bovw import run_all, build_vocab, train_clf, evaluate_run

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='BoVW + SIFT pipeline CLI')
    parser.add_argument('--config', '-c', required=True, help='Path to config yaml')
    parser.add_argument('--mode', '-m', choices=['build_vocab', 'train', 'evaluate', 'run-all'], default='run-all',
                        help='Mode to run')
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == 'build_vocab':
        build_vocab(cfg)
    elif args.mode == 'train':
        train_clf(cfg)
    elif args.mode == 'evaluate':
        evaluate_run(cfg)
    else:
        run_all(cfg)

if __name__ == '__main__':
    main()
