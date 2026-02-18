import os
import yaml
import argparse
import numpy as np

from vocab import Vocab
from word2vec import Word2Vec

import wandb

np.random.seed(21)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def init_wandb(config):
    """Initialize Weights & Biases"""
    if not config['wandb']['enabled']:
        return None
    
    run = wandb.init(
        project=config['wandb']['project'],
        config={
            'model_type': config['model']['type'],
            'embedding_dim': config['model']['embedding_dim'],
            'epochs': config['training']['epochs'],
            'learning_rate': config['training']['learning_rate'],
            'window_size': config['training']['window_size'],
            'num_negative_samples': config['training']['num_negative_samples'],
            'min_count': config['vocabulary']['min_count'],
            'subsample_threshold': config['vocabulary']['subsample_threshold']
        }
    )

    return run

def parse_args():
    """Parse arguments from the terminal."""
    parser = argparse.ArgumentParser(description='Train Word2Vec model')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')

    return parser.parse_args()

def main():
    args = parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config_path = os.path.join(current_dir, '..', 'configs', 'cbow_config.yaml')
        config = load_config(config_path)

    if args.no_wandb:
        config['wandb']['enabled'] = False

    corpus_path = os.path.join(current_dir, '..', config['data']['corpus_path'])
    with open(corpus_path, 'r') as file:
        corpus = file.read()
        corpus = corpus[:int((config['data']['p'] * len(corpus)))]

    vocabulary = Vocab(corpus=corpus, min_count=config['vocabulary']['min_count'])

    model = Word2Vec(
        vocab=vocabulary,
        embedding_dim=config['model']['embedding_dim'],
        model_type=config['model']['type'],
        learning_rate=config['training']['learning_rate'],
        window_size=config['training']['window_size'],
        num_negative_samples=config['training']['num_negative_samples'],
        subsample_threshold=config['vocabulary']['subsample_threshold'],
        epochs=config['training']['epochs'],
        simlex_path=str(os.path.join(current_dir, '..', config['evaluating']['simlex_path'])),
        checkpoint_dir=str(os.path.join(current_dir, '..', config['data']['checkpoint_dir'])),
    )

    init_wandb(config)
    model.train(corpus, config['wandb']['enabled'], config['wandb']['log_interval'], config['wandb']['eval_log_interval'])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'models', f'{config['model']['save_name']}.pkl')
    model.save_model(output_path)

if __name__ == '__main__':
    main()