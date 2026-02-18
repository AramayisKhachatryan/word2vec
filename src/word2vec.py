import numpy as np
from scipy.stats import spearmanr
import pickle
from tqdm import tqdm
import time

from vocab import Vocab
from skipgram import SkipGram
from cbow import CBOW
from utils import cosine_similarity

from typing import List, Tuple, Iterator

import wandb

class Word2Vec:

    def __init__(self, 
                 vocab: Vocab,
                 embedding_dim: int = 100,
                 model_type: str = 'skipgram',
                 learning_rate: float = 0.01,
                 window_size: int = 5,
                 num_negative_samples: int = 5,
                 subsample_threshold: float = 1e-5,
                 epochs: int = 5,
                 simlex_path: str = '',
                 checkpoint_dir: str = '') -> None:
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.model_type = model_type.lower()
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.epochs = epochs
        self.simlex_path = simlex_path
        self.checkpoint_dir = checkpoint_dir

        self.initial_learning_rate = learning_rate

        if model_type == 'skipgram':
            self.model = SkipGram(len(vocab), embedding_dim)
        elif model_type == 'cbow':
            self.model = CBOW(len(vocab), embedding_dim)

    def _generate_skipgram_pairs(self, corpus_indices: np.ndarray) -> Iterator[Tuple[int, int]]:
        corpus_len = len(corpus_indices)

        for i in range(corpus_len):
            target_idx = corpus_indices[i]

            # During our training we use small window size so we don't chose random window size between [1, self.window_size] during training
            start = max(0, i - self.window_size)
            end = min(corpus_len, i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_idx = corpus_indices[j]
                    yield target_idx, context_idx

    def _generate_cbow_pairs(self, corpus_indices: np.ndarray) -> Iterator[Tuple[List[int], int]]:
        corpus_len = len(corpus_indices)

        for i in range(corpus_len):
            target_idx = corpus_indices[i]

            # During our training we use small window size so we don't chose random window size between [1, self.window_size] during training
            start = max(0, i - self.window_size)
            end = min(corpus_len, i + self.window_size + 1)

            context_indices = []
            for j in range(start, end):
                if i != j:
                    context_indices.append(corpus_indices[j])

            if len(context_indices) > 0:
                yield context_indices, target_idx

    def train(self, corpus: str, wandb_enabled: bool=False, log_interval: int=10, eval_log_interval: int=50000) -> None:
        corpus_indices = []
        for w in corpus.split():
            if w not in self.vocab.word2idx.keys():
                continue
            if np.random.rand() > self.vocab.get_subsampling_prob(w, self.subsample_threshold):
                continue
            corpus_indices.append(self.vocab.word_to_index(w))
        
        if self.model_type == 'skipgram':
            self._train_skipgram(np.array(corpus_indices), wandb_enabled, log_interval, eval_log_interval)

        elif self.model_type == 'cbow':
            self._train_cbow(corpus_indices, wandb_enabled, log_interval, eval_log_interval)

    def _train_skipgram(self, corpus_indices: np.ndarray, wandb_enabled: bool=False, log_interval: int=10, eval_log_interval: int=50000) -> None:
        pairs_per_epoch = len(corpus_indices) * 2 * self.window_size # approximate number
        total_samples = self.epochs * pairs_per_epoch
        processed_samples = 0
        checkpoint_count = 1

        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_loss = 0
            num_samples = 0

            train_data = self._generate_skipgram_pairs(corpus_indices)
            pbar = tqdm(train_data,
                        total=pairs_per_epoch,
                        desc=f'Epoch {epoch + 1}/{self.epochs}',
                        unit='pairs')

            for pair in pbar:
                target_idx, context_idx = pair
                negative_indices = self.vocab.get_negative_samples(target_idx, self.num_negative_samples)
                loss = self.model.backward(target_idx, context_idx, negative_indices, self.learning_rate)

                epoch_loss += loss
                num_samples += 1
                processed_samples += 1

                if wandb_enabled:
                    if processed_samples % log_interval == 0:
                        wandb.log({
                            'loss': loss,
                            'learning_rate': self.learning_rate,
                        })

                    if processed_samples % eval_log_interval == 0:
                        wandb.log({
                            'simlex999_correlation': self._eval_simlex()
                        })

                if processed_samples % eval_log_interval == 0:
                    self.save_model(self.checkpoint_dir + f'skipgram-{checkpoint_count}')
                    checkpoint_count += 1


                self._update_learning_rate(processed_samples, total_samples)

            pbar.close()

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / num_samples

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

            if wandb_enabled:
                wandb.log({
                    'epoch': epoch + 1,
                    'epoch_loss': avg_loss,
                    'epoch_time': epoch_time
                })



    def _train_cbow(self, corpus_indices: List[int], wandb_enabled: bool=False, log_interval: int=10, eval_log_interval: int=500000) -> None:
        pairs_per_epoch = len(corpus_indices) # approximate number
        total_samples = self.epochs * pairs_per_epoch
        processed_samples = 0
        checkpoint_count = 0

        for epoch in range(self.epochs):
            start_time = time.time()
            epoch_loss = 0
            num_samples = 1

            train_data = self._generate_cbow_pairs(np.array(corpus_indices))

            pbar = tqdm(train_data,
                        total=pairs_per_epoch,
                        desc=f"Epoch {epoch + 1}/{self.epochs}",
                        unit="pairs")

            for pair in pbar:
                context_indices, target_idx = pair

                negative_indices = self.vocab.get_negative_samples(target_idx, self.num_negative_samples)

                loss = self.model.backward(np.array(context_indices), target_idx, negative_indices, self.learning_rate)

                epoch_loss += loss
                num_samples += 1
                processed_samples += 1

                if wandb_enabled:
                    if processed_samples % log_interval == 0:
                        wandb.log({
                            'loss': loss,
                            'learning_rate': self.learning_rate,
                        })

                    if processed_samples % eval_log_interval == 0:
                        wandb.log({
                            'simlex999_correlation': self._eval_simlex()
                        })

                if processed_samples % eval_log_interval == 0:
                    self.save_model(self.checkpoint_dir + f'cbow-{checkpoint_count}')
                    checkpoint_count += 1

                self._update_learning_rate(processed_samples, total_samples)

            pbar.close()

            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / num_samples

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

            if wandb_enabled:
                wandb.log({
                    'epoch': epoch + 1,
                    'epoch_loss': avg_loss,
                    'epoch_time': epoch_time
                })

    def _update_learning_rate(self, processed_samples: int, total_samples: int) -> None:
        """
        Linearly update learning rate

        Args:
            processed_samples: Number of processed samples
            total_samples: Number of total samples
        """
        self.learning_rate = max(1e-5, self.initial_learning_rate * (1 - (processed_samples / total_samples)))

    def _eval_simlex(self) -> float:
        human_scores = []
        model_scores = []
        total = 0

        with open(self.simlex_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                w1, w2, score = parts[0], parts[1], float(parts[3])
                total += 1

                if w1 in self.vocab.word2idx.keys() and w2 in self.vocab.word2idx.keys():
                    v1 = self.get_embedding(w1)
                    v2 = self.get_embedding(w2)

                    sim = cosine_similarity(v1, v2)

                    human_scores.append(score)
                    model_scores.append(sim)

        correlation, _ = spearmanr(human_scores, model_scores)

        return correlation

    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding vector of given word.

        Args:
            word: String of word to get embedding of

        Returns:
            Vector representation of the given word.
        """
        if word in self.vocab.word2idx.keys():
            return self.model.W_in[self.vocab.word2idx[word]]
        
        return self.model.W_in[self.vocab.word2idx['<UNK>']]
    
    def most_similar(self, word: str, top_n: int=10) -> List[Tuple[str, int]]:
        """  
        Find most similar words using cosine similarity.

        Args:
            word: Query word
            top_n: Number of most similar words to return

        Returns:
            List of (word, similarity) tuples
        """

        if word not in self.vocab.word2idx.keys():
            print(f'{word} not in vocabulary')
            return []

        word_idx = self.vocab.word_to_index(word)
        word_vec = self.model.W_in[word_idx]

        all_vecs = self.model.W_in
        
        word_vec_norm = word_vec / np.linalg.norm(word_vec)
        all_vecs_norm = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
        similarities = np.dot(all_vecs_norm, word_vec_norm)
        
        similar_indices = np.argsort(similarities)[::-1]

        results = []
        for i in similar_indices:
            if i == word_idx:
                continue
            similar_word = self.vocab.idx_to_word(i)
            if similar_word == '<UNK>':
                continue
            results.append((similar_word, similarities[i]))
            if len(results) == top_n:
                break
        
        return results

    def save_model(self, filepath: str) -> None:
        """
        Save the entire model.
        
        Args:
            filepath: Path to save the model
        """

        model_data = {
            'vocab': self.vocab,
            'embedding_dim': self.embedding_dim,
            'model_type': self.model_type,
            'learning_rate': self.learning_rate,
            'window_size': self.window_size,
            'num_negative_samples': self.num_negative_samples,
            'subsample_threshold': self.subsample_threshold,
            'epochs': self.epochs,
            'simlex_path': self.simlex_path,
            'checkpoint_dir': self.checkpoint_dir,
            'W_in': self.model.W_in,
            'W_out': self.model.W_out,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Word2Vec instance with loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        w2v = cls(
            vocab=model_data['vocab'],
            embedding_dim=model_data['embedding_dim'],
            model_type=model_data['model_type'],
            learning_rate=model_data['learning_rate'],
            window_size=model_data['window_size'],
            num_negative_samples=model_data['num_negative_samples'],
            subsample_threshold=model_data['subsample_threshold'],
            simlex_path=model_data['simlex_path'],
            checkpoint_dir=model_data['checkpoint_dir'],
            epochs=model_data['epochs']
        )
        w2v.model.W_in = model_data['W_in']
        w2v.model.W_out = model_data['W_out']

        return w2v