import numpy as np
from utils import sigmoid

class CBOW:

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        Initialize cbow model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        low = (-1) / np.sqrt(embedding_dim)
        high = 1 / np.sqrt(embedding_dim)
        self.W_in = np.random.uniform(low, high, (vocab_size, embedding_dim))
        self.W_out = np.random.uniform(low, high, (vocab_size, embedding_dim))

    def backward(self,
                 context_indices: np.ndarray,
                 target_idx: int,
                 negative_indices: np.ndarray,
                 learning_rate: float) -> float:
        """
        Backpropagation and weight updates using negative sampling.
        Also calculates loss and returns it.

        Args:
            context_indices: Array of context word indices
            target_idx: Index of target word
            negative_indices: Array of negative sample indices
            learning_rate: Learning rate for gradient descent

        Returns:
            Loss of the pass
        """

        h = np.mean(self.W_in[context_indices], axis=0)
        u_pos = self.W_out[target_idx]
        u_neg = self.W_out[negative_indices]

        pos_score = sigmoid(np.dot(u_pos, h))
        neg_score = sigmoid(np.dot(u_neg, h))

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1 - neg_score + 1e-10))

        grad_pos = pos_score - 1
        grad_neg = neg_score

        self.W_out[target_idx] -= learning_rate * np.clip(grad_pos * h, -5, 5)
        self.W_out[negative_indices] -= learning_rate * np.clip(grad_neg[:, None] * h, -5, 5)

        grad_h = grad_pos * u_pos + np.dot(grad_neg, u_neg)

        grad_context = np.clip(grad_h, -5, 5) / len(context_indices)

        for i in context_indices:
            self.W_in[i] -= learning_rate * grad_context

        return loss