import numpy as np

class Vocab:

    def __init__(self, corpus: str, min_count: int) -> None:
        """
        Initialize Vocab class.

        Args:
            corpus: String of all the sentences to build vocabulary on
            min_count: Minimum frequency threshold for including word in vocabulary
        """
        self.corpus = corpus
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = {}
        self.total_words = 0
        self.negative_sampling_table = None

        self.build_vocab(corpus)

    def build_vocab(self, corpus: str) -> None:
        """
        Build vocabulary from corpus by counting words and filtering by min_count.

        Args:
            corpus: String of all the sentences to build vocabulary on
        """
        word_counts = {}
        for w in corpus.split():
            word_counts[w] = word_counts.get(w, 0) + 1
        
        self.total_words = sum(word_counts.values())

        filtered_words = {}
        for w in word_counts.keys():
            if word_counts[w] >= self.min_count:
                filtered_words[w] = word_counts[w]

        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        self.word2idx['<UNK>'] = 0
        self.idx2word[0] = '<UNK>'
        self.word_freq['<UNK>'] = 0

        for i, (w, c) in enumerate(sorted_words, start=1):
            self.word2idx[w] = i
            self.idx2word[i] = w
            self.word_freq[w] = c


        self._init_negative_sampling_table()
    
    def _init_negative_sampling_table(self) -> None:
        """
        Initialize sampling table for negative sampling.
        Uses word frequency raised to 0.75 paper as described in word2vec paper.
        """
        vocab_size = len(self.word2idx)

        pow_freq = np.array([self.word_freq.get(self.idx2word[i], 0) ** 0.75 for i in range(vocab_size)])
        
        self.negative_sampling_table = pow_freq / np.sum(pow_freq)
    
    def word_to_index(self, word: str) -> int:
        """Get index of given word in vocabulary"""
        return self.word2idx.get(word, 0)

    def idx_to_word(self, idx: int) -> str:
        """Get string representation of word given its index in vocabulary"""
        return self.idx2word.get(idx, '<UNK>')
    
    def get_negative_samples(self, target_idx: int, num_samples: int) -> np.ndarray:
        """  
        Sample negative examples for a target word.
        
        Args:
            target_idx: Index of the target word to exclude from samples
            num_samples: Number of negative samples to draw
        
        Returns:
            Array of negative sample indices
        """
        while True:
            samples = np.random.choice(
                len(self.word2idx),
                size=num_samples * 2,
                p=self.negative_sampling_table
            )
            mask = (samples != target_idx) & (samples != 0)
            valid = samples[mask]
            if len(valid) >= num_samples:
                return valid[:num_samples]

    def get_subsampling_prob(self, word: str, threshold: float=1e-5) -> float:
        """
        Calculate probability of keeping a word during subsampling.
        Frequent words are randomly downsamples as described in word2vec paper
        
        Args:
            word: Word to calculate subsampling probability for
            threshold: Subsampling threshold
        
        Returns:
            Probability of keeping this word.
        """
        if word not in self.word_freq:
            return 1.0
        
        freq = self.word_freq[word] / self.total_words

        # prob = np.sqrt(threshold / freq)
        prob = (np.sqrt(freq / threshold) + 1) * (threshold / freq)

        return min(prob, 1.0)

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.word2idx)
    
    def __getitem__(self, word: str) -> int:
        """Allow dictionary-like access: vocab[word] returns index."""
        return self.word_to_index(word)
        