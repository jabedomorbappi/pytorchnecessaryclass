import torch
from collections import Counter
from typing import List

class PyTorchTokenizer:
    def __init__(self, num_words=None, oov_token="<OOV>", padding_token="<PAD>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.padding_token = padding_token
        self.word_index = {}
        self.index_word = {}
        self.oov_index = None
        self.pad_index = None

    def fit_on_texts(self, texts: List[str]):
        # Count all words
        word_counter = Counter(word for text in texts for word in text.split())
        most_common = word_counter.most_common(self.num_words)
        
        # Create word_index
        self.word_index = {
            word: i + 1 for i, (word, _) in enumerate(most_common)  # Start index from 1
        }
        
        # Handle special tokens
        if self.oov_token:
            self.oov_index = len(self.word_index) + 1
            self.word_index[self.oov_token] = self.oov_index
        if self.padding_token:
            self.pad_index = 0
            self.word_index[self.padding_token] = self.pad_index
        
        # Create reverse mapping
        self.index_word = {index: word for word, index in self.word_index.items()}

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            sequence = [
                self.word_index.get(word, self.oov_index) for word in text.split()
            ]
            sequences.append(sequence)
        return sequences

    def pad_sequences(self, sequences: List[List[int]], maxlen: int, padding: str = "post") -> torch.Tensor:
        padded_sequences = []
        for seq in sequences:
            if len(seq) > maxlen:
                padded_seq = seq[:maxlen]
            else:
                pad_length = maxlen - len(seq)
                if padding == "post":
                    padded_seq = seq + [self.pad_index] * pad_length
                elif padding == "pre":
                    padded_seq = [self.pad_index] * pad_length + seq
                else:
                    raise ValueError("Padding must be 'pre' or 'post'.")
            padded_sequences.append(padded_seq)
        return torch.tensor(padded_sequences, dtype=torch.long)