import hashlib
from typing import List
import numpy as np
from scipy.stats import norm
import torch
from transformers import LogitsWarper


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0, multiple_key: bool = False, num_keys: int = 1,):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.multiple_key = multiple_key
        self.num_keys = num_keys
        if self.multiple_key:
            hash_key_list = [0, 5823667, 68425619, 1107276647, 751783477, 563167303, 440817757, 368345293, 259336153, 131807699, 65535, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]
            green_list_mask_list = []
            for i in range(self.num_keys):
                rng = np.random.default_rng(self._hash_fn(hash_key_list[i]))
                mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
                rng.shuffle(mask)
                green_list_mask_list.append(torch.tensor(mask, dtype=torch.float32).clone())
            self.green_list_mask_list = green_list_mask_list


    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsWarper):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        if not self.multiple_key:
            watermark = self.strength * self.green_list_mask
            new_logits = scores + watermark.to(scores.device)
            return new_logits
        else:
            score_list = []
            for i in range(self.num_keys):
                watermark = self.strength * self.green_list_mask_list[i]
                new_logits = scores + watermark.to(scores.device)
                score_list.append(new_logits)
            return score_list


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        if not self.multiple_key:
            green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
            return self._z_score(green_tokens, len(sequence), self.fraction)
        else:
            z_scores = []
            for key in range(self.num_keys):
                green_tokens = int(sum(self.green_list_mask_list[key][i] for i in sequence))
                z_scores.append(self._z_score(green_tokens, len(sequence), self.fraction))
            return np.max(np.array(z_scores))

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        if not self.multiple_key:
            green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
            return self._z_score(green_tokens, len(sequence), self.fraction)
        else:
            z_scores = []
            for key in range(self.num_keys):
                green_tokens = int(sum(self.green_list_mask_list[key][i] for i in sequence))
                z_scores.append(self._z_score(green_tokens, len(sequence), self.fraction))
            return np.max(np.array(z_scores))
    
    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        return z_score > tau, z_score
