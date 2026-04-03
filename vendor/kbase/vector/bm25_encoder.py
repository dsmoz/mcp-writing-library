"""
BM25 Encoder for Sparse Vector Generation

This module implements BM25 (Best Matching 25) algorithm for generating sparse vectors
for keyword-based search. Used in hybrid search alongside dense semantic vectors.

BM25 Formula:
    score(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))

Where:
    - IDF(qi): Inverse Document Frequency of term qi
    - f(qi,D): Frequency of term qi in document D
    - |D|: Length of document D
    - avgdl: Average document length in the collection
    - k1: Term frequency saturation parameter (default: 1.5)
    - b: Length normalization parameter (default: 0.75)

Usage:
    from kbase.vector import BM25Encoder, get_bm25_encoder

    # Option 1: Load pre-trained encoder
    encoder = get_bm25_encoder("/path/to/encoder.pkl")

    # Option 2: Train new encoder
    encoder = BM25Encoder()
    encoder.fit(documents)
    encoder.save("/path/to/encoder.pkl")

    # Encode text to sparse vector
    sparse_vector = encoder.encode("search query")
"""

import math
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qdrant_client import models


@dataclass
class BM25Stats:
    """Statistics about the BM25 encoder."""

    corpus_size: int
    vocabulary_size: int
    average_document_length: float
    k1: float
    b: float
    top_idf_terms: List[Tuple[str, float]]


class BM25Encoder:
    """
    BM25 encoder for generating sparse vectors from text.

    This encoder calculates BM25 scores for terms and converts them to
    Qdrant SparseVector format for hybrid search.

    Attributes:
        k1 (float): Term frequency saturation parameter (default: 1.5)
        b (float): Length normalization parameter (default: 0.75)
        corpus_size (int): Number of documents in the corpus
        avgdl (float): Average document length
        doc_freqs (Counter): Document frequencies for each term
        idf (dict): Inverse document frequency for each term
        vocab_to_index (dict): Mapping from term to vocabulary index
        index_to_vocab (dict): Mapping from vocabulary index to term
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 encoder.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
                Higher values give more weight to term frequency
            b: Length normalization parameter (default: 0.75)
                0 = no length normalization, 1 = full length normalization
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.vocab_to_index: Dict[str, int] = {}
        self.index_to_vocab: Dict[int, str] = {}

    def fit(self, documents: List[str], verbose: bool = True) -> "BM25Encoder":
        """
        Fit the BM25 encoder on a corpus of documents.

        This builds the vocabulary, calculates document frequencies,
        and computes IDF values for all terms.

        Args:
            documents: List of document texts to fit on
            verbose: Whether to print progress messages (default: True)

        Returns:
            self for method chaining
        """
        self.corpus_size = len(documents)
        total_length = 0

        # Calculate document frequencies
        if verbose:
            print(
                f"Calculating document frequencies for {self.corpus_size} documents...",
                file=sys.stderr,
            )

        for i, doc in enumerate(documents):
            if verbose and (i + 1) % 10000 == 0:
                print(
                    f"  Processed {i + 1}/{self.corpus_size} documents", file=sys.stderr
                )

            tokens = self._tokenize(doc)
            total_length += len(tokens)

            # Count unique tokens in document
            unique_tokens = set(tokens)
            self.doc_freqs.update(unique_tokens)

        # Calculate average document length
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0

        if verbose:
            print(f"  Vocabulary size: {len(self.doc_freqs)}", file=sys.stderr)
            print(
                f"  Average document length: {self.avgdl:.2f} tokens", file=sys.stderr
            )

        # Build vocabulary index
        if verbose:
            print("Building vocabulary index...", file=sys.stderr)

        for idx, term in enumerate(sorted(self.doc_freqs.keys())):
            self.vocab_to_index[term] = idx
            self.index_to_vocab[idx] = term

        # Calculate IDF for all terms
        if verbose:
            print("Calculating IDF values...", file=sys.stderr)

        self._calculate_idf()

        if verbose:
            print("BM25 encoder fitted successfully", file=sys.stderr)

        return self

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Simple whitespace tokenization with lowercasing.
        Can be enhanced with more sophisticated tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase and split on whitespace
        # Remove punctuation and empty tokens
        tokens = text.lower().split()
        tokens = [t.strip('.,!?;:()[]{}"\'-') for t in tokens]
        tokens = [t for t in tokens if t and len(t) > 1]  # Filter single chars
        return tokens

    def _calculate_idf(self) -> None:
        """
        Calculate IDF (Inverse Document Frequency) for all terms.

        IDF formula: log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)

        Where:
            - N: Total number of documents
            - n(t): Number of documents containing term t
        """
        for term, freq in self.doc_freqs.items():
            # BM25 IDF formula with smoothing
            idf_value = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )
            self.idf[term] = idf_value

    def encode(self, text: str) -> models.SparseVector:
        """
        Encode text to sparse vector using BM25 scores.

        Args:
            text: Text to encode

        Returns:
            Qdrant SparseVector with indices and values
        """
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        doc_len = len(tokens)

        # Calculate BM25 scores for each token
        scores: Dict[str, float] = {}
        for token, count in token_counts.items():
            if token in self.idf:
                idf = self.idf[token]

                # BM25 formula
                numerator = count * (self.k1 + 1)
                denominator = count + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avgdl)
                )

                score = idf * (numerator / denominator)
                scores[token] = score

        # Convert to sparse vector format
        indices: List[int] = []
        values: List[float] = []

        for token, score in scores.items():
            if token in self.vocab_to_index:
                indices.append(self.vocab_to_index[token])
                values.append(score)

        return models.SparseVector(indices=indices, values=values)

    def encode_batch(self, texts: List[str]) -> List[models.SparseVector]:
        """
        Encode multiple texts to sparse vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of Qdrant SparseVectors
        """
        return [self.encode(text) for text in texts]

    def save(self, filepath: str) -> None:
        """
        Save the fitted BM25 encoder to disk.

        Args:
            filepath: Path to save the encoder
        """
        encoder_data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_freqs": dict(self.doc_freqs),
            "idf": self.idf,
            "vocab_to_index": self.vocab_to_index,
            "index_to_vocab": self.index_to_vocab,
        }

        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(encoder_data, f)

        print(f"BM25 encoder saved to {filepath}", file=sys.stderr)

    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> "BM25Encoder":
        """
        Load a fitted BM25 encoder from disk.

        Args:
            filepath: Path to the saved encoder
            verbose: Whether to print loading info (default: True)

        Returns:
            Loaded BM25Encoder instance

        Raises:
            FileNotFoundError: If encoder file doesn't exist
        """
        with open(filepath, "rb") as f:
            encoder_data = pickle.load(f)

        encoder = cls(k1=encoder_data["k1"], b=encoder_data["b"])
        encoder.corpus_size = encoder_data["corpus_size"]
        encoder.avgdl = encoder_data["avgdl"]
        encoder.doc_freqs = Counter(encoder_data["doc_freqs"])
        encoder.idf = encoder_data["idf"]
        encoder.vocab_to_index = encoder_data["vocab_to_index"]
        encoder.index_to_vocab = encoder_data["index_to_vocab"]

        if verbose:
            print(f"BM25 encoder loaded from {filepath}", file=sys.stderr)
            print(f"  - Corpus size: {encoder.corpus_size}", file=sys.stderr)
            print(f"  - Vocabulary size: {len(encoder.vocab_to_index)}", file=sys.stderr)
            print(
                f"  - Average document length: {encoder.avgdl:.2f} tokens",
                file=sys.stderr,
            )

        return encoder

    def get_stats(self) -> BM25Stats:
        """
        Get statistics about the BM25 encoder.

        Returns:
            BM25Stats dataclass with encoder statistics
        """
        top_terms = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:10]

        return BM25Stats(
            corpus_size=self.corpus_size,
            vocabulary_size=len(self.vocab_to_index),
            average_document_length=self.avgdl,
            k1=self.k1,
            b=self.b,
            top_idf_terms=top_terms,
        )


# Global encoder instance (lazy loaded)
_bm25_encoder: Optional[BM25Encoder] = None
_encoder_path: Optional[str] = None


def get_bm25_encoder(encoder_path: Optional[str] = None) -> Optional[BM25Encoder]:
    """
    Get the global BM25 encoder instance.

    Loads the encoder from disk if available, otherwise returns None.
    The encoder must be fitted and saved first.

    Args:
        encoder_path: Path to the encoder file. If None, uses default
                     location (.data/bm25_encoder.pkl).

    Returns:
        BM25Encoder instance or None if not available
    """
    global _bm25_encoder, _encoder_path

    # If path changed, reset the encoder
    if encoder_path is not None and encoder_path != _encoder_path:
        _bm25_encoder = None
        _encoder_path = encoder_path

    if _bm25_encoder is None:
        path = encoder_path or os.path.join(".data", "bm25_encoder.pkl")
        if os.path.exists(path):
            _bm25_encoder = BM25Encoder.load(path)
            _encoder_path = path
        else:
            print(f"BM25 encoder not found at {path}", file=sys.stderr)
            return None

    return _bm25_encoder


def set_bm25_encoder(encoder: BM25Encoder) -> None:
    """
    Set the global BM25 encoder instance.

    Useful when you've trained an encoder and want to use it globally
    without saving to disk first.

    Args:
        encoder: The BM25Encoder instance to use globally
    """
    global _bm25_encoder
    _bm25_encoder = encoder


def encode_text_to_sparse_vector(text: str) -> Optional[models.SparseVector]:
    """
    Encode text to sparse vector using the global BM25 encoder.

    Args:
        text: Text to encode

    Returns:
        Qdrant SparseVector or None if encoder not available
    """
    encoder = get_bm25_encoder()
    if encoder is None:
        return None

    return encoder.encode(text)


__all__ = [
    "BM25Encoder",
    "BM25Stats",
    "get_bm25_encoder",
    "set_bm25_encoder",
    "encode_text_to_sparse_vector",
]
