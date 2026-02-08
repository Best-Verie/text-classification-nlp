import numpy as np
from gensim.models import Word2Vec
from typing import List, Iterable


def train_word2vec(sentences: Iterable[Iterable[str]],
                   vector_size: int = 100,
                   window: int = 5,
                   min_count: int = 2,
                   sg: int = 1,
                   epochs: int = 10) -> Word2Vec:
    """Train a Word2Vec model using gensim.

    Args:
        sentences: Iterable of tokenized sentences/documents (list of lists of str).
        vector_size: embedding dimensionality.
        window: context window size.
        min_count: min token frequency.
        sg: 1 for skip-gram, 0 for CBOW.
        epochs: training epochs.

    Returns:
        Trained gensim Word2Vec model.
    """
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=4,
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model


def doc_vector(model: Word2Vec, tokens: List[str]) -> np.ndarray:
    """Compute the average Word2Vec vector for a tokenized document.

    Missing words are ignored. If no tokens match the model vocabulary, returns a zero vector.
    """
    vecs = []
    for tok in tokens:
        if tok in model.wv:
            vecs.append(model.wv[tok])
    if len(vecs) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def build_doc_vectors_w2v(model: Word2Vec, tokenized_texts: Iterable[List[str]]) -> np.ndarray:
    """Build document vectors for a list of tokenized texts using a trained Word2Vec model."""
    return np.vstack([doc_vector(model, tokens) for tokens in tokenized_texts])
