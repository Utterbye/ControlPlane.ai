"""
SHARED ENCODER

Detector 2 (unsafe intent) and Detector 4 (high-stakes topic) both turn text
into vectors. If each one loaded its own model we would pay for the model
twice - twice the memory, twice the start-up time - for no benefit.

So the model lives here, loaded once, and both detectors borrow it.

Primary  : sentence-transformers all-MiniLM-L6-v2
Fallback : TF-IDF from scikit-learn, so the file always runs with no download

The fallback needs to be FITTED on some text before it can transform anything,
which is why register_corpus() exists. Every detector hands over its example
sentences at start-up, we fit once on all of them together, and after that any
detector can encode anything.
"""

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_name = None          # "minilm" or "tfidf"
_model = None
_corpus = []          # only used by the TF-IDF fallback
_fitted = False


def register_corpus(texts, space=None):
    """
    A detector calls this at import time with its example sentences.
    Harmless for MiniLM. Essential for TF-IDF, which cannot transform text
    until it has seen a vocabulary.
    """
    global _fitted
    _corpus.extend(texts)
    _fitted = False           # new text means the fallback must be refitted


def _l2(mat):
    """Make every row length 1, so a dot product IS the cosine similarity."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-10)


def load(verbose: bool = False):
    """Load the encoder once. Safe to call as often as you like."""
    global _name, _model, _fitted
    if _model is not None and _fitted:
        return _name

    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
            _name = "minilm"
        except Exception as e:
            from sklearn.feature_extraction.text import TfidfVectorizer
            _model = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1)
            _name = "tfidf"
            if verbose:
                print(f"sentence-transformers unavailable ({type(e).__name__}), "
                      f"using the TF-IDF fallback")

    if _name == "tfidf" and not _fitted:
        # fit on the example sentences only - never on live queries
        if not _corpus:
            # Silent failure trap: with no corpus the fallback fits on nothing,
            # every vector comes out all zeros, and every similarity is 0.000.
            # That looks like "no match" instead of "broken", so it must shout.
            raise RuntimeError(
                "shared_encoder: no corpus registered. Import the detector "
                "modules (tier1_unsafe / tier1_stakes) before encoding, or "
                "call register_corpus() yourself.")
        _model.fit(_corpus)

    _fitted = True
    return _name


def encode(texts, space=None):
    """Text in, normalised vectors out. The only place that knows the encoder."""
    load()
    texts = list(texts)
    if _name == "minilm":
        return _l2(_model.encode(texts, normalize_embeddings=True))

    vecs = _model.transform(texts).toarray()
    # An all-zero row means not one word of this text is in the vocabulary.
    # Cosine against it is 0.000 for everything, which silently reads as
    # "nothing matched". Flag it rather than let it look like a clean result.
    dead = int((vecs.sum(axis=1) == 0).sum())
    if dead:
        globals()["_last_oov"] = dead
    return _l2(vecs)


def name():
    return load()


def dims():
    return encode(["probe"]).shape[1]
