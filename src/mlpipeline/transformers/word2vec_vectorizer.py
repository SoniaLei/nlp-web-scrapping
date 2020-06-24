from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    """
    
    def __init__(self, size=50, iter=50, window=10):
        self._size = size
        self._iter = iter
        self._window = window
        self._model = None

    def fit(self, X, y=None):
        """

        """
        self._model = Word2Vec(X, size=self._size, iter=self._iter, window=self._window, min_count=2)
        
        return self

    def transform(self, X, y=None):

        vectors = np.empty((X.shape[0], self._size))
        i = 0

        for tokens in X:

            vectorized_tokens = []

            for t in tokens:
                try:
                    vectorized_tokens.append(self._model.wv[t])
                except KeyError:
                    pass

            if len(vectorized_tokens) > 0:
                vectors[i] = np.mean(vectorized_tokens, axis=0)
            
            i += 1

        return vectors
    