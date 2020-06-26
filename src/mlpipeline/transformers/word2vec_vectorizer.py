from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Converts an array of lists with tokens into an array of lists with embeddings.

    Arguments:
    size -- vector size (dimension) returned for each token
    iter -- number of iterations to fit all tokens to Word2Vec model
    window -- 
    max_embeddings -- maxinimum number of embeddings to store in each list
    """
    
    def __init__(self, size=50, iter=50, window=10, max_embeddings=None):
        self._size = size
        self._iter = iter
        self._window = window
        self._model = None
        self._max_embeddings = max_embeddings

    def fit(self, X, y=None):
        self._model = Word2Vec(X, size=self._size, iter=self._iter, window=self._window, min_count=2)
        
        return self

    def transform(self, X, y=None):
        """
        Converts array of lists with tokens into an array of lists with embeddings.

        Arguments:
        X -- array of lists with tokens
    
        Returns:
        embeddings - array of lists with embeddings, each embedding having a shape (size, )
        """

        embeddings = np.empty((X.shape[0], ), dtype=object)
        i = 0

        for list_tokens in X:

            embeddings[i] = []
            num_tokens = 0

            for token in list_tokens:
                try:
                    embeddings[i].append(self._model.wv[token])

                    num_tokens += 1
                    if self._max_embeddings is not None and num_tokens >= self._max_embeddings:
                        break

                except KeyError:
                    pass

            i += 1

        return embeddings
    