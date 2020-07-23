from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    The vectorizer converts a list of tokens into embeddings and returns a vector, that represents
    element-wise average of embeddings for each sample. The embeddings are calculated during fit()
    method. Current implementation doesn't use pretrained embeddings.

    Arguments:
        size - number of dimensions of the returned vector
        iters - number of iterations to train Word2Vec model and calculate embeddings for each token
        window - maximum distance between the current and predicted word within a sentence
    """
    
    def __init__(self, size=50, iters=50, window=10):
        self.size = size
        self.iters = iters
        self.window = window
        self.model = None

    def fit(self, x, y=None):
        self.model = Word2Vec(x, size=self.size, iter=self.iters, window=self.window, min_count=2)
        
        return self

    def transform(self, x):
        """
        Arguments:
            x - an array of lists with tokens

        Returns:
            vectors - an array of shape (num_samples, size)
        """

        vectors = np.empty((x.shape[0], self.size))
        i = 0

        for tokens in x:

            vectorized_tokens = []

            for t in tokens:
                try:
                    vectorized_tokens.append(self.model.wv[t])
                except KeyError:
                    pass

            if len(vectorized_tokens) > 0:
                vectors[i] = np.mean(vectorized_tokens, axis=0)
            
            i += 1

        return vectors
