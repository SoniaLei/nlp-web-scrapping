import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Tokenizer(BaseEstimator, TransformerMixin):
    """
    Takes a string as input and returns a list of tokens.

    Arguments:
        to_lower - boolean, convert all tokens to lowercase
        remove_stopwords - boolean, remove stopwords
    """

    def __init__(self, to_lower=True, remove_stopwords=True):
        self._to_lower = to_lower
        self._remove_stopwords = remove_stopwords

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            self._stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self._stop_words = set(stopwords.words('english'))
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """
        Arguments:
            x - an array or pandas Series of strings, or a single string

        Returns:
            A numpy array, where each element is a list of tokens.
        """

        if isinstance(x, pd.Series):
            data = x.values
        elif isinstance(x, np.ndarray):
            data = x
        elif isinstance(x, str):
            data = np.array([x])
        else:
            raise TypeError('X must be either pandas Series, numpy array or string')
        
        num_elements = data.shape[0]

        tokenized_text = np.empty((num_elements, ), dtype=object)

        i = 0
        for x in data:
            if self._remove_stopwords:
                tokenized_text[i] = [token for token in nltk.word_tokenize(str.lower(str(x)) if self._to_lower else x) if token not in self._stop_words]
            else:
                tokenized_text[i] = [token for token in nltk.word_tokenize(str.lower(str(x)) if self._to_lower else x)]
            i += 1

        return tokenized_text
