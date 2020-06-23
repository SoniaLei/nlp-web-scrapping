import string
from sklearn.base import TransformerMixin, BaseEstimator

class PunctuationRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, to_lower=True, remove_stopwords=True):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #the tokenized text from the other transform method is entered here. this is a list of lists.
        #loop through the lists in the list to remove all punctuation. return the result.
        for tweet in X:
            for token in tweet:
                if token in '!"#$%&()*+, -./:;<=>?@[\]^_`{|}~' or token == '...':
                    tweet.remove(token)
        return X


remove_punctuation = PunctuationRemoval()