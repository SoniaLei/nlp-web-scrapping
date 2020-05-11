from sklearn.base import BaseEstimator, TransformerMixin


class TweetVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('TweetVectorizer: fit() called')
        return self

    def transform(self, X, y=None):
        print('TweetVectorizer: transform() called')

        # The function has to return transformed data
        return None

