from sklearn.base import BaseEstimator, TransformerMixin


class TweetCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('TweetCleaner: fit() called')
        return self

    def transform(self, X, y=None):
        print('TweetCleaner: transform() called')

        # The function has to return transformed data
        return None
