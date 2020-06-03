from sklearn.base import BaseEstimator, TransformerMixin


class StopWordsRemoval(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('StopWordsRemoval: fit() called')
        return self

    def transform(self, X, y=None):
        print('StopWordsRemoval: transform() called')

        # The function has to return transformed data
        return None


class Lematization(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('Lematization: fit() called')
        return self

    def transform(self, X, y=None):
        print('Lematization: transform() called')

        # The function has to return predicted values
        return None


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

class Estimator(BaseEstimator):
    """temporal for demo purposes
    latter this will need to be removed"""
    def __init__(self):
        pass

    def fit(self, x, y):
        print('Estimator: fit() called')
        return self


    def predict(self, x):
        print('Estimator: predict() called')
        return None

    def predict_proba(self, x):
        print('Estimator: predict_proba() called')
        return None

