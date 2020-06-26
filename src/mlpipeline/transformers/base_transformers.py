from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # The transformation of data will be done in function transform(), however this function
        # allows to do some initial calculation before the transformation

        return self

    def transform(self, X, y=None):

        # The function has to return transformed data
        return X
