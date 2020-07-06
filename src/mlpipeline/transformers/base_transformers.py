from sklearn.base import BaseEstimator, TransformerMixin


class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy transformer is an empty transformer that doesn't do anything. This transformer is required to workaround
    the current design of pipeline, which requires at least one transformer and one vectorizer specified. 
    
    Example of use: 
    When TfidfVectorizer is used as a vectorizer, it implements both data transformation (tokenization, etc.) and
    vectorization. Therefore, a seperate transformer is not required any more.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # The transformation of data will be done in function transform(), however this function
        # allows to do some initial calculation before the transformation

        return self

    def transform(self, X, y=None):

        # The function has to return transformed data
        return X
