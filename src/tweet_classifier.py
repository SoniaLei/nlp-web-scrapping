from sklearn.base import BaseEstimator


class TweetClassifier(BaseEstimator):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('TweetClassifier: fit() called')
        return self

    def predict(self, X, y=None):
        print('TweetClassifier: predict() called')

        # The function has to return predicted values
        return None
