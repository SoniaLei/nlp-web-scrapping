from sklearn import pipeline
from transformers import *
import sklearn


class Pipeline:

    def __init__(self):
        self._pipeline = None

        """
        if self._validate_steps(pipeline_config):
            self.pipeline_steps = None
        """

    def _validate_steps(self, pipe_conf):
        pass

    def init(self, config):
        # loop through all estimators and transformers and instantiate using factory
        self._pipeline = pipeline.Pipeline(steps=[
            ('cleaner', StopWordsRemoval()),
            ('classifier', TweetCleaner()),
            ('vectorizer', sklearn.feature_extraction.text.CountVectorizer()),
            ('estimator', sklearn.linear_model.LogisticRegression()])

        return self

    def fit(self, X=None, y=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        self._pipeline.fit(X, y)
        return self

    def predict(self, X=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        return self._pipeline.predict(X)

    def predict_proba(self, X=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        return self._pipeline.predict_proba(X)
