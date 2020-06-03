from sklearn import pipeline
from src.registry import *
from src.transformers import Estimator


class Pipeline:

    def __init__(self, train, test, pipeline_config, cv_=5):
        #self.exp_name = exp_name
        self.train = train
        self.test = test
        self._pipeline_config = pipeline_config
        self.cv_ = cv_
        if self._validate_steps(pipeline_config):
            self.pipeline_steps = None

    def _validate_steps(self, pipe_conf):
        """
        """
        return True

    def create(self):
        # loop through all estimators and transformers and instantiate using factory
        self.pipeline = pipeline.Pipeline(steps=[
            ('cleaner', StopWordsRemoval()),
            ('classifier', TweetCleaner()),
            ('vectorizer', Lematization()),
            ('estimator', Estimator())
        ])

        return self

    def fit(self, X=None, y=None):
        print("Fitting the pipeline")
        self.pipeline.fit(X, y)
        return self

    def predict(self, X=None):
        return self.pipeline.predict(X)

    def predict_proba(self, X=None):
        return self.pipeline.predict_proba(X)
