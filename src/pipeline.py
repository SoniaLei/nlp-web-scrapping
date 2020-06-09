from sklearn import pipeline
from src.transformers import *


class Pipeline:

    def __init__(self, transformers, vectorizers, estimators):
        self.transformers = transformers
        self.vectorizers = vectorizers
        self.estimators = estimators
        self._pipeline = None

    @property
    def transformers(self):
        return self._transformers

    @transformers.setter
    def transformers(self, value):
        if self.follows_parameter_pipeline_format('trasformers', value):
            self._transformers = value

    @property
    def vectorizers(self):
        return self._vectorizers

    @vectorizers.setter
    def vectorizers(self, value):
        if self.follows_parameter_pipeline_format('vectorizers', value):
            self._vectorizers = value

    @property
    def estimators(self):
        return self._estimators

    @estimators.setter
    def estimators(self, value):
        if self.follows_parameter_pipeline_format('estimators', value):
            self._estimators = value

    def follows_parameter_pipeline_format(self, param_name, parameter):
        """Validates parameters have the type of objects expected
        before adding those to the pipeline"""
        if not isinstance(parameter, list):
            raise TypeError(f"Parameter {param_name} passed in pipeline "
                            f"must be of type list found {type(parameter)} instead.")
        for sub_param in parameter:
            if not isinstance(sub_param, tuple):
                raise TypeError(f"For parameter {param_name} passed in pipeline"
                                f"{sub_param} must be of type Tuple found type"
                                f"{type(sub_param)} instead")
            if not isinstance(sub_param[0], str):
                raise TypeError(f"For parameter {param_name} passed in pipeline"
                                f"{sub_param} tuple first value must be of type "
                                f"str found {type(sub_param[0])} instead")
            if not isinstance(sub_param[1], object):
                raise TypeError(f"For parameter {param_name} passed in pipeline"
                                f"{sub_param} tuple first value must be of type "
                                f"instance object found {type(sub_param[1])} instead")
            return True

    def init(self):
        """Puts all transformers one vectorizer for now and first estimator
        This will change once implementing grid search approach"""
        self._pipeline = pipeline.Pipeline(steps=[
           # *self.transformers,
             self.vectorizers[0],  # getting first vec being passed
             self.estimators[0]  # getting first est being passed
        ])
        return self

    def fit(self, X=None, y=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        self._pipeline.fit(X, y)
        return self

    def fit_transform(self, X=None, y=None):
        self._pipeline.fit_transform(X, y)
        return self

    def predict(self, X=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        return self._pipeline.predict(X)

    def predict_proba(self, X=None):
        if self._pipeline is None:
            raise ValueError("Pipeline hasn't been created. Use method create() to create it first.")

        return self._pipeline.predict_proba(X)
