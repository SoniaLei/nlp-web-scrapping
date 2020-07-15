"""
Module for creating and organizing a collection of nlp pipelines,
with its respective parameters, cross validation and name information.
"""
from .gridsearch import GridSearch
from .experiment import Experiments
from sklearn import pipeline


class Pipelines:
    """
    Orchestrator - Creates a list of many nlp pipelines with its respective
    names, parameters and steps/sequences of transformations with one
    vectorizer and one estimator at the end of each pipe line.
    """

    def __init__(self, exp_name, data, transformers, vectorizers, estimators,
                 cv_keyword='cv'):
        """
        Initiates a Pipelines Object by given data, transformers, vectorizers,
        and estimators. Note, vectorizers can be `None` if so, pipelines assumes
        last transformer in transformers is a vectorizer. Also, transformers can
        be `None`. If so, Pipelines assumes data already is transformed and only
        requires vectorizers and estimators.

        """
        self.exp_name = exp_name
        self.data = data
        self.transformers = transformers
        self.vectorizers = vectorizers
        self.estimators = estimators
        self.cv_keyword = cv_keyword

        self.names = self.set_pipelines_names()
        self.sequences = self.set_pipelines_lists()
        self.parameters = self.set_pipelines_parameters()
        self.cvs = self.set_pipelines_cv()
        self.predictions = {}
        self.experiments = Experiments()

    def set_pipelines_names(self):
        """
        Loops through transformes vectorizers and estimators names from conf file,
        and concatenates all steps in one single `str` name used as detailed
        experiment name. EX: 'exp_005_22_07_2020_Tokenizer_CountVectorizer_SVC'.
        """
        if self.vectorizers.names:
            names = [self.exp_name + '_' + self.transformers.names + '_' + vectorizer + '_' + estimator
                     for vectorizer in self.vectorizers.names
                     for estimator in self.estimators.names]
        else:
            names = [self.exp_name + '_' + self.transformers.names + '_' + estimator
                     for estimator in self.estimators.names]
        return names

    def set_pipelines_lists(self):
        """
        Creates multiple nlp pipelines by looping through each vectorizer and
        estimator. Note if no vectorizers all pipelines will implement all the
        transformers and one estimator from estimators. If multiple vectorizers,
        All pipelines will have all transformers plus a combination of one
        vectorizer and one estimator.
        """
        if self.vectorizers.sequence:
            pipelines_lists = [self.transformers.sequence + [vectorizer] + [estimator]
                               for vectorizer in self.vectorizers.sequence
                               for estimator in self.estimators.sequence]

        else:
            pipelines_lists = [self.transformers.sequence + [estimator]
                               for estimator in self.estimators.sequence]
        return pipelines_lists

    def set_pipelines_parameters(self):
        """
        Creates multiple parameter dictionaries for each pipeline created.
        Ready to be injected in `GridSearchCV` object.
        """
        if self.vectorizers.parameters:
            parameters = [{**self.transformers.parameters, **vectorizer, **estimator}
                          for vectorizer in self.vectorizers.parameters
                          for estimator in self.estimators.parameters]
        else:
            parameters = [{**self.transformers.parameters, **estimator}
                          for estimator in self.estimators.parameters]
        return parameters

    def set_pipelines_cv(self):
        """
        Extracts cross validation information from estimators objects.
        """
        if self.vectorizers.names:
            cvs = [cv[self.cv_keyword]
                   for _ in self.vectorizers.names
                   for cv in self.estimators.cvs]
        else:
            cvs = [cv[self.cv_keyword]
                   for cv in self.estimators.cvs]
        return cvs

    def start_runs(self, safe_run=True):
        """
        Sets Experiments general characteristics and loop through all pipelines,
        fitting and getting predictions. Adding model/pipeline predictions to
        Experiments collection. if safe run saves metrics to `MlFlow` before
        fitting the next pipeline. Otherwise, fits/predicts all pipelines and
        saves all pipelines results at the end.
        """
        # Setting Experiments safe_run as a default!
        # TODO maybe have this one in config file?
        Experiments.runtime_save = safe_run
        Experiments.test_Y = self.data.test_Y

        for name, pipe_seq, params, cv in zip(self.names, self.sequences, self.parameters, self.cvs):
            pipeline = Pipeline(name=name,
                                steps=pipe_seq,
                                parameters=params)
            gridsearch = GridSearch(pipeline=pipeline.sk_pipeline,
                                    parameters=params,
                                    cv=cv)
            gridsearch.fit(X=self.data.train_X, y=self.data.train_Y)

            predictions = gridsearch.predict_proba(X=self.data.test_X)

            if Experiments.classes is None:
                Experiments.classes = gridsearch.classes

            self.experiments.add_experiment(exp_name=name,
                                            predictions=predictions,
                                            gridsearch=gridsearch,
                                            is_aggregated_model=False)
        return self.experiments

    def __len__(self):
        return len(self.names)


class Pipeline:
    """
    Pipeline object storing single pipeline information,
    providing sklearn pipeline plugin 'if needed in future'.
    """
    def __init__(self, *, name, steps, parameters):
        """
        Initializing pipeline object.
        """
        self.name = name
        self.steps = steps
        self.parameters = parameters
        self.sk_pipeline = steps

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) and len(value) > 0:
            self._name = value

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, values):
        self._steps = values

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        self._parameters = values

    @property
    def sk_pipeline(self):
        """
        Returns a sklearn pipeline object.
        """
        return self._sk_pipeline

    @sk_pipeline.setter
    def sk_pipeline(self, steps):
        """
        Initializes a pipeline from sklearn with steps from
        Pipelines combinations.
        """
        self._sk_pipeline = pipeline.Pipeline(steps=steps)
