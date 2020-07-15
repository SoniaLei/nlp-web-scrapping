"""
Gridsearch module for GridSearch object and hyperparameter tunning
"""
from sklearn.metrics import SCORERS
from sklearn.model_selection import GridSearchCV


class GridSearch:

    def __init__(self, pipeline, parameters, cv=None, scoring='accuracy',
                 n_jobs=-1, refit=True, verbose=3, return_train_score=True):
        self.pipeline = pipeline
        self.parameters = parameters
        self.cv = cv or 0
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.sk_gridsearch = None

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, value):
        if not isinstance(value, (str, callable, list, tuple)):
            raise TypeError(f"Scoring parameter in GridSearch invalid."
                            f"Possible options are {SCORERS.keys()}")
        self._scoring = value

    @property
    def sk_gridsearch(self):
        return self._sk_gridsearch

    @sk_gridsearch.setter
    def sk_gridsearch(self, _):
        self._sk_gridsearch = GridSearchCV(estimator=self.pipeline,
                                           param_grid=self.parameters,
                                           n_jobs=self.n_jobs,
                                           cv=self.cv,
                                           refit=self.refit,
                                           verbose=self.verbose,
                                           scoring=self.scoring,
                                           return_train_score=self.return_train_score)

    def fit(self, X=None, y=None):
        return self.sk_gridsearch.fit(X, y)

    def predict(self, X=None):
        return self.sk_gridsearch.predict(X)

    def predict_proba(self, X=None):
        return self.sk_gridsearch.predict_proba(X)

    @property
    def classes(self):
        return self.sk_gridsearch.best_estimator_.classes_

    @property
    def best_estimator(self):
        return self.sk_gridsearch.best_estimator_

    @property
    def cv_results(self):
        return self.sk_gridsearch.cv_results_

    @property
    def best_score(self):
        return self.sk_gridsearch.best_score_
