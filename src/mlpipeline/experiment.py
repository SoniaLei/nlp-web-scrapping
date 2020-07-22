"""
Experiment module to create, run and store experiments' artifacts.
"""
from itertools import combinations
from .metrics import Metrics
from .mlflow import MlFlow
import numpy as np
import os


class Experiment:
    """
    Experiment class that orchestrates the execution of an experiment.
    """

    path = os.path.abspath(os.path.dirname(""))

    def __init__(self, name, predictions, gridsearch, is_aggregated_model, classes, test_Y):
        self.name = name
        self.predictions = predictions
        self.gridsearch = gridsearch
        self.is_aggregated_model = is_aggregated_model
        self.metrics = Metrics(exp_name=name,
                               predictions=predictions,
                               classes=classes,
                               test_Y=test_Y)
        # Tag to track if experiment has been logged in `MlFlow`.
        self.saved = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, value):
        self._predictions = value

    def get_parameters_dict(self):
        """
        Gets gridSearch best estimator model and
        loops through each step of the pipeline and
        for each step gets the object `transformers, vectorizers, estimator`,
        gets their parameters and save/returns them in a param dict.
        """
        param_dict = {}
        if self.gridsearch:
            pipeline = self.gridsearch.best_estimator
            for step_name, step_obj in pipeline.steps:
                param_dict.update({step_name+'_' + param: str(value)
                                   for param, value in step_obj.get_params().items()})
        return param_dict

    def get_metrics_dict(self):
        """
        Gets Information from classification report dict and saves/returns
        those values along with overall model accuracy and F1 score.
        """
        # TODO Add accuracy score for each of the classes One Vs All method
        metrics_dict = {}
        for label, dic_values in self.metrics.classification_report.items():
            if isinstance(dic_values, dict):
                for metric, value in dic_values.items():
                    metrics_dict[label + "_" + metric] = value

        metrics_dict.update({'Accuracy': self.metrics.accuracy_score,
                             'F1': self.metrics.f1_score})
        return metrics_dict

    def get_artifact_objs(self):
        """
        Gets plots objects from Metrics class and returns the
        plot names and objects in a dict format.
        """
        confusion_matrix = self.metrics.plot_confusion_matrix()
        single_roc_curve = self.metrics.plot_single_roc_curve()
        multi_roc_curves = self.metrics.plot_multi_label_roc_curves()
        return {'confusion_matrix': confusion_matrix,
                'single_roc_curve': single_roc_curve,
                'multi_roc_curves': multi_roc_curves}

    def save_to_mlflow(self):
        """
        Instantiate an MlFlow object and uses logging to log:
        prams, metrics, artifacts along side if the model is an
        aggregation of other models.
        """
        # TODO check if works for Nat and Ernest mlflow_uri_path
        mlflow = MlFlow(self.name, self.path)
        # TODO params dict will be empty for combined models results
        # address in next sprint if needed
        mlflow.logging(params_dictionary=self.get_parameters_dict(),
                       metrics_dictionary=self.get_metrics_dict(),
                       artifact_objs=self.get_artifact_objs(),
                       is_aggregated=self.is_aggregated_model)
        self.saved = True


class Experiments:
    """
    Experiments class that orchestrates the execution of multiple `Experiment`,
    including combination of experiments.
    """
    runtime_save = True
    test_Y = None
    classes = None
    collection = {}

    def add_experiment(self, exp_name, predictions, gridsearch, is_aggregated_model):
        """
        Adds an experiment object inside Experiments collection dictionary.
        If runtime_save is `True` it saves the experiment right after. Otherwise,
        It logs the experiment at the end after fitting and predicting all pipelines.
        """
        experiment = Experiment(name=exp_name,
                                predictions=predictions,
                                gridsearch=gridsearch,
                                is_aggregated_model=is_aggregated_model,
                                classes=Experiments.classes,
                                test_Y=Experiments.test_Y)
        if self.runtime_save:
            experiment.save_to_mlflow()

        self.collection[exp_name] = experiment

    def save_experiments(self, experiments=None):
        """
        Saves multiple `Experiments` at one.
        Experiments must be a `dict` object {exp_name, `Experiment`}.
        If no experiment has been passed it will attempt to log
        experiments from collection.
        """
        experiments = experiments or self.collection
        for name, exp in experiments.items():
            exp.save_to_mlflow()

    def add_experiment_combinations(self):
        """
        Loops through all `Experiments` in collection,
        and computes the aggregation of the different probabilities results.
        Creates a new experiment name by concatenating the experiments names
        used for that combination.
        Logs The experiment in collection and this is saved in MlFlow.
        """
        combined_models_names = []
        # we start at 1 since we want only aggregation of results
        # from 2+ models.
        num_models = list(range(1, len(self.collection.keys())))

        for num_model in num_models:
            # combinations from 2+ models results.
            combination = combinations(self.collection.keys(), num_model + 1)
            combined_models_names.extend(list(combination))

        print("Number of combined/aggregated ml results: ", len(combined_models_names))
        print("Combined Model Names: ")
        [print(" - ", "__".join(name)) for name in combined_models_names]
        print("\n")

        for num, combined_model in enumerate(combined_models_names):
            num_models = len(combined_model)
            weight = 1 / num_models

            probabilities = np.zeros((len(self.test_Y), len(self.classes)))

            for model in combined_model:

                prob_weighted = self.collection[model].predictions * weight
                probabilities += prob_weighted

            combined_model_name = "__".join(combined_model)

            self.add_experiment(exp_name=combined_model_name,
                                predictions=probabilities,
                                gridsearch=None,
                                is_aggregated_model=True)

        return self
