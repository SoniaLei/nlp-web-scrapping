"""
Experiment module to create, run and store experiments' artifacts.
"""
import tempfile
from itertools import combinations
from .metrics import Metrics
import mlflow
import numpy as np


class Experiment:
    """
    Experiment class that orchestrates the execution of an experiment.
    """
    mlflow_uri_path = '$PROJECT_PATHS$/../../mlruns'

    def __init__(self, name, predictions, classes, test_Y):
        self.name = name
        self.predictions = predictions
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

    def save_to_mlflow(self):
        #mlflow.set_tracking_uri(Experiment.mlflow_uri_path)
        print("mlflow_uri_path.join(self.name)  --> ", self.mlflow_uri_path.join(self.name))
        mlflow.set_experiment(self.mlflow_uri_path.join(self.name))

       # print(mlflow.get_artifact_uri())
      #  print(mlflow.get_tracking_uri())

        with mlflow.start_run(run_name=self.name):
            mlflow.log_param('Exp name', self.metrics.exp_name)
            mlflow.log_param('f1 score', self.metrics.f1_score())
            mlflow.log_metric('accuracy', self.metrics.accuracy_score)

            for plot in [self.metrics.plot_confusion_matrix(), self.metrics.plot_model_roc()]:

                with tempfile.NamedTemporaryFile(suffix=".png", prefix="Plot_", delete=False) as tmpfile:
                    plot.savefig(tmpfile, format='png')
                    tmpfile.seek(0)
                    mlflow.log_artifact(tmpfile.name)

            with tempfile.NamedTemporaryFile(suffix=".png", prefix="Plot_", delete=False) as tmpfile:
                self.metrics.plot_classes_roc().savefig(tmpfile, format='png')
                tmpfile.seek(0)
                mlflow.log_artifact(tmpfile.name)

        self.saved = True
        print(f"Experiment name ({self.name}) has been saved in MlFlow.")


class Experiments:
    """
    Experiments class that orchestrates the execution of multiple `Experiment`,
    including combination of experiments.
    """
    runtime_save = True
    test_Y = None
    classes = None
    collection = {}

    def add_experiment(self, exp_name, predictions):
        experiment = Experiment(name=exp_name,
                                predictions=predictions,
                                classes=Experiments.classes,
                                test_Y=Experiments.test_Y)
        if self.runtime_save:
            experiment.save_to_mlflow()

        self.collection[exp_name] = experiment

    def save_experiments(self, experiments=None):
        experiments = experiments or self.collection
        for name, exp in experiments.items():
            exp.save_to_mlflow()

    def add_experiment_combinations(self):

        combined_models_names = []
        # we start at 1 since we want only aggregation of results
        # from 2+ models.
        num_models = list(range(1, len(self.collection.keys())))

        for num_model in num_models:
            # combinations from 2+ models results.
            combination = combinations(self.collection.keys(), num_model + 1)
            combined_models_names.extend(list(combination))

        for num, combined_model in enumerate(combined_models_names):
            num_models = len(combined_model)
            weight = 1 / num_models

            probabilities = np.zeros((len(self.test_Y), len(self.classes)))

            for model in combined_model:

                prob_weighted = self.collection[model].predictions * weight
                probabilities += prob_weighted

            combined_model_name = "__".join(combined_model)

            self.add_experiment(exp_name=combined_model_name,
                                predictions=probabilities)

        return self
