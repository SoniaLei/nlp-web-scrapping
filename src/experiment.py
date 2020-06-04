import json
import pandas as pd
from pipeline import Pipeline

class Experiment:
    """
    Experiment class that orchestrates the execution of an experiment.
    """

    def __init__(self):
        self._config = None
        self._name = None
        self._train_data = None
        self._test_data = None
        self._pipeline = None
        self._metrics = None

    def _validate_datafile(self):
        raise NotImplementedError

    def _read_config(self, conf_path):
        print("Transformed json file to dict")
        self._config = None

        """json_file = open(conf_path)
        try:
            exp_params = json.load(json_file)
            self.exp_conf = exp_params
        except FileNotFoundError:
            pass
        finally:
            json_file.close()"""

    def _csv_to_pandas(self, filename, *args, **kwargs):
        return pd.read_csv(filename, *args, **kwargs)

    def _read_data(self):
        self._train_data = None
        self._test_data = None

        """print("Reading test train files.")
        train = self.exp_conf['traindata_path']
        test = self.exp_conf['testdata_path']
        self.train = self.csv_to_pandas(train)
        self.test = self.csv_to_pandas(test)"""

    def _create_pipeline(self):
        print("Getting pipeline.")
        self._pipeline = Pipeline()
        self._pipeline.init(self._config)

    def init(self, conf_path=None):
        self._name = ""
        self._read_config(conf_path)
        self._read_data()
        self._create_pipeline()

    def run(self):
        """
        Starts experiment.
        """
        print("Starting Experiment")
        self._pipeline.fit()
        predictions = self._pipeline.predict()
     
    def save_mlflow(self):
        pass
