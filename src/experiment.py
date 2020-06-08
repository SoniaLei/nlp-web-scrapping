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
        self._train_X, self._train_Y = None, None
        self._test_X, self._test_Y = None, None
        self._pipeline = None
        self._metrics = None

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

    def _csv_to_pandas(self, datafile='train'):
        
        data = None
        
        if datafile == 'train':
            # load train data file
            pass
        elif datafile == 'test':
            # load test data file
            pass
        else:
            raise ValueError("Parameter 'datafile' accepts only 'train' and 'test' values")
        
        return data

    def _read_data(self):
        train = self._csv_to_pandas('train')
        test = self._csv_to_pandas('test')

        # needs to be changed to handle situations when train or test data is not loaded correctly
        if train is None or test is None:
            return

        self._train_X = train['text']
        self._train_Y = train['sentiment']

        self._test_X = test['text']
        self._test_Y = test['sentiment']

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

        # print/output experiment results
     
    def save_mlflow(self):
        pass
