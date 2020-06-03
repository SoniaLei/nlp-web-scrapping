import json
import pandas as pd
from src.pipeline import Pipeline

class Experiment:
    def __init__(self):
        self.exp_conf = None
        self.exp_name = None
        self.train = None
        self.test = None
        self.crosval = None
        self.pipeline = None
        self.metrics = None

    def _validate_datafile(self):
        raise NotImplementedError

    def read_json(self, conf_path):
        print('Experiment.read_json()')

        print("Transformed json file to dict")
        json_file = open(conf_path)
        try:
            exp_params = json.load(json_file)
            self.exp_conf = exp_params
        except FileNotFoundError:
            pass
        finally:
            json_file.close()

    def csv_to_pandas(self, filename, *args, **kwargs):
        return pd.read_csv(filename, *args, **kwargs)

    def _read_test_train(self):
        print("Reading test train files.")
        train = self.exp_conf['traindata_path']
        test = self.exp_conf['testdata_path']
        self.train = self.csv_to_pandas(train)
        self.test = self.csv_to_pandas(test)

    def create_pipeline(self):
        print("Getting pipeline.")
        self.pipeline = Pipeline(self.train,
                                  self.test,
                                  self.exp_conf['pipeline_steps'])
        self.pipeline.create()


    def run(self):
        print("Starting Experiment")
        self._read_test_train()
        self.create_pipeline()
        self.pipeline.fit()
        predictions = self.pipeline.predict_proba()

    def save_mlflow(self):
        pass


