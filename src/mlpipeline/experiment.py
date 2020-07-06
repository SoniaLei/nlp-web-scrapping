"""Experiment module to create, run and store experiments' artifacts."""
from .pipeline import Pipeline
from .context import Context
from .metrics import Metrics
from .data import Data
from .mlflow import MLFlow
import tempfile

class Experiment:
    """
    Experiment class that orchestrates the execution of an experiment.
    """
    
    mlflow_uri_path = '$PROJECT_PATHS$/../../mlruns'
        
    def __init__(self, conf=None, data=None, pipeline=None):
        self.config = conf
        self.data = data
        self.pipeline = pipeline
        self.results = None

    @property
    def config(self):
        return self._conf

    @config.setter
    def config(self, conf):
        if isinstance(conf, Context) or conf is None:
            self._conf = conf
        else:
            raise ValueError("Configuration must be a Context object.")

    @property
    def name(self):
        return self.config.exp_name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data_conf):
        if data_conf is None and self.config is None:
            raise ValueError("Data object must be provided if conf parameter is None")
        if data_conf is None and self.config is not None:
            self._data = Data(train=self.config.train,
                              test=self.config.test,
                              target=self.config.target,
                              features=self.config.features)
        if isinstance(data_conf, Data):
            self._data = data_conf

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline_conf):
        if pipeline_conf is None and self.config is None:
            raise ValueError("Pipeline object must be provided if conf parameter is None")

        if pipeline_conf is None and self.config is not None:
            self._pipeline = Pipeline(self.config.transformers,
                                      self.config.vectorizers,
                                      self.config.estimators).init()

        if isinstance(pipeline_conf, Pipeline):
            self._pipeline = pipeline_conf.init()

    def run(self, probabilities=False):
        pipeline_fitted = self.pipeline.fit(self.data.train_X,
                                            self.data.train_Y)
        classes_ = pipeline_fitted._pipeline.steps[-1][1].classes_ if probabilities else 'outcome'
        
        func = pipeline_fitted.predict_proba if probabilities else pipeline_fitted.predict
        
        predicted = func(self.data.test_X)

        self.results = Metrics(exp_name=self.name,
                               classes=classes_,
                               test_Y=self.data.test_Y,
                               results=predicted)
        
        return self
    
    def parameters_dictionary(self):
        
        estimator = ",".join(str(self.pipeline.estimators).split(",")[1:])[:-2]
        vectoriser = str(self.pipeline.vectorizers).split("'")[2][2:-2]
        transformer = str(self.pipeline.transformers).split("'")[2][2:-2]
        
        params_dict = {"Estimator": estimator,
                       "Vectoriser": vectoriser,
                       "Transformer": transformer}
        
        return params_dict
    
    def metrics_dictionary(self, accuracy=True, precision=True, recall=True, f1=True):
        
        metrics_dict = {}
        
        if accuracy:
            metrics_dict['accuracy'] = self.results.accuracy_score
            
        if precision:
            metrics_dict['precision'] = self.results.precision_score
            
        if recall:
            metrics_dict['recall'] = self.results.recall_score
            
        if f1:
            metrics_dict['f1_score'] = self.results.f1_score
            
        return metrics_dict
    
    def artifact_list(self, confusion_matrix=True, roc_curve=True):
        
        artifact_list = []
        
        if confusion_matrix:
            artifact_list.append(self.results.plot_confusion_matrix())
            
        if roc_curve:
            artifact_list.append(self.results.plot_classes_roc())
            
        return artifact_list            

    def save_to_mlflow(self, uri_path=mlflow_uri_path,
                       accuracy=True, precision=True, recall=True, f1=True,
                       confusion_matrix=True, roc_curve=True):
        
        mlflow = MLFlow(experiment_name=self.name,
                        tracking_uri=uri_path)
        
        # Parameters        
        params_dict = self.parameters_dictionary()
        
        # Metrics        
        metrics_dict = self.metrics_dictionary(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
          
        # Artifacts        
        artifact_list = self.artifact_list(confusion_matrix=confusion_matrix, roc_curve=roc_curve)

        # Logging        
        mlflow.Logging(params_dictionary=params_dict,
                      metrics_dictionary=metrics_dict,
                      artifact_filepaths=artifact_list)
        
        print("Parameters, Metrics and Artifacts logged!")