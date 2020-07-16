# Imports
import mlflow
from urllib.parse import urlparse
import mlflow.sklearn
from tempfile import NamedTemporaryFile
import os

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

class MLFlow:
    
# Constructor
    
    def __init__(self, experiment_name = None, tracking_uri = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
        
# Getters and Setters

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# experiment_name
    # The name of the experiment the results will be saved in for the MLFlow UI

    @property
    def experiment_name(self):
        return self._experiment_name
    
    @experiment_name.setter
    def experiment_name(self, experiment_name):
        
        if isinstance(experiment_name, str):
            self._experiment_name = experiment_name
            
        else:
            raise TypeError("Input needs to be a String")
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# tracking_uri
    # The filepath the results folder will be saved in
    
    @property
    def tracking_uri(self):
        return self._tracking_uri
    
    @tracking_uri.setter
    def tracking_uri(self, tracking_uri):
        
        if isinstance(tracking_uri, str):
        
            print("--" * 50)

            print("Tracking_uri: " + str(tracking_uri))

            uri = ""

            if tracking_uri == None:

                print("In 'None'")

                uri = None

            else:

                print("Not in 'None'")

                filepath_list = tracking_uri.split("\\")

                # The uri needs to end with the mlruns folder
                if filepath_list[-1] != "mlruns":

                    tracking_uri += "\mlruns"

                # To set, uri needs "file:///" at the start
                #uri = "file:///" + str(tracking_uri)
                uri = "file:/" + str(tracking_uri)

            self._tracking_uri = tracking_uri

            print("self._tracking_uri: " + str(self.tracking_uri))

            print("URI: " + str(uri))

            # Set mlflow tracking_uri
            mlflow.set_tracking_uri(uri)

            print("--" * 50)
            
        else:
            raise TypeError("Input needs to be a String")            

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
# Functions

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    """
    Get experiment_id using self.experiment_name
    """

    def Experiment_Id(self):
        
        experiment_name = self.experiment_name
        
        if experiment_name == None:
            
            experiment_id = None
            
        else:
        
            # Get experiment_id from already created experiment
            try:
                experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                # Inputs:
                    # name = the experiment's name
                # Returns: Experiment object
                    # .experiment_id = Integer - value wanted from object

            # Create new experiment and get experiment_id
            except AttributeError: # Error: if experiment doesn't exist ...
                experiment_id = mlflow.create_experiment(experiment_name)
                # Inputs:
                    # name = the experiment's name (unique)
                    # artifact_location (optional) = location to store run artifacts
                # Returns: integer Id of the experiment
                
                print("New experiment created in tracking_uri filepath:" +\
                      "\n  tracking_uri: " + str(self.tracking_uri) +\
                      "\n  experiment_name: " + str(experiment_name))
            
        return experiment_id
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    """
    Using TemporaryFiles to log figures/artifacts
        Input:
            Artifact - figure (usually matplotlib)
    """
    
    def LogArtifact_TempFile(self, artifact):
        
        tmpfile = NamedTemporaryFile(
            delete=False,
            prefix="artifact_",
            suffix=".png"
        )

        artifact.savefig(tmpfile)

        tmpfile.close()
        
        mlflow.log_artifact(tmpfile.name)

        os.remove(tmpfile.name)
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    """
    Logging to mlflow():
        Inputs:
            Parameters - Dictionary
            Metrics    - Dictionary
            Artifacts  - List (Optional)
    """

    def Logging(self, params_dictionary, metrics_dictionary = {}, artifact_filepaths = []):
                
        # Getting experiment_id
        experiment_id = self.Experiment_Id()
        
        # Start mlflow():
        with mlflow.start_run(experiment_id = experiment_id):
            
            # Parameters
            for key in params_dictionary:
                
                mlflow.log_param(key, params_dictionary[key])
            
            # Metrics
            for key in metrics_dictionary:
                
                mlflow.log_metric(key, metrics_dictionary[key])
                
            # Figures / Artifacts            
            for artifact in artifact_filepaths:
                
                try:
                    self.LogArtifact_TempFile(artifact)
                    
                except FileNotFoundError:
                    
                    print("Unable to upload artifact")
                    print("-- File not found --")