# Imports
import mlflow
from urllib.parse import urlparse
import mlflow.sklearn

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
        print("Getting 'experiment_name' ...")
        return self._experiment_name
    
    @experiment_name.setter
    def experiment_name(self, experiment_name):
        print("Setting 'experiment_name' ...")
        self._experiment_name = experiment_name
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# tracking_uri
    # The filepath the results folder will be saved in
    
    @property
    def tracking_uri(self):
        print("Getting 'tracking_uri' ...")
        return self._tracking_uri
    
    @tracking_uri.setter
    def tracking_uri(self, tracking_uri):
        print("Setting 'tracking_uri' ...")
        
        uri = ""
        
        if tracking_uri == None:
            
            uri = None
        
        else:
            
            filepath_list = tracking_uri.split("/")
            
            # The uri needs to end with the mlruns folder
            if filepath_list[-1] != "mlruns":
            
                tracking_uri += "/mlruns"
                
                # To set, uri needs "file:///" at the start
                uri = "file:///" + str(tracking_uri)

        self._tracking_uri = tracking_uri
        
        # Set mlflow tracking_uri
        mlflow.set_tracking_uri(uri)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
# Functions

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    """
    Get experiment_id using self.experiment_name
    """

    def Experiment_Id(self):
        
        print("Attaining 'experiment_id' ...")
        
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
                
                print("New experiment created in tracking_uri filepath:\n" + str(self.tracking_uri))
            
        return experiment_id
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    """
    Logging to mlflow():
        Inputs:
            Parameters - Dictionary
            Metrics    - Dictionary
            Artifacts  - List (Optional)
    """

    def Logging(self, params_dictionary, metrics_dictionary, artifact_filepaths = []):
        
        print("Starting 'MLFlow_Logging()' ...")
                
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
            if isinstance(artifact_filepaths, list): 
                for filepath in artifact_filepaths:

                    try:

                        mlflow.log_artifact(filepath)

                    except FileNotFoundError:

                        print("Unable to upload artifact")
                        print("-- File not found --")

            elif isinstance(artifact_filepaths, str):
                
                try:
                    
                    mlflow.log_artifact(artifact_filepaths)
                    
                except FileNotFoundError:
                    
                    print("Unable to upload artifact")
                    print("-- File not found --")