from .src.mlpipeline.mlflow import MLFlow

# Test 1
print("Initialising a blank MLFlow() object - return None in both parameters")
print()

mlf = MLFlow()

print()

print(mlf.experiment_name)

print()

print(mlf.tracking_uri)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 2
print("Initialised an MLFlow() object - return correct parameters:\n" \
      "experiment_name = 'Changed_Experiment'\n" \
      "tracking_uri = "r"'C:\Users\nathi_000\Desktop\Python Files\NLP Project\nlp-web-scrapping\mlruns'")
print()

mlf = MLFlow("Changed_Experiment",
             r"..\..\mlruns")

print()

print(mlf.experiment_name)

print()

print(mlf.tracking_uri)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Setting Parameters for logging

LogReg = "Logisitic Regression Model"
Vect = "Count Vectoriser"

params_dict = {"model": LogReg,
              "vectoriser": Vect}

metrics_dict={"accuracy": 0.753,
             "precision": 0.853}

# Only work on Nathanael's computer

artifact_list = [
    r"mlflow_artifacts\CM_08-06-2020__13_11_47.png",
    r"mlflow_artifacts\CM_08-06-2020__13_11_56.png"
]

#Faulty List
artifact_list_broken = [
    r"mlflow_artifacts\WRONG_DIRECTORY\CM_08-06-2020__13_11_47.png", # this directory doesn't exist
    r"mlflow_artifacts\CM_08-06-2020__13_11_56.png"
]

artifact = r"mlflow_artifacts\CM_08-06-2020__13_11_47.png"

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 3
print("Initialised an MLFlow() object - return correct parameters."\
      "Log parameters and metrics in '/nlp-web-scrapping/mlruns',"\
      "string-object artifact saved in mlflow:\n"\
      "Parameters = 'Logistic Regression', 'Count Vectoriser' - "\
      "Metrics = 'accuracy': 0.753, 'precision': 0.853'\n"\
      "Artifact = "r"'mlflow_artifacts\CM_08-06-2020__13_11_47.png")
print()

mlf = MLFlow("Changed_Experiment",
             r"..\..\mlruns")

print()

print(mlf.experiment_name)

print()

print(mlf.tracking_uri)

mlf.MLFlow_Logging(params_dictionary = params_dict, metrics_dictionary = metrics_dict, artifact_filepaths = artifact)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 4
print("Initialised an MLFlow() object - return correct parameters."\
      "Log parameters and metrics in '/nlp-webscrapping/mlruns' folder, list-object artifact, only one saved in mlflow:\n"\
      "experiment_name = 'Test_Experiment'\n"\
      "tracking_uri = "r"'C:\Users\nathi_000\Desktop\Python Files\NLP Project\nlp-web-scrapping'\n"\
      "Parameters = 'Logistic Regression', 'Count Vectoriser' - "\
      "Metrics = 'accuracy': 0.753, 'precision': 0.853\n"\
      "Artifact = "r"'mlflow_artifacts\CM_08-06-2020__13_11_56.png")
print()

mlf.tracking_uri = r"..\..\mlruns"
mlf.experiment_name = "Test_Experiment"

print()

print(mlf.experiment_name)

print()

print(mlf.tracking_uri)

mlf.MLFlow_Logging(params_dictionary = params_dict, metrics_dictionary = metrics_dict, artifact_filepaths = artifact_list_broken)
