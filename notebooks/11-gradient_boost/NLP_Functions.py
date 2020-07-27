                        # Imports
    
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

import mlflow
import mlflow.sklearn

import requests
import io

def abc(string):
    
    print(string)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
                        # Reading .csv file function

def readCSVs_URL(url_train, url_test):
    
    # Training csv file
    csv_train = requests.get(url_train).content
    df_train = pd.read_csv(io.StringIO(csv_train.decode('utf-8')))
    
    X_train = df_train['text'].astype(str)
    Y_train = df_train['sentiment'].astype(str)

    # Testing csv file
    csv_test = requests.get(url_test).content
    df_test = pd.read_csv(io.StringIO(csv_test.decode('utf-8')))  

    X_test = df_test['text'].astype(str)
    Y_test = df_test['sentiment'].astype(str)
    
    return X_train, Y_train, X_test, Y_test

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def text_cleaner(sentence):
    
    Lemmatiser = nltk.stem.WordNetLemmatizer()
    # Instantiating the NLTK Lemmatiser

    punctuations = string.punctuation
    # Putting punctuation symbols into an object

    stopwords = STOP_WORDS
    # A list of stopwords that can be filtered out
                    
    sentence = "".join([char for char in sentence.strip() if char not in punctuations])
    # Getting rid of any punctuation characters
    
    myTokens = re.split(r'\W+', sentence)
    # Tokenising the words
    
    myTokens = [token.lower() for token in myTokens if token not in stopwords]
    # Removing stop words
    
    myTokens = [Lemmatiser.lemmatize(token) for token in myTokens]
    # Lemmatising the words and putting in lower case except for proper nouns
    
    return myTokens

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                        # Create new file directory function
    
def Create_FileDirectory(filepath):
    
    if not os.path.exists(os.path.dirname(filepath)):
    # If the Filepath directories don't exist...
        
        try:
            os.makedirs(os.path.dirname(filepath))
            # Create the directories that don't exist
            
        except OSError as exc:
        # Guard against race condition
        
            if exc.errno != errno.EEXIST:                
                raise

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                        # Create an experiment / get experiment Id function
                
# Create an experiment to put data for relevant models in

def GetExper_CreateExper(Experiment_Name):
# Experiment_Name = STRING - name of experiment in mlruns folder, or create a new one
    
    try:
        exp_id = mlflow.get_experiment_by_name(Experiment_Name).experiment_id
        # Inputs:
            # name = the experiment's name
        # Returns: Experiment object
            # .value wanted from object
    
    except AttributeError: # Error if experiment already created
        exp_id = mlflow.create_experiment(Experiment_Name)
        # Inputs:
            # name = the experiment's name (unique)
            # artifact_location (optional) = location to store run artifacts
        # Returns: integer Id of the experiment
    
    return exp_id

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                        # Create Sentiment Analysis Model's name function
    
def SAM_Name(pipe):
    
    try:
        
        SAM = str(pipe.named_steps['classifier'])
    
    except AttributeError:
        
        SAM = str(pipe)
    
    match = re.search("\(estimator=", SAM)
    
    if match:
        
        classifier = SAM.split("(")[0].replace("{'classifier':","").strip()
        
        estimator = SAM.split("(")[1].replace("estimator=","").strip()
        
        SAM = estimator + " (" + classifier + ")"
        
    else:
        
        SAM = SAM.split("(")[0].replace("{'classifier':","").strip()
    
    return SAM

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                        # Pipeline function
    
def Create_Pipeline(vectoriser, classifier):

    pipe = Pipeline([('vectorizer', vectoriser)
                     ,('classifier', classifier)])

    return pipe

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

                        # Get names of varied aspects of the pipeline function

def Names_GridSearch(tuned_parameters, model):
    
    values_list = []
    
    parameters_list = [parameter for eachObject in tuned_parameters for parameter in eachObject.keys()]
    
    params_to_print = [parameter.split("__")[-1] for parameter in parameters_list]
    
    for parameter, param in zip(parameters_list, params_to_print):
        
        match = re.search("classifier", param)
        
        if match:
            
            value = str(param) + ": " + SAM_Name(str(model))
            
        else:
            
            value = str(param) + ": " + str(model[parameter]).split("(")[0].strip()
            
        values_list.append(value)
    
    best_params = ", ".join(value for value in values_list)
        
    return best_params