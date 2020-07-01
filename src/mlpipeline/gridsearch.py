# Imports
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import re

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

class GridSearch:
    
# Constructor
    
    def __init__(self, estimator, parameters, cv = None, scoring = None):
        self.estimator = estimator
        self.parameters = parameters
        self.cv = cv
        self.scoring = scoring
        self._model = None
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Getters & Setters

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# estimator
    # Object: object that will have its parameters looped through
    # It can be an SAM model or a pipeline containing a vectoriser, model etc.
    
    @property
    def estimator(self):
        print("Getting 'estimator' ...")
        return self._estimator
    
    @estimator.setter
    def estimator(self, estimator):
        print("Setting 'estimator' ...")
        
        if isinstance(estimator, Pipeline):
            self._estimator = estimator
            
        else:            
            raise TypeError("Input needs to be a Classifier or Pipeline object")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# parameters
    # Dictionary: parameters to be looped through in the Grid Search
    
    @property
    def parameters(self):
        print("Getting 'parameters' ...")
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters):
        print("Setting 'parameters' ...")
        
        if isinstance(parameters, list):
            self._parameters = parameters
            
        else:
            raise TypeError("Input needs to be a List")
            
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# cv (cross validation)
    # Integer: number of different datasets the whole set will be split into
    
    @property
    def cv(self):
        print("Getting 'cv' ...")
        return self._cv
    
    @cv.setter
    def cv(self, cv):
        print("Setting 'cv' ...")
        
        if isinstance(cv, int) or cv == None:
            self._cv = cv
            
        else:
            raise TypeError("Input needs to be an Integer")
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# model
    # Grid_Search_CV() model: model generated from Grid_Search_CV module
    
    @property
    def model(self):
        print("Getting 'model' ...")
        return self._model
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# scoring
    # String: scoring system for the Grid Search
    
    @property
    def scoring(self):
        print("Getting 'scoring' ...")
        return self._scoring
    
    @scoring.setter
    def scoring(self, scoring):
        print("Setting 'scoring' ...")
        
        if isinstance(scoring, str) or scoring == None:
            self._scoring = scoring
            
        else:
            raise TypeError("Input needs to be a String")
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Functions

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Shows all the scoring strings that can be entered into Grid Search
    
    def scoring_keys(self):
        
        print(sklearn.metrics.SCORERS.keys())
        
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Shortens the name of permutations - easier to read
        
    def Names_GridSearch(self, tuned_parameters={}): #, model):
        
        values_list = []

        parameters_list = [parameter for parameter in tuned_parameters.keys()]
        
        params_to_print = [parameter.split("__")[-1] for parameter in parameters_list]

        for parameter, param in zip(parameters_list, params_to_print):

            match = re.search("classifier", param)
            
            value = str(param) + ": " + str(tuned_parameters[parameter]).split("(")[0].strip()

            values_list.append(value)

        best_params = ", ".join(value for value in values_list)

        return best_params

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
    def fit(self, X_train, Y_train):
    # Estimator is 'fitted' with the trained data:
        """
        Inputs:
            X_train - DataFrame
            Y_train - DataFrame
        Outputs:
            model set to self.model
        """
        
        model = GridSearchCV(
            estimator = self.estimator, param_grid = self.parameters,
            scoring=self.scoring, cv=self.cv, n_jobs=-1
        )
        
        print("Fitting Grid_Search model ...")
        
        model.fit(X_train, Y_train)
        
        print("Setting 'model' ...")
        
        self._model = model
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
    
    def split_scores(self):
    # Scores for each dataset from the cross-validation
        """
        Outputs:
            List - List of the split scores
        """
        
        split_scores_list = []
        
        k = self.cv
        n = k
        
        model = self.model
        
        if n != None:
            while n > 0:
                
                score_number = str(k - n)
                
                split_score = str("split" + score_number + "_test_score")
                
                split_scores_list.append(model.cv_results_[split_score])
                
                n -= 1
                
        else:
            print("No cross-validation scores")
            
        return split_scores_list
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def best_params(self):
    # The parameters with the best result for the scoring metric
        """
        Outputs:
            Object - Object containing the best parameters
            Float - score of the best parameters                        
        """        
        model = self.model
        
        best_params = model.best_params_
        
        best_score = model.best_score_
        
        return best_params, best_score
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    def permutations_score(self):
    # Each permutations score, with their standard deviation
        """
        Outputs:
            List - List of the score, std and parameters used
        """
        
        model = self.model
        parameters = self.parameters
        
        scores = []
        
        means = model.cv_results_['mean_test_score']
        stds = model.cv_results_['std_test_score']
        params = model.cv_results_['params']
        
        for mean, std, param in zip(means, stds, params):
            
            score = str("%0.3f (+/-%0.03f) for %r" %
                        (mean, std * 2,
                         self.Names_GridSearch(tuned_parameters=param)))
                        
            scores.append(score)
            
        return scores