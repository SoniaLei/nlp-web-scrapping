                        # Grid_Search Tests

# Imports

from .src.mlpipeline.Grid_Search import Grid_Search

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier    
from sklearn.pipeline import Pipeline
import NLP_Functions as Func

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Initialising variables

X_train, Y_train, X_test, Y_test = Func.readCSVs_URL(
    "https://raw.githubusercontent.com/SoniaLei/nlp-web-scrapping/development/data/raw/tweets-train.csv",
    "https://raw.githubusercontent.com/SoniaLei/nlp-web-scrapping/development/data/raw/tweets-test.csv"
)

bow_vector = CountVectorizer(tokenizer = Func.text_cleaner, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = Func.text_cleaner)

LogReg = LogisticRegression(max_iter=1000)
ovr = OneVsRestClassifier(LogReg)

rfc = RandomForestClassifier(n_estimators = 10, n_jobs=-1)

pipe_1 = Pipeline([('vectorizer', bow_vector),
                 ('classifier', ovr)])

pipe_2 = Pipeline([('vectorizer', None),
                 ('classifier', rfc)])

tuned_parameters_1 = [{'classifier__estimator__penalty': ['l2', 'none']}]

tuned_parameters_2 = [
    {'vectorizer': [bow_vector, tfidf_vector],
    'classifier': [rfc],
    'classifier__n_estimators': [10, 15, 20]}
]

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 1
print("TEST 1:\nInitialise a Grid_Search object - return correct parameters:\n" \
      "estimator = 'pipe_1'\n" \
      "parameters = 'tuned_parameters_1\n" \
      "cv = None\n" \
      "scoring= None" \
      "model = None")
print()

gs = Grid_Search(estimator=pipe_1, parameters=tuned_parameters_1)

print()

print(gs.estimator)

print()

print(gs.parameters)

print()

print(gs.cv)

print()

print(gs.model)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 2
print("TEST 2:\nInitialise a Grid_Search object - return correct parameters:\n" \
      "estimator = 'pipe_2'\n" \
      "parameters = 'tuned_parameters_2'\n" \
      "cv = '5'\n" \
      "scoring = 'accuracy'" \
      "model = None")
print()

gs = Grid_Search(estimator=pipe_2, parameters=tuned_parameters_2, cv=5, scoring="accuracy")

print()

print(gs.estimator)

print()

print(gs.parameters)

print()

print(gs.cv)

print()

print(gs.model)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 3
print("TEST 3:\n Using TEST 2's Grid_Search object - fit model using training_data and return correct model:\n" \
      "estimator = 'pipe_2'\n" \
      "parameters = 'tuned_parameters_2'\n" \
      "cv = '5'\n" \
      "scoring = 'accuracy'" \
      "model = '[Above hyper-parameters]'")
print()

gs.fit(X_train=X_train, Y_train=Y_train)

print(gs.model)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 4
print("TEST 4:\n Using TEST 3's fitted Grid_Search - return 5 split_scores from model:\n" \
      "estimator = 'pipe_2'\n" \
      "parameters = 'tuned_parameters_2'\n" \
      "cv = '5'\n" \
      "scoring = 'accuracy'" \
      "model = '[Above hyper-parameters]'")
print()

print("Split Scores:")

for eachScore in gs.split_scores():
    print(eachScore)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 5
print("TEST 5:\n Using TEST 3's fitted Grid_Search - return mean_scores and standard devitation for each permutation:\n" \
      "estimator = 'pipe_2'\n" \
      "parameters = 'tuned_parameters_2'\n" \
      "cv = '5'\n" \
      "scoring = 'accuracy'" \
      "model = '[Above hyper-parameters]'")
print()

print("Each permutation's score:")

for eachPerm in gs.permutations_score():
    print(eachPerm)

print("\n", "--" * 30, "\n")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# Test 6
print("TEST 6:\n Using TEST 3's fitted Grid_Search - return the best_parameters and their score for this model:\n" \
      "estimator = 'pipe_2'\n" \
      "parameters = 'tuned_parameters_2'\n" \
      "cv = '5'\n" \
      "scoring = 'accuracy'" \
      "model = '[Above hyper-parameters]'")
print()

params, score = gs.best_params()

print("Best Parameters:")

print("Params:\n", params)

print(str(gs.scoring) + " score: ", score)