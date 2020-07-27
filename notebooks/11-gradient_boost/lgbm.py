from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from NLP_Functions import text_cleaner, readCSVs_URL

X_train, Y_train, X_test, Y_test = readCSVs_URL(
    url_train="https://raw.githubusercontent.com/SoniaLei/nlp-web-scrapping/development/data/raw/tweets-train.csv",
    url_test="https://raw.githubusercontent.com/SoniaLei/nlp-web-scrapping/development/data/raw/tweets-test.csv"
)

encoder_lb = LabelEncoder()

Y_train = encoder_lb.fit_transform(Y_train)
Y_test = encoder_lb.fit_transform(Y_test)

tfidf_vector = TfidfVectorizer(tokenizer = text_cleaner)

lgbm = LGBMClassifier()

pipe = Pipeline([("vectorizer", tfidf_vector),
                ("classifier", lgbm)])

pipe.fit(X_train, Y_train)

predicted = pipe.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, predicted))
print("Precision:",metrics.precision_score(Y_test, predicted, average="macro"))
print("Recall:",metrics.recall_score(Y_test, predicted, average="macro"))