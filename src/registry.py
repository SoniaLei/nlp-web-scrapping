from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import StopWordsRemoval, Lematization, TweetCleaner

vectorizers_registry = {'CountVectorizer': CountVectorizer,
                        'TfidfVectorizer': TfidfVectorizer}

models_registry = {'LogisticRegression': LogisticRegression,
                   'svm': SVC,
                   'bayes': MultinomialNB}

transformers_registry = {'stopwords': StopWordsRemoval,
                         'lematization': Lematization,
                         'TweetClassifier': TweetCleaner}

metrics_registry = {'accuracy_score': accuracy_score,
                    'classification_report': classification_report,
                    'confusion_matrix': confusion_matrix,
                    'r2_score': r2_score}

