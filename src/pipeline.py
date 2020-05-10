from sklearn.pipeline import Pipeline
from src.tweet_cleaner import TweetCleaner
from src.tweet_vectorizer import TweetVectorizer
from src.tweet_classifier import TweetClassifier


classifiers = [

]

pipeline = Pipeline(steps=[
    ('cleaner', TweetCleaner()),
    ('vectorizer', TweetVectorizer()),
    ('classifier', TweetClassifier())])

print('Fitting train data to labels...')
pipeline.fit(None, None)

print('\nMaking predictions...')
y_pred = pipeline.predict(None)

