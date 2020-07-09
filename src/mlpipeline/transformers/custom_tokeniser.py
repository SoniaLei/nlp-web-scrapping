from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem import WordNetLemmatizer
import string
from spacy.lang.en.stop_words import STOP_WORDS
import re

class Custom_tokeniser(BaseEstimator, TransformerMixin):
    
    def __init__(self, to_lower=True, remove_stopwords=True):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
    #def text_cleaner(sentence):
        
        Lemmatiser = WordNetLemmatizer()
        # Instantiating the NLTK Lemmatiser

        punctuations = string.punctuation
        # Putting punctuation symbols into an object

        stopwords = STOP_WORDS
        # A list of stopwords that can be filtered out
        
        for tweet in X:

            tweet = "".join([char for char in tweet.strip() if char not in punctuations])
            # Getting rid of any punctuation characters

            myTokens = re.split(r'\W+', sentence)
            # Tokenising the words

            myTokens = [token.lower() for token in myTokens if token not in stopwords]
            # Removing stop words and putting words into lowercase

            myTokens = [Lemmatiser.lemmatize(token) for token in myTokens]
            # Lemmatising the words

        return myTokens