import nltk
import string
from spacy.lang.en.stop_words import STOP_WORDS
import re

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