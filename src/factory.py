from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from src.transformers import StopWordsRemoval, Lematization, TweetCleaner
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class ObjectFactory:
    """Factory class returning object instance of type object_type"""

    @classmethod
    def create_object(self, obj_type, kargs=None):
        if not kargs:
            kargs = {}
        obj = globals()[obj_type](**kargs)
        return obj

