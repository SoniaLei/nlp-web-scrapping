from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from .transformers import *
from .estimators import *


class ObjectFactory:
    """Factory class returning object instance of type object_type"""

    _registry = {
        'CountVectorizer': CountVectorizer,
        'TfidfVectorizer': TfidfVectorizer,
        'Word2VecVectorizer': Word2VecVectorizer,
        'SVC': SVC,
        'LogisticRegression': LogisticRegression,
        'Tokenizer': Tokenizer,
        'DummyTransformer': DummyTransformer,
    }

    @classmethod
    def create_object(self, class_name, kargs=None):
        """Creates an instance of an estimator.
        """

        if kargs is None:
            kargs = {}
            
        return ObjectFactory._registry[class_name](**kargs)
