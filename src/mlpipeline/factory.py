from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from .transformers import *
from .estimators import *


class ObjectFactory:
    """Factory class returning object instance of type object_type"""

    _registry = {
        'StopWordsRemoval': StopWordsRemoval,
        'TweetCleaner': TweetCleaner,
        'Lematization': Lematization,
        'CountVectorizer': CountVectorizer,
        'TfidfVectorizer': TfidfVectorizer,
        'Word2VecVectorizer': Word2VecVectorizer,
        'SVC': SVC,
        'LogisticRegression': LogisticRegression,
        'Tokenizer': Tokenizer,
        'TweetVectorizer': TweetVectorizer,
    }

    @classmethod
    def create_object(self, class_name, kwargs=None):
        """Creates an instance of an estimator.
        """
        kwargs = kwargs or {}
        obj = ObjectFactory._registry.get(class_name, None)
        if obj is None:
            raise KeyError(f"Object {class_name} is not found in "
                           f"registry. Available options are:"
                           f"{list(ObjectFactory._registry.keys())}")
        return obj(**kwargs)
