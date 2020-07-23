"""
Factory class to register and instantiate objects.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from .transformers import *


class ObjectFactory:
    """Factory class returning object instance of type object_type"""

    _registry = {
        # Transformers
        'Tokenizer': Tokenizer,
        'CountVectorizer': CountVectorizer,
        'TfidfVectorizer': TfidfVectorizer,
        'Word2VecVectorizer': Word2VecVectorizer,

        # Estimators classifiers
        'SVC': SVC,
        'MultinomialNB': MultinomialNB,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
    }

    @classmethod
    def create_object(self, class_name, kwargs=None):
        """
        Returns an instance of the Object if present in registry.
        """
        kwargs = kwargs or {}
        obj = ObjectFactory._registry.get(class_name, None)
        if obj is None:
            raise KeyError(f"Object {class_name} is not found in "
                           f"registry. Available options are:"
                           f"{list(ObjectFactory._registry.keys())}")
        return obj(**kwargs)
