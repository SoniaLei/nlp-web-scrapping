"""
Module components responsible to set ml components objects used in nlp pipeline.
Those are Data, Transformers, Vectorizers and Estimators.
"""
import ast
import pandas as pd
from collections import ChainMap
from .factory import ObjectFactory


class Data:
    """
    Data class responsible to set data properties with right format to be digested by sklearn pipeline.
    """
    def __init__(self, *, train, test, features, target):
        """
        Parses train test filenames.cvs into `pd.DataFrame`.
        Slices dataframes and sets train_x, train_y, test_x, test_y.
        """
        self._train = pd.read_csv(train)
        self._test = pd.read_csv(test)
        self._train.dropna(inplace=True)
        self._test.dropna(inplace=True)
        # slicing for testing purposes
        self.train_X = self._train[features][:3000]
        self.train_Y = self._train[target][:3000]
        self.test_X = self._test[features][3000:5000]
        self.test_Y = self._test[target][3000:5000]


class CompositeConstructor:
    """
    CompositeConstructor responsible to set names, sequences of steps and parameters
    for transformers, vectorizers and estimators. Ready to be digested by GridSearchCv.
    """
    def __init__(self, **kwargs):
        """
        Initiates class CompositeConstructor and loops through
        specific config file layout, setting parameters for each key, value pair.
        """
        if not isinstance(kwargs, dict):
            raise TypeError(f"{type(self).__name__} must be a `dict` object"
                            f"found {type(kwargs)} instead.")

        self.names = []
        self.sequence = []
        self.parameters = []

        for name, params in kwargs.items():
            self.names.append(name)
            self.sequence.append((name, ObjectFactory.create_object(name)))
            self.parameters.append(self.get_params(name, params))

    def get_params(self, name, params):
        """
        Parses parameters names to follow GridSearch convention,
        parses values to original data type and encapsulates into a [],
        also for GridSearch usability.
        """
        params = params or {}

        if not isinstance(params, dict):
            raise TypeError(f"key {name} value must be a dictionary"
                            f"found {type(params)} instead.")
        if isinstance(self, Estimators):
            params = params.get('params', {})

        params = CompositeConstructor.rename_parameters_keys(name, params)
        params = CompositeConstructor.parse_parameters_values(params)
        return params

    @staticmethod
    def rename_parameters_keys(name, params_dic, sep='__'):
        """
        Renames parameters to contain the name of the transformer,
        vectorizer, or estimator. Adds '__' separator.
        """
        params_dic = params_dic or {}
        params = {name + sep + k: v for k, v in params_dic.items()}
        return params

    @staticmethod
    def parse_value(value):
        """
        Parses value type that json left as `str`.
        """
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        return value

    @staticmethod
    def parse_values(value):
        """
        Ensures if a list is passed parses all values inside the list.
        If not parse the value and returns a list containing the value.
        """
        if isinstance(value, list):
            value = [CompositeConstructor.parse_value(val) for val in value]

        else:
            value = CompositeConstructor.parse_value(value)
            # parsing all parameters to list
            # so that can be injected in GridSearch
            value = [value]

        return value

    @staticmethod
    def parse_parameters_values(parameters_dicts):
        """
        Loops through each parameters dict and parses the values
        found in the conf file.
        """
        return {k: CompositeConstructor.parse_values(v)
                for k, v in parameters_dicts.items()}


class Transformers(CompositeConstructor):
    """
    Transformers to be applied to all experiments in the config file.
    """
    def __init__(self, **kwargs):
        if not kwargs:
            self.names = ''
            self.sequence = []
            self.parameters = {}
        else:
            super().__init__(**kwargs)
            # Transformers is one single step for all pipelines
            # join transformers names to have only one name describing trans.
            self.names = "_".join(self.names)
            # same with parameters have one single dict.
            self.parameters = dict(ChainMap(*self.parameters))


class Vectorizers(CompositeConstructor):
    """
    Vectorizers types iterate from for each experiment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Estimators(CompositeConstructor):
    """
    Estimators to iterate from for each experiment,
    with cross validation for each of them.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cvs = [{sk: sv for sk, sv in v.items() if 'cv' in sk} for k, v in kwargs.items()]
