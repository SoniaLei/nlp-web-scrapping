"""Module data dealing with all data aspects"""
import pandas as pd

class ValidData:
    """
    Class responsible to validate datasets.
    """
    def __init__(self, type_ds, type_col):
        """Instantiates a new ValidData object"""
        self.type_ds = type_ds
        self.type_col = type_col

    def __set_name__(self, owner, name):
        """ValidData object name"""
        self.name = name

    def __set__(self, instance, value):
        """validate datasets and sets pd.series"""
        dataset, feature = value
        if not isinstance(dataset, self.type_ds):
            raise TypeError(f"{self.name} must be of type {self.type_ds}")
        if not isinstance(feature, self.type_col):
            raise TypeError(f"{feature} must be of type {self.type_col} found {type(feature)}.")
        if feature not in list(dataset.columns):
            raise ValueError(f"{feature} column not found in dataset for {self.name}.")
        dataset = instance.remove_nan(dataset)
        instance.__dict__[self.name] = dataset[feature]

    def __get__(self, instance, owner):
        """returns pd.series object"""
        if instance is None:
            return self
        else:
            return instance.__dict__.get(self.name, None)


class Data:
    """
    Data class responsible to set data properties with right format to be digested by sklearn pipeline.
    """
    train_X = ValidData(pd.DataFrame, str)
    train_Y = ValidData(pd.DataFrame, str)
    test_X = ValidData(pd.DataFrame, str)
    test_Y = ValidData(pd.DataFrame, str)

    def __init__(self, *, train, test, features, target):
        """Creates a new instance of type data"""
        self.train_X = (train, features)
        self.train_Y = (train, target)
        self.test_X = (test, features)
        self.test_Y = (test, target)

    def remove_nan(self, ds):
        """Remove nan values from dataset"""
        ds = ds.dropna()
        return ds
