from src.data import Data
import pytest
import pandas as pd


def test_data_ok():
    d = Data(train=pd.DataFrame({'feat1': [1, 2, 3], 'target1': [10, 20, 30]}),
             test=pd.DataFrame({'feat1': [100, 200, 300], 'target1': [1000, 2000, 3000]}),
             features="feat1",
             target="target1")
    assert isinstance(d, Data)
    assert isinstance(d.train_X, pd.Series)
    assert isinstance(d.train_Y, pd.Series)
    assert isinstance(d.test_X, pd.Series)
    assert isinstance(d.test_Y, pd.Series)
    assert list(d.train_X) == [1, 2, 3]
    assert list(d.train_Y) == [10, 20, 30]
    assert list(d.test_X) == [100, 200, 300]
    assert list(d.test_Y) == [1000, 2000, 3000]


def test_data_instance():
    with pytest.raises(TypeError) as ex:
        Data(train='string', test='string', target="None", features="None")
    assert "train_X must be of type <class 'pandas.core.frame.DataFrame'>" == str(ex.value)


def test_train_data_instance():
    with pytest.raises(ValueError) as ex:
        Data(train=pd.DataFrame(), test='string', target="None", features="None")
    assert "None column not found in dataset for train_X." == str(ex.value)


def test_data_features_not_in_dfs():
    with pytest.raises(ValueError) as ex:
        Data(train=pd.DataFrame(), test=pd.DataFrame(), target="target_name", features="feat_name")
    assert "feat_name column not found in dataset for train_X." == str(ex.value)


def test_data_features_not_in_test_df():
    with pytest.raises(ValueError) as ex:
        Data(train=pd.DataFrame({'feat1': [1, 2, 3]}),
             test=pd.DataFrame(),
             target="target_name",
             features="feat1")
    assert "target_name column not found in dataset for train_Y." in str(ex.value)


def test_data_target_not_in_test_df():
    with pytest.raises(ValueError) as ex:
        Data(train=pd.DataFrame({'feat1': [1, 2, 3]}),
             test=pd.DataFrame({'feat1': [1, 2, 3]}),
             target="target_name",
             features="feat1")
    assert "target_name column not found in dataset for train_Y." in str(ex.value)
