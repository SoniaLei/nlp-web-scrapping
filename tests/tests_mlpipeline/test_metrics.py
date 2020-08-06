"""
Test for Metrics class
command line: python -m pytest test/unit/test_context.py
"""
import numpy as np
import pandas as pd
import pytest
from mlpipeline.metrics import Metrics
from unittest import mock


@pytest.fixture
def metrics_values():
    return {
        'exp_name': 'exp_name_test',
        'classes': ['possitive', 'neutral'],
        'test_Y': pd.Series(['possitive', 'neutral', 'possitive']),
        'results': np.array(['possitive', 'neutral', 'possitive'], dtype=np.object),
        'output_path': 'path/to/folder'
    }


@pytest.fixture
def metrics(metrics_values):
    return Metrics(**metrics_values)


def test_create_basic_metrics_with_labels(metrics_values, metrics):
    assert metrics._exp_name == metrics_values['exp_name']
    assert metrics._classes == metrics_values['classes']
    assert list(metrics.test_Y) == list(metrics_values['test_Y'])
    assert list(metrics.prediction_labels) == list(metrics_values['results'])
    assert metrics.prediction_probabilities is None
    assert metrics.output_path == metrics_values['output_path']


@pytest.mark.parametrize('results',  [np.array([[0.85, 0.2],
                                                [0.20, 0.89],
                                                [0.98, 0.10]], dtype=np.float)])
def test_create_basic_metrics_with_probabilities(metrics_values, results):
    metrics_values['results'] = results
    metrics = Metrics(**metrics_values)
    assert list(metrics.prediction_labels) == ['possitive', 'neutral', 'possitive']
    assert isinstance(metrics.prediction_probabilities, np.ndarray)
    assert metrics.prediction_probabilities.tolist() == results.tolist()


def test_types_basic_metrics(metrics):
    assert isinstance(metrics._exp_name, str)
    assert isinstance(metrics._classes, list)
    assert isinstance(metrics.test_Y, pd.Series)
    assert isinstance(metrics.prediction_labels, np.ndarray)
    assert isinstance(metrics.output_path, str)


@pytest.mark.parametrize('test_Y, results',
                          [(['possitive', 'neutral', 'possitive'], np.array(['possitive', 'neutral', 'possitive'])),
                            (pd.DataFrame({0: ['possitive', 'neutral', 'possitive']}), ['possitive', 'neutral', 'possitive'])])
def test_create_invalid_test_y(test_Y, results):
    with pytest.raises(ValueError):
        Metrics('exp_name', 'classses', test_Y, results)


def test_metrics_plots_params(metrics):
    assert isinstance(metrics._fpr, dict)
    assert isinstance(metrics._tpr, dict)
    assert isinstance(metrics._roc_auc, dict)


def test_f1_score(metrics):
    assert isinstance(metrics.f1_score(), float)
    assert metrics.f1_score() == 1.0


@pytest.mark.parametrize('test_Y, results, score',
                          [(pd.Series(['possitive', 'neutral', 'possitive']), np.array(['possitive', 'neutral', 'possitive'], dtype=np.object), 1.0),
                           (pd.Series(['possitive', 'neutral', 'possitive']), np.array(['neutral', 'neutral', 'possitive'], dtype=np.object), 0.6666666666666666666),
                           (pd.Series(['possitive', 'neutral', 'possitive']), np.array(['neutral', 'possitive', 'possitive'], dtype=np.object), 0.33333333333333333),
                           (pd.Series(['possitive', 'neutral', 'possitive']), np.array(['neutral', 'possitive', 'neutral'], dtype=np.object), 0.0),
                           ])
def test_accuracy_score(metrics_values, test_Y, results, score):
    metrics_values['test_Y'] = test_Y
    metrics_values['results'] = results
    metrics = Metrics(**metrics_values)
    assert isinstance(metrics.accuracy_score, float)
    assert metrics.accuracy_score == score


def test_plots_error(metrics):
    with pytest.raises(Exception) as ex:
        metrics.plot_model_roc()
        assert "Could not find probabilities for model results" == str(ex.value)


@mock.patch("mlpipeline.metrics.plt")
def test_plot(mock_plt, metrics_values):
    metrics_values['fpr'] = {"micro": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                             2: 0.558}
    metrics_values['tpr'] = {"micro": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                             2: 0.98}
    metrics_values['roc_auc'] = {"micro": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                                 2: 0.68}
    metrics_values['results'] = np.array([[0.85, 0.2],[0.20, 0.89],[0.98, 0.10]], dtype=np.float)
    metrics = Metrics(**metrics_values)
    metrics.plot_model_roc()
    assert mock_plt.figure.called
    assert mock_plt.plot.called
    mock_plt.title.assert_called_once_with("Receiver operating characteristic")
    mock_plt.xlabel.assert_called_once_with('False Positive Rate')
    mock_plt.ylabel.assert_called_once_with('True Positive Rate')
    mock_plt.legend.assert_called_once_with(**{'loc': "lower right"})


