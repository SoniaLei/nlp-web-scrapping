from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


class BasicMetrics:

    def __init__(self, exp_name, classes, test_Y, results, output_path):
        self._exp_name = exp_name
        self._classes = classes
        self.test_Y = test_Y
        self.prediction_labels = results
        self.prediction_probabilities = results
        self.output_path = output_path
        self.file_extension = 'csv'

    @property
    def test_Y(self):
        return self._test_Y

    @test_Y.setter
    def test_Y(self, value):
        if not isinstance(value, pd.Series) and not \
                isinstance(value, pd.DataFrame):
            raise ValueError(f"test_Y must be a `pd.Series` or `pd.DataFrame`"
                             f" object found {type(value)} instead.")

        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        self._test_Y = value

    def validate_results(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Results from predictions must be of type `np.ndarray`.")

        if not data.dtype == np.float \
                and not data.dtype == np.object:
            raise ValueError("Predictions must be a `np.ndarray` dtype float or str object.")

        return True

    @property
    def prediction_labels(self):
        return self._prediction_labels

    @prediction_labels.setter
    def prediction_labels(self, data):
        """sets the prediction labels attributes"""
        if self.validate_results(data) and data.shape[1] > 1 and data.dtype == np.float:
            pred_class_nums = np.argmax(data, axis=1)
            class_names = list(enumerate(self._classes))
            prediction_class_names = list(map(lambda x: class_names[x][1], pred_class_nums))

        else:
            prediction_class_names = data

        self._prediction_labels = prediction_class_names

    @property
    def prediction_probabilities(self):
        return self._prediction_probabilities

    @prediction_probabilities.setter
    def prediction_probabilities(self, data):
        if self.validate_results(data) and data.shape[1] > 1 and data.dtype == np.float:
            self._prediction_probabilities = data
        else:
            self._prediction_probabilities = None


class Metrics(BasicMetrics):

    def __init__(self, exp_name, classes, test_Y, results, output_path='../data/results/'):
        super().__init__(exp_name, classes, test_Y, results, output_path=output_path)
        self._fpr = dict()
        self._tpr = dict()
        self._roc_auc = dict()

    @property
    def summary_report(self):
        return NotImplemented

    def f1_score(self, average='weighted'):
        return f1_score(self.test_Y, self.prediction_labels, average=average)

    @property
    def accuracy_score(self):
        return accuracy_score(self.test_Y, self.prediction_labels)

    @property
    def classification_report(self):
        return classification_report(self.test_Y, self.prediction_labels)

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.test_Y, self.prediction_labels)

    def dump_results_csv(self):
        data = pd.DataFrame({'test_Y': self.test_Y, 'Y_hat': self.prediction_labels})
        data.to_csv(self.output_path + self._exp_name + self.file_extension)

    @staticmethod
    def get_binary_labels(labels):
        lb = LabelBinarizer()
        return lb.fit_transform(labels)

    def calculate_micro_rates(self):
        if self.prediction_probabilities is None:
            raise Exception("Could not find probabilities for model results")

        binary_y_test = Metrics.get_binary_labels(self.test_Y)

        for i in range(len(self._classes)):
            self._fpr[i], self._tpr[i], _ = roc_curve(binary_y_test[:, i], self.prediction_probabilities[:, i])
            self._roc_auc[i] = auc(self._fpr[i], self._tpr[i])

        self._fpr["micro"], self._tpr["micro"], _ = roc_curve(binary_y_test.ravel(), self.prediction_probabilities.ravel())
        self._roc_auc["micro"] = auc(self._fpr["micro"], self._tpr["micro"])

    def plot_model_roc(self, color='darkorange', label='ROC curve (area = %0.2f)',
                       xlabel='False Positive Rate', ylabel='True Positive Rate',
                       title='Receiver operating characteristic', legend_loc="lower right",
                       lw=2, figsize=(20, 10)):

        if not self._fpr.get('micro') and not self._tpr.get('micro'):
            self.calculate_micro_rates()

        plt.figure(figsize=figsize)
        plt.plot(self._fpr[2], self._tpr[2], color=color,
                 lw=lw, label=label % self._roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        return plt

    def calculate_macro_rates(self):
        if not all([self._tpr.get(i) for i in range(len(self._classes))]):
            self.calculate_micro_rates()

        all_fpr = np.unique(np.concatenate([self._fpr[i] for i in range(len(self._classes))]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(self._classes)):
            mean_tpr += np.interp(all_fpr, self._fpr[i], self._tpr[i])
            mean_tpr /= len(self._classes)

        self._fpr["macro"] = all_fpr
        self._tpr["macro"] = mean_tpr
        self._roc_auc["macro"] = auc(self._fpr["macro"], self._tpr["macro"])

    def plot_classes_roc(self, color_micro_avg='deeppink', color_macro_avg='navy',
                         color_classes=('aqua', 'darkorange', 'cornflowerblue'),
                         xlabel='False Positive Rate', ylabel='True Positive Rate',
                         title='Receiver operating characteristic to multi-class',
                         legend_loc="lower right", lw=2, figsize=(20, 10)):

        self.calculate_macro_rates()
        d = list(enumerate(self._classes))

        plt.figure(figsize=figsize)
        plt.plot(self._fpr["micro"], self._tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(self._roc_auc["micro"]),
                 color=color_micro_avg, linestyle=':', linewidth=4)

        plt.plot(self._fpr["macro"], self._tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(self._roc_auc["macro"]),
                 color=color_macro_avg, linestyle=':', linewidth=4)

        colors = cycle(color_classes)
        for i, color in zip(range(len(self._classes)), colors):
            plt.plot(self._fpr[i], self._tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(d[i][1], self._roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        return plt
