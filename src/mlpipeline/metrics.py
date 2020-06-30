"""Module Metrics containing functionality to gauge the effectively/performance of an ml classifier model."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


class BasicMetrics:
    """
    BasicMetrics class setting basic metrics parameters used
    in many sklearn metrics functions.
    """
    def __init__(self, exp_name, classes, test_Y, results, output_path):
        """
        Initializes a BasicMetrics class.
        :param exp_name: `str` experiment name.
        :param classes: `list` of unique strings found inside target np.ndarray.
        :param test_Y: `list` of real/actual y labels from the test split.
        :param results: `list` of predicted class/label for the test split.
        :param output_path: `str` path to save.
        """
        self._exp_name = str(exp_name)
        self._classes = classes
        self.test_Y = test_Y
        self.prediction_labels = results
        self.prediction_probabilities = results
        self.output_path = output_path
        self.file_extension = 'csv'

    @property
    def test_Y(self):
        """
        Gets real/actual labels for test split
        """
        return self._test_Y

    @test_Y.setter
    def test_Y(self, value):
        """
        Sets real/actual labels for test split
        """
        if not isinstance(value, pd.Series) and not \
                isinstance(value, pd.DataFrame):
            raise ValueError(f"test_Y must be a `pd.Series` or `pd.DataFrame`"
                             f" object found {type(value)} instead.")

        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        self._test_Y = value

    def validate_results(self, data):
        """
        Validates the data being passed in is of type `np.ndarray` and dtype float or object.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Results from predictions must be of type `np.ndarray`.")

        if not data.dtype == np.float \
                and not data.dtype == np.object:
            raise ValueError(f"Predictions must be a `np.ndarray` dtype float or str object got {data.dtype} instead.")

        return True

    @property
    def prediction_labels(self):
        """
        Gets the predictions outputted from the ml model.
        """
        return self._prediction_labels

    @prediction_labels.setter
    def prediction_labels(self, data):
        """
        Sets the predictions outputted from the ml model of string type.
        """
        if self.validate_results(data) and len(data.shape) > 1 and data.dtype == np.float:
            pred_class_nums = np.argmax(data, axis=1)
            class_names = list(enumerate(self._classes))
            prediction_class_names = list(map(lambda x: class_names[x][1], pred_class_nums))

        else:
            prediction_class_names = data

        self._prediction_labels = prediction_class_names

    @property
    def prediction_probabilities(self):
        """
        Gets the probabilities predictions for each of the existent classes outputted from ml model.
        If results are `np.ndarray` of list of string values/ classes probabilities is None.
        """
        return self._prediction_probabilities

    @prediction_probabilities.setter
    def prediction_probabilities(self, data):
        """
        Sets the probabilities predictions for each of the existent classes outputted from ml model.
        If results are `np.ndarray` of list of string values/ classes probabilities is None.
        """
        if self.validate_results(data) and len(data.shape) > 1 and data.dtype == np.float:
            self._prediction_probabilities = data
        else:
            self._prediction_probabilities = None


class Metrics(BasicMetrics):
    """
    Metrics class plugin for sklearn metrics functions.
    """

    def __init__(self, exp_name, classes, test_Y, results, output_path='../data/results/',
                 fpr=None, tpr=None, roc_auc=None):
        """
        Initialized a Metrics class used to get metrics results from sklearn metrics class.
        :param exp_name: `str` experiment name.
        :param classes: `list` of unique strings found inside target np.ndarray.
        :param test_Y: `list` of real/actual y labels from the test split.
        :param results: `list` of predicted class/label for the test split.
        :param output_path: `str` path to save.
        :param fpr: false positives rates used for ROC plot.
        :param tpr: true positives rates used for ROC plot.
        :param roc_auc: Roc auc values used for ROC plot labels.
        """
        super().__init__(exp_name, classes, test_Y, results, output_path=output_path)
        self._fpr = fpr or dict()
        self._tpr = tpr or dict()
        self._roc_auc = roc_auc or dict()

    @property
    def summary_report(self):
        # TODO generate text file docs with all metrics.
        return NotImplemented

    def f1_score(self, average='weighted'):
        """
        F1 score metrics plugin from sklearn.
        """
        return f1_score(self.test_Y, self.prediction_labels, average=average)

    @property
    def accuracy_score(self):
        """
        Accuracy score plugin from sklearn.
        """
        return accuracy_score(self.test_Y, self.prediction_labels)

    @property
    def classification_report(self):
        """
        Classification report plugin from sklearn.
        """
        return classification_report(self.test_Y, self.prediction_labels)

    @property
    def confusion_matrix(self):
        """
        Confusion matrix plugin from sklearn.
        """
        return confusion_matrix(self.test_Y, self.prediction_labels)
    
    def plot_confusion_matrix(self, figuresize=(8, 8), labels='', title='Confusion Matrix of the Classifier\n'):
        """
        Returns confusion matrix plot object.
        """
        cm = self.confusion_matrix

        fig = plt.figure(figsize=figuresize)

        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)

        plt.title(title)
        fig.colorbar(cax)

        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)

        plt.xlabel('Predicted')
        plt.ylabel('True')

        thresh = cm.max() / 2

        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(cm[i, j],'d'), horizontalalignment='center', color='white' if cm[i,j] < thresh else 'black', fontsize = 26)

        return plt

    def dump_results_csv(self):
        """
        Dumps results (Predictions and real Y_test values) in a csv file.
        """
        data = pd.DataFrame({'test_Y': self.test_Y, 'Y_hat': self.prediction_labels})
        data.to_csv(self.output_path + self._exp_name + self.file_extension)

    @staticmethod
    def get_binary_labels(labels):
        """
        Binarises "encodes" string list predictions into n dimentional categories.
        """
        lb = LabelBinarizer()
        return lb.fit_transform(labels)

    def _calculate_micro_rates(self):
        """
        Calculates micro rates used in ROC plot.
        """
        if self.prediction_probabilities is None:
            raise Exception("Could not find probabilities for model results make sure when "
                            "running the experiment probabilities param is set to True.")

        binary_y_test = Metrics.get_binary_labels(self.test_Y)

        for i in range(len(self._classes)):
            self._fpr[i], self._tpr[i], _ = roc_curve(binary_y_test[:, i], self.prediction_probabilities[:, i])
            self._roc_auc[i] = auc(self._fpr[i], self._tpr[i])

        self._fpr["micro"], self._tpr["micro"], _ = roc_curve(binary_y_test.ravel(), self.prediction_probabilities.ravel())
        self._roc_auc["micro"] = auc(self._fpr["micro"], self._tpr["micro"])

    def _calculate_macro_rates(self):
        """
        Calculates macro rates used in ROC plot.
        """
        if not all([self._tpr.get(i) for i in range(len(self._classes))]):
            self._calculate_micro_rates()

        all_fpr = np.unique(np.concatenate([self._fpr[i] for i in range(len(self._classes))]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(self._classes)):
            mean_tpr += np.interp(all_fpr, self._fpr[i], self._tpr[i])
            mean_tpr /= len(self._classes)

        self._fpr["macro"] = all_fpr
        self._tpr["macro"] = mean_tpr
        self._roc_auc["macro"] = auc(self._fpr["macro"], self._tpr["macro"])

    def plot_model_roc(self, linecolor='darkorange', label='ROC curve (area = %0.2f)',
                       xlabel='False Positive Rate', ylabel='True Positive Rate',
                       title='Receiver operating characteristic', legend_loc="lower right",
                       lw=2, figsize=(20, 10)):
        """
        Plots one single line for all true and false classes.
        :param color: `str` line color.
        :param label: `str` appearing as a plot label. %0.2f precision for area score value
        :param xlabel: `str` label for the x axes.
        :param ylabel: `str` label for the y axes.
        :param title: `str` title for the plot.
        :param legend_loc: `str` legged position inside the plot.
        :param lw: `int` line width.
        :param figsize: `tuple` width length plot size in px.
        :return: `plt` plot figure.
        """
        if not self._fpr.get('micro') and not self._tpr.get('micro'):
            self._calculate_micro_rates()

        plt.figure(figsize=figsize)
        plt.plot(self._fpr[2], self._tpr[2], color=linecolor,
                 lw=lw, label=label % self._roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        return plt

    def plot_classes_roc(self, color_micro_avg='deeppink', color_macro_avg='navy',
                         color_classes=('aqua', 'darkorange', 'cornflowerblue'),
                         xlabel='False Positive Rate', ylabel='True Positive Rate',
                         title='Receiver operating characteristic to multi-class',
                         legend_loc="lower right", lw=2, figsize=(20, 10)):
        """

        :param color_micro_avg: `str` line plot color for micro average of all classes results.
        :param color_macro_avg: `str` line plot color for macro average of all classes results.
        :param color_classes: `list` of colors for each of the classes lines in the plot.
        :param xlabel: `str` label for the x axes.
        :param ylabel: `str` label for the y axes.
        :param title: `str` title for the plot.
        :param legend_loc: `str` legged position inside the plot.
        :param lw: `int` line width.
        :param figsize: `tuple` width length plot size in px.
        :return: `plt` plot figure.
        """
        self._calculate_macro_rates()
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

        colors = itertools.cycle(color_classes)
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
