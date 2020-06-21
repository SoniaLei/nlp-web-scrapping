from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
import pandas as pd

class Metrics:
    def __init__(self, exp_name, test_Y, predictions, output_path='../data/results/'):
        self.exp_name = exp_name
        self.test_Y = test_Y
        self.predictions = predictions
        self.output_path = output_path
        self.file_extension = 'csv'

    @property
    def summary_report(self):
        return NotImplemented

    @property
    def accuracy_score(self):
        return accuracy_score(self.test_Y, self.predictions)

    @property
    def classification_report(self):
        return classification_report(self.test_Y, self.predictions)

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.test_Y, self.predictions)

    def dump_results_csv(self):
        data = pd.DataFrame({'test_Y': self.test_Y, 'Y_hat': self.predictions})
        data.to_csv(self.output_path + self.exp_name + self.file_extension)


