from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score

class Metrics:
    def __init__(self, results):
        self.results = results
        self.score = None
        self.accuracy = None
        self._metrics_summary = None

    def get_summary_report(self):
        pass

    def get_score(self):
        pass

    def get_accuracy(self):
        pass

    def get_class_report(self):
        pass

    def get_confusion_matrix(self):
        pass

    def dump_metrics_results_csv(self):
        pass

