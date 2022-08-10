import numpy as np
from sklearn.metrics import classification_report
import keras

r1_ratio = 7.5  
const = 0.86666667  #magic number for our metric ;) 

def new_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Metric which you can't change! We calculate there propotion of false positives to false negatives.
    """
    y_pred = np.rint(y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    if "False" in report:
        report["0"] = report.pop("False")
    if "True" in report:
        report["1"] = report.pop("True")
    r0 = report["0"]
    try:
        r1 = report["1"]
    except KeyError:
        return 0.0
    try:
        divider = const + (1 / (r1_ratio * r1["recall"])) + (1 / (r0["recall"]))
    except ZeroDivisionError:
        divider = (
            const
            + (1 / (r1_ratio * r1["recall"] + np.nextafter(0, 1)))
            + (1 / (r0["recall"] + np.nextafter(0, 1)))
        )

    return (r1["recall"] + r0["recall"]) / divider


class NewMetricCallback(keras.callbacks.Callback):
    """
    Child method to input our metric in keras API. 
    It is necessary for optkeras package to work properly.
    """
    def __init__(self, val_data: tuple) -> None:
        super().__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.metric = 0
        self.new_metric = []

    def on_epoch_end(self, epoch, logs={}):
        model = self.model
        preds = np.asarray(model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.metric = new_metric(targ, preds)
        self.new_metric.append(self.metric)
        logs["new_metric"] = self.metric
