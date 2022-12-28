#
#

import numpy as np

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(preds):
    predictions, labels = preds
    predictions = np.argmax(predictions, axis=1)

    metrics = {}

    metrics['accuracy'] = round(accuracy_score(labels, predictions), 4) * 100
    return metrics