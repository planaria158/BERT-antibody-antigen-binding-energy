import math
import numpy as np

"""
    Calculate some error metrics on the test set
    Args:
        y: list, the ground truth values
        preds: list, the predicted values

    Returns:
        metrics: dict, a dictionary with the error metrics
"""
def test_metrics(y, preds):
    preds = [p[0] for p in preds]
    y = [a[0] for a in y]

    metrics = {}

    # RMSE
    rmse = math.sqrt(np.mean((np.array(y) - np.array(preds))**2))
    metrics['rmse'] = rmse

    # MSE
    mse = np.mean((np.array(y) - np.array(preds))**2)
    metrics['mse'] = mse

    # MAE
    mae = np.mean(np.abs(np.array(y) - np.array(preds)))
    metrics['mae'] = mae

    # Mean of the absolute percentage error
    mape = np.mean(np.abs(np.array(y) - np.array(preds))/np.array(y))
    metrics['mape'] = mape

    # Median of the absolute percentage error
    mdape = np.median(np.abs(np.array(y) - np.array(preds))/np.array(y))
    metrics['mdape'] = mdape

    # PPE10: percentage of time the prediction is within 10 percent of the ground truth
    ppe10 = np.mean(np.abs(np.array(y) - np.array(preds))/np.array(y) < 0.1)
    metrics['ppe10'] = ppe10

    # PPE20: percentage of time the prediction is within 10 percent of the ground truth
    ppe20 = np.mean(np.abs(np.array(y) - np.array(preds))/np.array(y) < 0.2)
    metrics['ppe20'] = ppe20

    return metrics
