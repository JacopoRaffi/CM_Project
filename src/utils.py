import numpy as np

def mae(y_true, y_pred):
    '''
    Mean Absolute Error (MAE) metric

    Parameters:
    -----------
    y_true: np.ndarray
        True labels
    y_pred: np.ndarray
        Predicted labels

    Returns:
    --------
    float
    '''
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    '''
    Mean Squared Error (MSE) metric
    
    Parameters:
    -----------
    y_true: np.ndarray
        True labels
    y_pred: np.ndarray
        Predicted labels
    
    Returns:
    --------
    float
    '''
    return np.mean((y_true - y_pred) ** 2)