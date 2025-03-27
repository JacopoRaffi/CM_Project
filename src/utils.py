import numpy as np
import time
import matplotlib.pyplot as plt

from math_utils import *

def mae(y_true, y_pred):
    '''
    Mean Absolute Error (MAE) metric

    Parameters
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
    
    Parameters
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


def residual_QR(X):
    '''
    Compute the residual of the QR factorization of a matrix X

    Parameters
    -----------
    X: np.ndarray
        Input matrix

    Returns:
    --------
    float
    '''
    m, n = X.shape
    h, R = thin_QR(X)
    R = np.vstack((R, np.zeros((m-n, n))))

    return np.linalg.norm(X - apply_householders_matrix(h, R)) / np.linalg.norm(X)

def plot_errorbar(START=1000, END=5001, STEP=1000, mean_and_variance=None):
    '''
    Plot the mean and variance of the time

    Parameters
    -----------
    START: int
        Starting value of m
    END: int
        Ending value of m
    STEP: int
        Step of m
    mean_and_variance: iterable
        List of tuple (mean, variance) for each value of m
    '''

    m_values = np.array(range(START, END, STEP))
    means = np.array([x[0] for x in mean_and_variance])
    variances = np.array([x[1] for x in mean_and_variance])

    a, b = np.polyfit(m_values, means, 1)
    fit_line = a * m_values + b  # compute y values for the fit line

    plt.errorbar(range(START, END, STEP), means, yerr=variances, fmt='o')
    plt.plot(m_values, fit_line, linestyle='--', color='orange')
    plt.xlabel('m')
    plt.ylabel('Time (s)')
    plt.show()





def plot_time_mean_variance(n=256, trials=5, START=1000, END=5001, STEP=1000):
    '''
    Plot the mean and variance of the time taken to compute the QR factorization of a matrix X

    Parameters
    -----------
    n: int
        Number of columns of the random data
    trials: int
        Number of trials to compute the mean and variance
    START: int
        Starting value of m
    END: int
        Ending value of m
    STEP: int
        Step of m
    
    Returns:
    --------
    None
    '''
    a, b = -1, 1  # range of the random data
    
    mean_and_variance = []  # list of tuple (mean, variance) for each value of m

    for m in range(START, END, STEP):
        times_trials = []
        for _ in range(trials):
            # generate random data
            X = np.random.uniform(a, b, size=(m, n))

            start_time = time.time()
            _, _ = thin_QR(X)
            end_time = time.time()
            times_trials.append(end_time - start_time)

        # compute the mean and variance of the data
        mean_and_variance.append((np.mean(times_trials), np.var(times_trials)))

    m_values = np.array(range(START, END, STEP)) # TODO: plot code to be moved in a separate function
    means = np.array([x[0] for x in mean_and_variance])
    variances = np.array([x[1] for x in mean_and_variance])

    a, b = np.polyfit(m_values, means, 1)
    fit_line = a * m_values + b  # compute y values for the fit line

    plt.errorbar(range(START, END, STEP), means, yerr=variances, fmt='o')
    plt.plot(m_values, fit_line, linestyle='--', color='orange')
    plt.xlabel('m')
    plt.ylabel('Time (s)')
    plt.show()


