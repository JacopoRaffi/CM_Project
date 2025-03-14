import numpy as np

def thin_QR(X:np.ndarray) -> tuple:
    '''
    Thin QR decomposition of matrix X

    Parameters:
    -----------
    X: np.ndarray
        Input matrix

    Returns:
    --------
    tuple
        Householder vectors and R matrix (only the upper triangular part)
    '''
    pass

def forwad_substitution(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    '''
    Forward substitution for solving a lower triangular system of equations Ax=b

    Parameters:
    -----------
    A: np.ndarray
        Lower triangular matrix
    b: np.ndarray
        Right-hand side vector
    
    Returns:
    --------
    np.ndarray
        Solution vector
    '''

    w = np.zeros(b.shape)

    for i in range(len(b)):
        if A[i,i] == 0: #TODO: come risolvere?
            raise ValueError("The diagonal element is zero")
        
        w[i] = (b[i] - np.dot(A[i, :i], w[:i])) / A[i, i]

    return w

def backward_substitution(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    '''
    Backward substitution for solving an upper triangular system of equations Ax=b

    Parameters:
    -----------
    A: np.ndarray
        Upper triangular matrix
    b: np.ndarray
        Right-hand side vector
    
    Returns:
    --------
    np.ndarray
        Solution vector
    '''
    
    w = np.zeros(b.shape)

    for i in range(len(b)-1, -1, -1):
        if A[i,i] == 0: #TODO: come risolvere?
            raise ValueError("The diagonal element is zero")
        
        w[i] = (b[i] - np.dot(A[i, i+1:], w[i+1:])) / A[i, i]
    
    return w

def incr_QR(X_new:np.ndarray, householders:list, R:np.ndarray) -> tuple:
    '''
    Incremental QR decomposition of matrix X

    Parameters:
    -----------
    X_new: np.ndarray
        Input matrix
    householder: list
        Householder vectors from the previous QR factorization of X
    R_0: np.ndarray
        R matrix from the previous QR factorization of X

    Returns:
    --------
    tuple
        new Householder vectors and R matrix (only the upper triangular part)
    '''
    pass