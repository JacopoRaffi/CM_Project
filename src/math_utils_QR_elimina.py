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
    
    m, n = X.shape
    R = X.copy().astype(float)
    householder_vectors = []
    
    for j in range(n):
        # Select column below diagonal
        x = R[j:, j]
        
        # Compute Householder vector
        norm_x = np.linalg.norm(x)
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u /= np.linalg.norm(u)
        
        # Apply Householder transformation
        R[j:, :] -= 2 * np.outer(u, u.T @ R[j:, :])
        
        # Store Householder vector
        householder_vectors.append(u)
    
    return householder_vectors, R[:n, :n]


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
    pass

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
    pass

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