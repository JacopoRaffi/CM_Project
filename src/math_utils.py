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
        
        w[i] = (b[i] - np.dot(A[i, :i], w[:i])) / A[i, i] #TODO: controlla stabilitá numerica (se A[i,i] é vicino a 0 trova soluzione)

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
        
        w[i] = (b[i] - np.dot(A[i, i+1:], w[i+1:])) / A[i, i] #TODO: controlla stabilitá numerica (se A[i,i] é vicino a 0 trova soluzione)
    
    return w

def apply_householders(householder_vectors:list, v:np.ndarray) -> np.ndarray:
    '''
    Perform the product Q*v using Householder vectors instead of the full Q matrix

    Parameters:
    -----------
    householder_vectors: np.ndarray
        Householder vectors
    v: np.ndarray
        Vector to be transformed

    Returns:
    --------
    np.ndarray
        Transformed vector
    '''
    pass

def incr_QR(x_new:np.ndarray, householder_vectors:list, R:np.ndarray) -> tuple:
    '''
    Incremental QR decomposition of matrix X

    Parameters:
    -----------
    x_new: np.ndarray
        The new feature colunmn added to the original matrix
    householder: list
        Householder vectors from the previous QR factorization of X
    R_0: np.ndarray
        R matrix from the previous QR factorization of X

    Returns:
    --------
    tuple
        The new Householder vector and R matrix (only the upper triangular part)
    '''
    
    m, n = x_new.shape[0], len(householder_vectors)

    z = apply_householders(householder_vectors, x_new)
    z_0, z_1 = z[:n], z[n:]

    # Compute the new Householder vector
    norm_z = np.linalg.norm(z)
    u_new = z.copy()
    u_new[0] += np.sign(z[0]) * norm_z
    u_new /= np.linalg.norm(u_new)

    #TODO: form the new matrix R and consider only the upper triangular part (n+1 x n+1)
    # H_new * z_1 = z_1 - 2*np.outer(u_new, u_new.T @ z_1)


    return householder_vectors + [u_new], 
