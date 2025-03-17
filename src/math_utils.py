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
    R = X.copy() # avoid to modify the original matrix
    householder_vectors = []

    #TODO: add threshold for numerical stability
    #TODO: test right 
    
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
    
    return householder_vectors, R[:n, :]



def apply_householders_matrix(householder_vectors:list, A:np.ndarray) -> np.ndarray:
    '''
    Perform the product Q*A using Householder vectors instead of the full Q matrix

    Parameters:
    -----------
    householder_vectors: np.ndarray
        Householder vectors
    A: np.ndarray
        Matrix to be transformed

    Returns:
    --------
    np.ndarray
        Transformed matrix
    '''

    X = A.copy() # avoid to modify the original matrix

    for i, u in reversed(list(enumerate(householder_vectors))):
        # Restrict the operation to the active submatrix A[i:, i:]
        X[i:, i:] -= 2 * np.outer(u, u.T @ X[i:, i:])
    
    return X


def apply_householders_vector(householder_vectors:list, b:np.ndarray, reverse:bool=False) -> np.ndarray:
    '''
    Perform the product Q*b using Householder vectors instead of the full Q matrix

    Parameters:
    -----------
    householder_vectors: np.ndarray
        Householder vectors
    b: np.ndarray
        Vector to be transformed
    reverse: bool
        If True, apply the transformation in reverse order (represent Q^T)

    Returns:
    --------
    np.ndarray
        Transformed vector
    '''

    y = b.copy() # avoid to modify the original vector

    if reverse: # represent Q^T
        for i, u in reversed(list(enumerate(householder_vectors))):
            y[i:] -= 2 * np.outer(u, u.T @ y[i:])
    else: # represent Q
        for i, u in enumerate(householder_vectors):
            y[i:] -= 2 * np.outer(u, u.T @ y[i:])

    return y

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
        
        w[i] = (b[i] - np.dot(A[i, :i], w[:i])) / A[i, i] #TODO: controlla stabilità numerica (se A[i,i] è vicino a 0 trova soluzione)

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
        
        w[i] = (b[i] - np.dot(A[i, i+1:], w[i+1:])) / A[i, i] #TODO: controlla stabilità numerica (se A[i,i] è vicino a 0 trova soluzione)
    
    return w

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
        The updated list of Householder vectors and the new R matrix (only the upper triangular part)
    '''
    
    m, n = x_new.shape[0], len(householder_vectors)

    z = apply_householders_vector(householder_vectors, x_new, reverse=True)
    z0, z1 = z[:n], z[n:]

    # Compute the new Householder vector
    norm_z1 = np.linalg.norm(z1)
    u_new = z1.copy()
    u_new[0] += np.sign(z1[0]) * norm_z1
    u_new /= np.linalg.norm(u_new) #TODO: controlla stabilità numerica (se A[i,i] è vicino a 0 trova soluzione)

    z1 -= - 2*np.outer(u_new, u_new.T @ z1)

    return householder_vectors + [u_new], np.block([[R, z0], [np.zeros((m-n, n)), z1]])[:n+1, :]
