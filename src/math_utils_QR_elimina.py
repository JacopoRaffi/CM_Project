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
    
    return householder_vectors, R



def apply_householders(householder_vectors:list, A:np.ndarray) -> np.ndarray:
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
    
    for i, u in reversed(list(enumerate(householder_vectors))):
        # Restrict the operation to the active submatrix A[i:, i:]
        A[i:, i:] -= 2 * np.outer(u, u.T @ A[i:, i:])
    
    return A