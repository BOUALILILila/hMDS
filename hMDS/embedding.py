from typing import Tuple

import time

import numpy as np
from numpy import linalg as LA


def poincare_dist(x: np.ndarray, y:np.ndarray) -> float:
    """ Distance between two points in the Poincaré Model.

    Parameters
    ----------
    x : np.ndarray
        Embedding of the first point.
    y : np.ndarray
        Embedding of the second point.
    
    Returns
    -------
    float
        The distance between x and y in the Poincaré Ball Model
        
    """
    t = LA.norm(x-y)**2 / ((1 - LA.norm(x)**2) * (1 - LA.norm(y)**2))
    return np.arccosh(1.0 + 2.0 * t)

def gans_to_poincare(X: np.ndarray) -> np.ndarray:
    """ Projection from Gans Model to Poincaré Model.
    
    Parameters
    ----------
    X : np.ndarray
        The embedding matrix.
        
    Returns
    -------
    np.ndarray
        The matrix of embeddings in the Poincaré Model
    """
    assert len(X.shape) == 2, f"X is not a matrix: X.shape = {X.shape}"

    Y = 1.0 + np.sqrt(1.0 + LA.norm(X, axis=1)**2)
    Y = np.expand_dims(Y, axis=-1)
    return X/Y

def get_eig_vals_and_vecs(
    A: np.ndarray, 
    verbose: bool=False, 
) -> Tuple[np.ndarray, np.ndarray]:
    """ Determine the Eigenvalues and corresponding Eigenvectors (eigendecomposition).
    
    Parameters
    ----------
    A : np.ndarray
        The square matrix.
    verbose : bool, default=False
        Print logs.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The approximations for the (eigenvalues, and eigenvectors) of A
        
    Raises
    ------
    LinAlgError
        if the input tensor A is not a square array.
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1], f'A={A} is not a square matrix, cannot perform power method'
    (n,n) = A.shape
    # Slow power_method implemented in eigendecomposition.py 
    # _d, _U = power_method(torch.matmul(A.T,A), r, tol, verbose=verbose, max_iter=max_iter)

    # Faster linalg implementation which computes the eigenvalues and right eigenvectors of a square array
    _d, _U = LA.eig(np.matmul(A.T, A))
    _d = _d.astype(np.float32)
    _U = _U.astype(np.float32)
    # Sort from largest to smallest eigenvalue (abs)
    ind = np.argsort(np.abs(_d))[::-1]
    _d = _d[ind]
    _U = _U[:,ind]

    X = np.matmul(np.matmul(_U.T, A), _U) # _U.T @ A @ _U
    _d_signed = np.diag(X, k=0) # the eigenvalues

    if verbose:
        print(f"Log Off Diagonals: {float(np.log(np.norm( X - np.diag(_d_signed))))}")

    return _d_signed, _U

def PCA(Z: np.ndarray, k: int, verbose=False) -> Tuple[np.ndarray, int] :
    """ Run Principal Component Analysis on Z to find the k most significant 
    non-negative eigenvalues to recover X.
    
    Parameters
    ----------
    A : np.ndarray
        The square matrix.
    k : int
        The number of non-negative eigenvalues
    verbose : bool, default=False
        Print logs.
        
    Returns
    -------
    Tuple[np.ndarray, int]
        The recovered X matrix and the dimension of the submanifold
    
     Raises
    ------
    AssertionError
        if the number of eigenvalues k is less than 0.
    """
    assert k > 0, f"Rank k must be greater than 0, but k={k} was given."

    start = time.time() if verbose else None
    lambdasM, usM = get_eig_vals_and_vecs(Z, verbose=verbose) 
    lambdasM_pos = np.copy(lambdasM)
    usM_pos = np.copy(usM)

    # Among the n eigen values ordered by significance take the non-negative ones up to k
    n = Z.shape[0]
    idx = 0
    for i in range(n):
        if idx >= k:
            break
        if lambdasM[i] > 0 :
            lambdasM_pos[idx] = lambdasM[i]
            usM_pos[:,idx] = usM[:,i]
            idx += 1

    Xrec = np.matmul(usM_pos[:,0:idx], np.diag(np.sqrt(lambdasM_pos[0:idx])))
    
    if verbose:
        print(f"Time Elapsed = {time.time()-start}")
    
    return Xrec, idx

def hmds(D: np.ndarray, k: int, scale: float=1.) -> np.ndarray:
    Y = np.cosh(D*scale)
    print("Launching h-mds ...")
    start = time.time() 
    # this is a Gans model set of points
    Xrec, found_dimension = PCA(-Y, k)
    print(f"found {found_dimension} dimensions")

    # project from Gans Model to Poincaré Ball model
    X = gans_to_poincare(Xrec)
    print(f"Time Elapsed = {time.time()-start}")
    return X
