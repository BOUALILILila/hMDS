# Python implementation from Julia code in https://github.com/HazyResearch/hyperbolics/blob/master/hMDS/hmds-simple.jl
from typing import Tuple

import time

import numpy as np
from numpy import linalg as LA

def power_method(
    A: np.ndarray, 
    d: int, 
    tol: float=1e-8, 
    verbose: bool=False, 
    max_iter: int=1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Power Method for determining the largest Eigenvalues and Eigenvectors.
    
    The power method is an iterative algorithm to determine the d most significant
    eigenvalues of a square matrix A. For each eigenvalue, the algorithm works 
    by starting with a random initial eigenvector, and then iteratively applying
    the matrix A to the eigenvector and normalizing the result to obtain a sequence
    of improved approximations for the eigenvector. Use the gram-schmidt process
    to insure that each new eigenvector is orthogonal to the previous ones. 
    
    Parameters
    ----------
    A : np.ndarray
        The square matrix.
    d : int 
        Number of largest eigenvalues to determine.
    tol :  float, default=1e-8
        The tolerance for the eigenvalue and eigenvector approximations.
        i.e. the maximum allowed difference between the approximations and the actual values.
    verbose : bool, default=False
        Print logs.
    max_iter : int, default=1000
        The maximum number of iterations.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The approximations for the d largest (eigenvalues, corresponding eigenvectors) of A
        
    Raises
    ------
    AssertionError
        if the input tensor A is not a square matrix.
        if the number of eigenvalues d is greater than the matrix size.
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1], f'A={A} is not a square matrix, cannot perform power method'
    (n,n) = A.shape

    assert d <= n, f"The number of eigenvalues {d} is greater than the matrix size {n}"

    # Random orthogonal basis for the eigenvecotrs x_all
    # using the orthogonal Q matrix from the QR decomposition 
    x_all, _ = LA.qr(np.random.randn(n,d))
    _eig  = np.zeros(d)

    if verbose:
        print(f">> Entering Power Method {d} {tol} {max_iter} {n}")

    for j in range(d):
        start = time.time() if verbose else None
        x = x_all[:,j]
        x /= LA.norm(x)
        for t in range(max_iter):        
            x = np.matmul(A, x)
            # After finding the most significant eigenvalue and its associated
            # eigenvector, use the gram-schmidt process to ensure orthogonal
            if j > 0:
                yy = np.matmul(x, x_all[:, 0:j])
                for k in range(j):
                    x -= x_all[:,k]*yy[k]
            nx = LA.norm(x)
            x /= nx
            cur_dist = np.abs(nx - _eig[j])
            if not np.isinf(cur_dist).item() and np.minimum(cur_dist, cur_dist/nx) < tol:
                if verbose:
                    print(f"Done with eigenvalue {j} at iteration {t} at abs_tol={float(abs(nx - _eig[j]))} rel_tol={float(abs(nx - _eig[j])/nx)}")
                    print(f"Time Elapsed = {time.time()-start}")
                break

            if t % 500 == 0 and verbose:
                print(f"\t iter={t} dist={cur_dist}\t\t {cur_dist/nx}")
            _eig[j] = nx
        x_all[:,j] = x 
    return _eig, x_all
