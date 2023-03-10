import pytest

import numpy as np
from numpy import linalg as LA

from hMDS.embedding import gans_to_poincare, get_eig_vals_and_vecs, PCA
from hMDS.utils import quadratic_form


@pytest.fixture
def gans_point():
    return np.array([[ 2.421e+02,  2.900e+00, -1.210e+00, -1.000e-02,  8.800e-01,
        0.000e+00, -0.000e+00]])

@pytest.fixture
def symmetric_matrix():
    return np.array([
              [0.,  6,  8,  9,  10, 11, 12, 12], 
              [6.,  0,  6,  7,  8,  9,  10, 10],
              [8.,  6,  0,  7,  8,  9,  10, 10],
              [9.,  7,  7,  0,  7,  8,  9,  9],
              [10,  8,  8,  7,  0,  5,  6,  6],
              [11,  9,  9,  8,  5,  0,  5,  5],
              [12, 10, 10,  9,  6,  5,  0,  4],
              [12, 10, 10,  9,  6,  5,  4,  0],
            ])

def test_gans_to_poincare(gans_point):
    x = gans_to_poincare(gans_point)
    assert LA.norm(x) < 1 # on the Poincaré Manifold 

def test_get_eig_vals_and_vecs(symmetric_matrix):
    eigvalues, eigvectors = get_eig_vals_and_vecs(symmetric_matrix)
    AX = np.matmul(symmetric_matrix,eigvectors)
    lambdaX = np.expand_dims(eigvalues, axis=0)*eigvectors
    assert np.allclose(AX, lambdaX, atol=1e-5, rtol=1e-5)

def test_PCA(symmetric_matrix):
    Y = np.cosh(symmetric_matrix)
    n = Y.shape[0]
    Xrec, found_dimension = PCA(-Y, k=n)
    assert found_dimension < n 
    assert np.allclose(
        np.stack([quadratic_form(Xrec[i]) for i in range(n)]), 
        np.ones(n), 
        atol=1e-5, rtol=1e-5
    ) 

