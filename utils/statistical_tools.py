import numpy as np
from scipy.stats import multivariate_normal
from decimal import Decimal, getcontext

def determinant(matrix):
    """
    Compute the determinant of a matrix. 
    Small or zero eigenvalues are replaced with a small positive value (0.01) 
    to avoid issues with singularity or negative determinants for covariance/correlation matrices.

    Parameters:
    - matrix: The input matrix (2D numpy array or equivalent).

    Returns:
    - The determinant of the matrix.
    """
    # Compute the eigenvalues and eigenvectors of the matrix
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Replace less than 0.01 eigenvalues with 0.01 to avoid singularity
    eigenvalues[eigenvalues < 0.0001 ] = 0.0001
    # Replace small or zero eigenvalues with 0.01 to avoid singularity
    determinant = np.prod(eigenvalues)
    return(determinant)

def make_positive_semidefinite(matrix, epsilon=1e-5):
    symmetric_matrix = (matrix + matrix.T) / 2  # Ensure symmetry
    min_eigenvalue = np.min(np.linalg.eigvalsh(symmetric_matrix))
    if min_eigenvalue < 0:
        adjustment = max(-min_eigenvalue + epsilon, epsilon)
        symmetric_matrix += adjustment * np.eye(matrix.shape[0])
    return symmetric_matrix


def multivariate_likelihood(x, mean_vector, covariance_matrix):
    """
    Calculate the multivariate Gaussian likelihood for each sample in X given the provided mean and covariance matrix.
    
    Parameters:
    - X: The input data matrix (NxD, where N is number of samples and D is dimensionality).
    - mean_vector: The mean vector of the multivariate Gaussian (1xD).
    - covariance_matrix: The covariance matrix of the multivariate Gaussian (DxD).

    Returns:
    - The log likelihood values for each sample in X (Nx1).
    """
    getcontext().prec = 6
    adjusted_matrix = make_positive_semidefinite(covariance_matrix)

    mvn = multivariate_normal(mean=mean_vector, cov = adjusted_matrix,allow_singular=True)
    pdf = mvn.pdf(x)
    return np.log(pdf)

def has_constant_columns(df):
    """Check if the DataFrame has any columns with zero variance."""
    return (df.var() == 0).any()