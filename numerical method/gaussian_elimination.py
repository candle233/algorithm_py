import numpy as np
from numpy.typing import NDArray

def back_substitution(
    upper_tri_matrix: NDArray[np.float64], constants: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    solve a system of linear equations with an upper triangular matrix.
    upper_tri_matrix: the upper triangular matrix of the system.
    constants: the constants of the system.
    return: the solution of the system.
    """
    num_vars = upper_tri_matrix.shape[0]
    solution = np.zeros((num_vars, 1), dtype=np.float64)
    
    for i in range(num_vars - 1, -1, -1):
        sum_values = np.dot(upper_tri_matrix[i, i + 1 :], solution[i + 1 :])
        solution[i, 0] = (constants[i, 0] - sum_values[0]) / upper_tri_matrix[i, i]
    
    return solution

def gaussian_elimination(
    coefficient_matrix: NDArray[np.float64], constant_terms: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    gauss elimination method to solve a system of linear equations.
    coefficient_matrix: the coefficient matrix of the system.
    constant_terms: the constants of the system.
    return: the solution of the system.
    """
    num_rows, num_cols = coefficient_matrix.shape
    if num_rows != num_cols:
        raise ValueError("The coefficient matrix must be square.")
    
    augmented_matrix = np.hstack((coefficient_matrix.astype(np.float64), constant_terms.astype(np.float64)))
    
    # 进行高斯消元，使其变为上三角矩阵
    for i in range(num_rows - 1):
        pivot = augmented_matrix[i, i]
        if pivot == 0:
            raise ValueError("Zero pivot!")
        
        for j in range(i + 1, num_rows):
            factor = augmented_matrix[j, i] / pivot
            augmented_matrix[j, :] -= factor * augmented_matrix[i, :]
    
    upper_triangular = augmented_matrix[:, :-1]
    new_constants = augmented_matrix[:, -1:]
    
    return back_substitution(upper_triangular, new_constants)

if __name__ == "__main__":
    '''test:
    x1-4x2-2x3=-2
    5x1+2x2-2x3=-3
    x1-x2=4
    '''
    test_A = np.array([[1, -4, -2], [5, 2, -2], [1, -1, 0]], dtype=np.float64)
    test_b = np.array([[-2], [-3], [4]], dtype=np.float64)
    result = gaussian_elimination(test_A, test_b)
    print("Solution:")
    print(result)
