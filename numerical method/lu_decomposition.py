import numpy as np

def lu_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs LU decomposition on a given square matrix.
    It raises an error if the matrix is not square or if LU decomposition is not possible.

    Parameters:
    matrix (np.ndarray): A square matrix to be decomposed.

    Returns:
    tuple[np.ndarray, np.ndarray]:
        - Lower triangular matrix (L) with unit diagonal.
        - Upper triangular matrix (U).

    Raises:
    ValueError: If the input matrix is not square.
    ArithmeticError: If LU decomposition is not possible.
    """
    # Extract the number of rows and columns
    num_rows, num_cols = matrix.shape
    
    # Ensure the matrix is square
    if num_rows != num_cols:
        raise ValueError(
            f"The input matrix must be square, but received a {num_rows}x{num_cols} matrix:\n{matrix}"
        )
    
    # Initialize lower and upper matrices with zeros
    L = np.zeros((num_rows, num_cols))
    U = np.zeros((num_rows, num_cols))
    
    # Perform LU decomposition
    for col in range(num_cols):
        # Compute elements of the lower triangular matrix (L)
        for row in range(col):
            sum_product = np.sum(L[row, :row] * U[:row, col])
            
            # If the diagonal of U is zero, LU decomposition is not possible
            if U[row, row] == 0:
                raise ArithmeticError("LU decomposition is not possible due to zero pivot element.")
            
            L[col, row] = (matrix[col, row] - sum_product) / U[row, row]
        
        # Set diagonal elements of L to 1
        L[col, col] = 1
        
        # Compute elements of the upper triangular matrix (U)
        for row in range(col, num_cols):
            sum_product = np.sum(L[col, :col] * U[:col, row])
            U[col, row] = matrix[col, row] - sum_product
    
    return L, U

if __name__ == "__main__":
    matrix = np.array([[4, 3, 2],
                   [2, 1, 3],
                   [6, 5, 4]])
    L, U = lu_decomposition(matrix)
