import numpy as np


def get_dims_from_border_matrix(border_matrix):
    """
    Returns the dimensions possible for given border matrix

    Parameters:
	-----------
	border_matrix: array size (N, N)
		The value border_matrix[i, j] equals 1, if j-th cell is the border of i-th
    """
    border_matrix = np.array(border_matrix)
    if border_matrix.ndim != 2:
        raise ValueError('border_matrix should be a square matrix (integer or boolean entries) representing border relations.')
    if border_matrix.shape[0] != border_matrix.shape[1]:
        raise ValueError('border_matrix should be a square matrix (integer or boolean entries) representing border relations.')

    dims = np.nan*np.zeros(len(border_matrix))
    i = 0
    while np.isnan(dims).any():
        nansum_previous = np.isnan(dims).sum()
        working_matrix = np.array(border_matrix)[np.isnan(dims), :]
        dims[(working_matrix == 0).all(axis=0) & np.isnan(dims)] = i
        if np.isnan(dims).sum() >= nansum_previous:
            raise ValueError("Given matrix can't represent the border matrix")
        i += 1

    dims = dims.astype(int)
    return dims