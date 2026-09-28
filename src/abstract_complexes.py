import numpy as np
from galois import GF2




def get_boundary_matrix_ranks(ns, bs):
    """
    Returns the ranks for boundary matrices for the given numbers of cells and betti numbers
    
    If such complex cant exist raise ValueError 
    
    Parameters:
    -----------
    ns: vector length d
        The anount of cells in each dimension

    bs: vector length d
        The amount of cycle in each dimension, Betti numbers

    Returns:
    --------
    rs: vecotor length d - 1
        The ranks of boundary matrices
        The k-th value corespond the boundary matrix between cells dimension k and k + 1
    """
    ns = np.asarray(ns, dtype=int)
    bs = np.asarray(bs, dtype=int)
    assert ns.shape == bs.shape
    d = len(ns) - 1
    A = np.zeros([d + 1, d], dtype=int)
    for i in range(d):
        A[i, i] = 1
        A[i + 1, i] = 1
    y = (ns - bs)
    rs = np.linalg.solve(A[:-1], y[:-1]).astype(int)

    if (A[-1]@rs != y[-1]).any():
        raise ValueError('')
    
    return rs


def get_random_invertible_matrix_over_GF2(n):
    """
    Returns random invertible matrix shape (n, n) over GF2
    """
    while True:
        A = GF2.Random((n, n))
        if GF2._det(A) != 0:
            return A


def get_random_matrix_with_given_rank_over_GF2(n: int, m: int, r: int | None=None):
    """
    Return random matrix over GF2 of the given rank

    If rank is not given, the rank is minimum number of rows or columns

    Parameters:
    -----------
    n: int
        Number of rows

    m: int
        Number of columns

    r: int | None
        Expected rank
        Fedined as `min(n, m)` if not given

    Returns:
    --------
    M: galois.GF2 matrix shape (n, m)
    """
    if n < m:
        return get_random_matrix_with_given_rank_over_GF2(m, n, r).transpose()
    if r is None:
        r = min(n, m)
    if r > min(n, m):
        raise ValueError('The rank should be lower then the size of matrix')

    A = get_random_invertible_matrix_over_GF2(n)[:, :r]
    B = get_random_invertible_matrix_over_GF2(max(r, m - r))[:r, :][:, :m - r]
    M = np.concat([A, A @ B], axis=1)
    
    M = M[np.random.choice(n, n, replace=False), :][:, np.random.choice(m, m, replace=False)]

    return M
    
def get_random_boundary_matrix_next(delta1, n2, r2):
    """
    Returns the boundary matrix describing boundary relations between cells k and k + 1, 
    based on the boundary matrix describing boundary relations between cells k - 1 and k.

    Parameters:
    -----------
    delta1: galois.GF2 matrix shape (n1, n0)
        The boundary matrix describing boundary relations between cells k - 1 and k

    n2: int
        The number f celld simension k+1
    
    r2: int
        The expected rank of the matrix, the amount of linearly independent boundaries  

    Returns:
    -------
    delta2: galois.GF2 matrix shape (n2, n1)
        The boundary matrix describing boundary relations between cells k and k + 1
    """
    ker1 = delta1.null_space()

    delta2_basis = ker1.transpose() @ get_random_matrix_with_given_rank_over_GF2(len(ker1), r2)
    delta2_rest = delta2_basis @ get_random_matrix_with_given_rank_over_GF2(r2, n2 - r2)
    delta2 = np.concatenate([delta2_basis, delta2_rest], axis=1)
    delta2 = delta2[:, np.random.choice(np.arange(n2), n2, replace=False)]
    
    return delta2

def get_random_boundary_matrix(ns, bs):
    """
    Returns the random boundary matrix of the abstract complex with the given amount of cells and Betti numbers
    
    Parameters:
    -----------
    ns: vector length d
        The anount of cells in each dimension

    bs: vector length d
        The amount of cycle in each dimension, Betti numbers

    Returns:
    --------
    delta_matrix: np.array dtype int shape (n, n)
        The boundary matrix of the complex

    """
    rs = get_boundary_matrix_ranks(ns, bs)
    n = np.sum(ns)

    delta_matrix = GF2.Zeros([n, n])

    i0, i1, i2, i3 = 0, 0, 0, ns[0]
    for n2, r2 in zip(ns[1:], rs):
        i0, i1, i2, i3 = i1, i2, i3, i3 + n2
        delta_matrix[i1:i2, i2:i3] = get_random_boundary_matrix_next(delta_matrix[i0:i1, i1: i2], n2, r2)
        assert (delta_matrix[i0:i1, i1:i2] @ delta_matrix[i1:i2, i2:i3] == 0).all()

    delta_matrix = np.array(delta_matrix.tolist())
    return delta_matrix