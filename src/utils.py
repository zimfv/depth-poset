import numpy as np
import itertools


def iterate_cubical_cells(shape, k=None, process=None):
    """
    Generate all k-dimensional hypercubes (as sets of 2^(dim-k) vertex indices) 
    in a regular grid of the specified shape.

    This function iterates over all axis-aligned k-dimensional hypercubes 
    embedded in a grid of a given shape. Each hypercube is represented by the 
    indices of its 2^(dim-k) corner points.

    Parameters
    ----------
    shape : tuple of int
        The shape of the grid. Each entry corresponds to the number of points 
        along that dimension.
    k : int or None, optional
        The dimension of the hypercubes to iterate over. If `k` is None, iterate 
        over all dimensions from 0 to len(shape) (inclusive).
    process : callable or None, optional
        A function to apply to each set of indices before yielding. It should 
        accept a NumPy array of shape (2^(dim-k), dim) and return a modified 
        version of it.

    Yields
    ------
    element : tuple of tuples of int
        A tuple representing a set of indices corresponding to the 2^(dim-k) 
        vertices of a k-dimensional hypercube in the grid. Each vertex is 
        represented as a tuple of coordinates.

    Notes
    -----
    - The function uses axis-aligned combinations of directions to define hypercubes.
    - Each k-dimensional hypercube lies in a (dim-k)-dimensional affine subspace 
      of the grid.
    - The yielded elements can be used to define cells (e.g., edges, squares, 
      cubes) in a cubical complex.

    Examples
    --------
    >>> list(iterate_neigboring_index((2, 2), k=1))
    [((0, 0), (1, 0)), ((0, 1), (1, 1)),
     ((0, 0), (0, 1)), ((1, 0), (1, 1))]
    >>> list(iterate_neigboring_index((2, 2), k=0))
    [((0, 0),), ((1, 0),), ((0, 1),), ((1, 1),)]
    """
    dim = len(shape)
    if k is None:
        for k in range(0, dim + 1):
            for element in iterate_cubical_cells(shape, k=k):
                yield element
            
    for shifting in itertools.combinations(np.arange(dim), dim - k):
        shifts = np.zeros([2**(dim - k), dim], dtype=int)
        shifts[:, list(shifting)] = list(itertools.product([0, 1], repeat=dim - k))

        current_shape = np.array(shape)
        current_shape[list(shifting)] -= 1
        current_shape = tuple(current_shape)
        for idx in np.ndindex(current_shape):
            element = shifts + np.array(idx)
            if process is not None:
                element = process(element)    
            element = tuple([tuple([int(i) for i in e]) for e in element])
            yield element


def array_to_tuple(arr, dtype=int):
    """
    Recursively converts a nested NumPy array into a nested tuple, casting elements to a given type.

    parameters:
    -----------
    arr: (np.ndarray or any nested iterable)
        The input array or nested structure to convert.
    
    dtype (type, optional)
        The desired data type for the elements. Defaults to int.

    Returns:
    --------
    tuple
        A nested tuple with all elements converted to the specified type.
    """
    if isinstance(arr, np.ndarray):
        return tuple(array_to_tuple(sub) for sub in arr)
    else:
        return dtype(arr)
    

def jacard_index(set0: set, set1: set) -> float:
    """
    Returns the Jacard Index of 2 sets.
    """
    if len(set0 | set1) == 0:
        return 1.0
    return len(set0 & set1) / len(set0 | set1)


def get_cross_parameters(y0, y1, t0=0, t1=1, filter_outside=True):
    """
    Returns the matrix of cross parameters between two arrays.

    Parameters:
    -----------
    y0, y1: arrays of the same length N
        The y0[i] and y1[i] are the points of line i at moments t0 and t1
    
    t0, t1: floats
        The moments of fixing the points
    
    filter_outside: bool
        If True, filter out the values outside the range [t0, t1]

    Returns:
    --------
    cross_parameters: the matrix shape (N, N)
        The moments when lines cross
    """
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    if y0.shape != y1.shape:
        raise ValueError("Arrays must have the same length.")
    if y0.ndim != 1:
        raise ValueError("Arrays must be 1D.")
    
    yi0 = y0.reshape(-1, 1)
    yi1 = y1.reshape(-1, 1)
    yj0 = y0.reshape(1, -1)
    yj1 = y1.reshape(1, -1)

    cross_parameters = (t1 - t0)*(yj0 - yi0)/(yi1 - yi0 - yj1 + yj0) + t0


    if filter_outside:
        cross_parameters[cross_parameters < min(t0, t1)] = np.nan
        cross_parameters[cross_parameters > max(t0, t1)] = np.nan
    return cross_parameters