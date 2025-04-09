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