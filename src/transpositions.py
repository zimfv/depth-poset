import numpy as np
import gudhi as gh

from src import depth
from src.utils_lefschetz import get_dims_from_border_matrix

class Transposition:
    def __init__(self, border_matrix, index0: int, index1: int, order: list=None, dims: list=None, dp: depth.DepthPoset=None):
        """
        Initializes a Transposition instance.

        Parameters
        ----------
        border_matrix : array-like of shape (n, n)
            A square matrix (integer or boolean entries) representing border relations.

        index0 : int
            Index of the first element to transpose.

        index1 : int
            Index of the second element to transpose.

        order : list, optional
            An optional list specifying the order of elements.

        dims : list, optional
            An optional list sepcifying the dimension of elements in th order

        dp : depth.DepthPoset, optional
            An optional DepthPoset object associated with the elements.
        """
        self.border_matrix = np.array(border_matrix)
        if self.border_matrix.ndim != 2:
            raise ValueError('border_matrix should be a square matrix (integer or boolean entries) representing border relations.')
        if self.border_matrix.shape[0] != self.border_matrix.shape[1]:
            raise ValueError('border_matrix should be a square matrix (integer or boolean entries) representing border relations.')
        
        self.order = order
        if self.order is None:
            self.order = list(np.arange(len(self.border_matrix)))

        self.dims = dims
        if self.dims is None:
            self.dims = get_dims_from_border_matrix(self.border_matrix)

        self.index0, self.index1 = index0, index1
        if self.index1 < self.index0:
            self.index0, self.index1 = self.index1, self.index0
        
        self.cell0 = order[self.index0]
        self.cell1 = order[self.index1]

        if self.dims[self.index0] == self.dims[self.index1]:
            self.dim = self.dims[self.index0]
        else:
            self.dim = None
        
        self.dp = dp
        if self.dp is None:
            # so I skip the filtration values. We dont need this for transpositions
            self.dp = depth.DepthPoset.from_border_matrix(border_matrix=self.border_matrix, dims=self.dims, sources=self.dims)


    @classmethod
    def from_simplex_tree(cls, stree: gh.SimplexTree, simplex0: tuple, simplex1: tuple):
        """
        Initializes a Transposition instance.

        Parameters
        ----------
        stree : gh.SimplexTree
            The Simplicial complex right before the transposition

        simplex0 : tuple
            The first transposing simplex

        simplex1 : tuple
            The second transposing simplex
        """
        order, border_matrix = depth.get_ordered_border_matrix_from_simplex_tree(stree)
        index0 = order.index(simplex0)
        index1 = order.index(simplex1)
        dims = [len(simplex) - 1 for simplex in order]
        dp = depth.DepthPoset.from_simplex_tree(stree)
        return cls(border_matrix, index0, index1, order, dims, dp)

        
    def get_transposition_type(self):
        """
        Returns the transposition type of the transposition.

        Defines this if it's not defined
        """
        try:
            return self.type
        except AttributeError:
            births = [node.birth_index for node in self.dp.nodes]
            deaths = [node.death_index for node in self.dp.nodes]
            if self.index0 in births and self.index1 in births:
                self.type = 'birth-birth'
            elif self.index0 in deaths and self.index1 in deaths:
                self.type = 'death-death'
            elif self.index0 in births and self.index1 in deaths:
                self.type = 'birth-death'
            elif self.index0 in deaths and self.index1 in births:
                self.type = 'birth-death'
            elif self.index0 in births or self.index1 in births:
                self.type = 'birth-unpaired'
            elif self.index0 in deaths or self.index1 in deaths:
                self.type = 'death-unpaired'
            else:
                self.type = 'unpaired-unpaired'
            return self.type
            

    def get_switch_type(self):
        pass


    def to_dict(self) -> dict:
        """
        """
        return {
            'index 0': self.index0, 
            'index 1': self.index1, 
            'cell 0': self.cell0, 
            'cell 1': self.cell1, 
            'dim': self.dim, 
            'type': self.get_transposition_type(), 
            'switch': self.get_switch_type(), 
        }
            
            