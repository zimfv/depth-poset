import warnings

import numpy as np
import gudhi as gh
from scipy.sparse import csr_matrix

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
        self.border_matrix = csr_matrix(border_matrix, dtype=int)
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
            self.dp = depth.DepthPoset.from_border_matrix(border_matrix=self.get_border_matrix(), dims=self.dims, sources=self.order)


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

    def __str__(self):
        """
        """
        return f'<{self.cell0}, {self.cell1}>'
        
    def get_border_matrix(self):
        # returns dense border matrix
        return self.border_matrix.toarray()

    def get_transposition_type(self):
        """
        Returns the transposition type of the transposition.

        Defines this if it's not defined

        There are 6 types of transpositions:
        - 'birth-birth': both indices correspond to birth cells
        - 'death-death': both indices correspond to death cells
        - 'birth-death': one index is a birth, the other a death
        - 'birth-unpaired': one index is a birth, the other is not part of any pair
        - 'death-unpaired': one index is a death, the other is not part of any pair
        - 'unpaired-unpaired': neither transposing cell appears in any birth-death pair

        Returns:
        --------
        self.type : str 
            The transposition type.
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

    
    def define_pairs(self):
        """
        Identifies the paired birth/death index for each of the two transposed indices.

        For each index (index0 and index1), finds the index it is paired with in the 
        birth-death pairing, if any. The results are stored in `paired_index0` and `paired_index1`.

        This information is used later to determine the transposition's effect on the diagram.
        """
        self.paired_index0 = None
        self.paired_index1 = None
        for node in self.dp.nodes:
            if node.birth_index == self.index0:
                self.paired_index0 = node.death_index
            if node.death_index == self.index0:
                self.paired_index0 = node.birth_index
            if node.birth_index == self.index1:
                self.paired_index1 = node.death_index
            if node.death_index == self.index1:
                self.paired_index1 = node.birth_index


    def get_xyab(self):
        """
        Extracts indices involved in the classification of the transposition.

        Based on the transposition type, determines the indices (x, y, a, b), where:
        - a and x are the transposed indices,
        - b and y are their respective pairings.

        This method is only valid for 'birth-birth', 'death-death', and 'birth-death' types.

        Returns:
        --------
        x, y, a, b: tuple
            A tuple (x, y, a, b) of involved indices.

        Raises:
        -------
        ValueError
            If the transposition type is unsupported.
        """
        self.get_transposition_type()
        self.define_pairs()

        if self.type == 'birth-birth':
            a, x, y, b = self.index0, self.index1, self.paired_index1, self.paired_index0
        elif self.type == 'death-death':
            a, x, y, b = self.paired_index0, self.paired_index1, self.index1, self.index0
        elif self.type == 'birth-death':
            a, b, x, y = self.paired_index0, self.index0, self.index1, self.paired_index1
        else:
            msg = f"The transposition {self} have type {self.type}, but should have one of three: 'birth-birth', 'death-death' or 'birth-death'."
            raise ValueError(msg)
        return x, y, a, b


    def get_classifying_matrix(self, reduce_matrix=False):
        """
        Computes the matrix that classifies the transposition.

        Uses algebraic reductions on the border matrix to determine the effect 
        of the transposition. For 'birth-death' types, both column and row reductions 
        are performed. Optionally reduces the matrix to a submatrix corresponding to 
        the involved indices.

        Parameters:
        -----------
        reduce_matrix : bool
            If True, return only the reduced 2x2 or 4x4 submatrix used in switch classification.

        Returns:
        --------
        delta: array-like
            The (possibly reduced) classifying matrix.

        Raises:
        -------
        ValueError
            If the transposition type is unsupported.
        """
        x, y, a, b = self.get_xyab()
        stop_condition = lambda alpha, b0, delta: len({(a, b), (a, y), (x, b), (x, y)} & set(alpha.values())) + \
                                                  len({(a, x), (x, a), (b, y), (y, b)} & set(b0)) > 0
        
        if self.type == 'birth-birth':
            alpha, b0, delta = depth.reduct_column_bottom_to_top(self.get_border_matrix(), stop_condition)
        elif self.type == 'death-death':
            omega, b1, delta = depth.reduct_row_left_to_right(self.get_border_matrix(), stop_condition)
        elif self.type == 'birth-death':
            alpha, b0, delta = depth.reduct_column_bottom_to_top(self.get_border_matrix(), stop_condition)
            omega, b1, delta = depth.reduct_row_left_to_right(delta, stop_condition)
        else:
            msg = f"The transposition {self} have type {self.type}, but should have one of three: 'birth-birth', 'death-death' or 'birth-death'."
            raise ValueError(msg)
        
        if reduce_matrix:
            if self.type == 'birth-death':
                cords0 = np.sort([x, y, a, b])
                cords1 = np.sort([x, y, a, b])
            else:
                cords0 = np.sort([a, x])
                cords1 = np.sort([y, b])
            delta = tuple(map(tuple, delta[cords0][:, cords1]))
        return delta

 
    def get_switch_type(self):
        """
        Determines the switch type induced by the transposition.

        The switch type reflects how the persistence pairing is altered by 
        the transposition. The switch is determined by comparing the 
        classifying matrix to known canonical forms.

        Possible results are:
        - 'switch forward': canonical forward switch pattern
        - 'switch backward': canonical backward switch pattern
        - 'no switch': no meaningful change in pairing
        - 'undefined': if the indices are not adjacent or the transposition type is unsupported

        Returns:
        --------
        self.switch: str
            The switch type.
        """
        try:
            return self.switch
        except AttributeError:
            if self.index1 - self.index0 != 1:
                warnings.warn('self.index1 - self.index0 != 1')
            if self.index1 - self.index0 != 1 or self.get_transposition_type() not in ['birth-birth', 'death-death', 'birth-death']:
                self.switch = 'undefined'
                return self.switch
            
            delta = self.get_classifying_matrix(reduce_matrix=True)
            match self.type, delta:
                case 'birth-birth', ((1, 1), 
                                     (1, 0)):
                    self.switch = "switch forward"
                case 'birth-birth', ((1, 0), 
                                     (1, 1)):
                    self.switch = "switch backward"
                case 'death-death', ((0, 1), 
                                     (1, 1)):
                    self.switch = "switch forward"
                case 'death-death', ((1, 0), 
                                     (1, 1)):
                    self.switch = "switch backward"
                case 'birth-death', ((0, 1, 1, 0), 
                                     (0, 0, 0, 1), 
                                     (0, 0, 0, 1), 
                                     (0, 0, 0, 0)):
                    self.switch = "switch forward"
                case _:
                    self.switch = "no switch"
            return self.switch

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the transposition.

        This includes indices, associated cells, dimension, transposition type,
        and switch type. Useful for serialization, logging, or exporting data
        for visualization or further analysis.

        The 'type' field indicates the kind of transposition, with possible values:
        - 'birth-birth': both indices correspond to birth cells.
        - 'death-death': both indices correspond to death cells.
        - 'birth-death': one index is a birth, the other a death.
        - 'birth-unpaired': one index is a birth, the other is not in any pair.
        - 'death-unpaired': one index is a death, the other is not in any pair.
        - 'unpaired-unpaired': neither index is part of any birth-death pair.

        The 'switch' field classifies how the transposition affects the persistence pairing:
        - 'switch forward': changes the pairing in a canonical forward direction.
        - 'switch backward': changes the pairing in a canonical backward direction.
        - 'no switch': no change in the persistence pairing.
        - 'undefined': switch type could not be determined (e.g., indices not adjacent or the transposition type is unsupported).

        Returns:
        --------
        dict
            A dictionary with the following keys:
            - 'index 0': int, the first transposed index.
            - 'index 1': int, the second transposed index.
            - 'cell 0': object, the cell corresponding to index 0.
            - 'cell 1': object, the cell corresponding to index 1.
            - 'dim': int, the dimension of the transposition.
            - 'type': str, the transposition type (see above).
            - 'switch': str, the switch classification (see above).
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
            
    def next_order(self):
        # returns the order after the transposition
        new_order = list(self.order)
        new_order[self.index0], new_order[self.index1] = new_order[self.index1], new_order[self.index0]
        return new_order
    
    def next_dims(self):
        # returns the order after the transposition
        new_dims = list(self.dims)
        new_dims[self.index0], new_dims[self.index1] = new_dims[self.index1], new_dims[self.index0]
        return new_dims

    def next_border_matrix(self, dense=False):
        # returns the border matrix the transposition (the matrix will be sparse if dense is False)
        bm_new = self.border_matrix.copy()
        bm_new[[self.index0, self.index1]] = bm_new[[self.index1, self.index0]]
        bm_new[:, [self.index0, self.index1]] = bm_new[:, [self.index1, self.index0]]
        if dense:
            bm_new = bm_new.toarray()
        return bm_new
    
    def next_depth_poset(self):
        # 
        new_dims = self.next_dims()
        new_order = self.next_order()
        new_border_matrix = self.next_border_matrix(dense=True)
        new_depth_poset = depth.DepthPoset.from_border_matrix(border_matrix=new_border_matrix, dims=new_dims, sources=new_order)
        return new_depth_poset
    


        