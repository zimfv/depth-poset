import itertools
import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata

from src import depth, utils

from functools import total_ordering


class CubicalTorusComplex():
    def __init__(self, shape, dim=None, vector_sets=None):
        """
        """
        if dim is None:
            dim = len(shape)
        else:
            shape = np.array(shape)[:dim]
        self.shape = tuple(shape)
        self.dim = int(dim)

        if vector_sets is None:
            # Defines the orientation of cells
            self.vector_sets = [np.unique(list(itertools.permutations(np.concatenate([np.zeros(dim - i, dtype=bool), 
                                                                                      np.ones(i, dtype=bool)]))), axis=0) for i in range(dim + 1)]
        else:
            self.vector_sets = list(vector_sets)
        
    def assign_filtration(self, filtration_values: list):
        """
        Assign the filtration values of cubical simplex

        Parameters:
        -----------
        filtration_values: list of self.dim + 1 np.arrays each i-th should have shape (comb(self.dim, i), self.chape)
            The i-th matrix coresponds the filtration values of i-skeleton of the Complex. 
            The orientation of the cells is defined by self.vector_sets
        """
        self.filtration_values = filtration_values
        return self
    
    def assign_random_barycentric_filtration(self, levels=None):
        """
        Assign random filtration values to the complex using barycentric filtration.

        Parameters:
        -----------
        levels: list of floats, optional
            The filtration levels for the complex. If None, the default levels are used, which are
            np.arange(self.dim + 2). The filtration values will be randomly assigned between these levels
            for each vector set.
            If levels are provided, they should be of length self.dim + 2.
            The filtration values will be randomly assigned between these levels for each vector set.
        
        Returns:
        --------
        self: CubicalTorusComplex
            The complex with assigned filtration values.
        """
        if levels is None:
            levels = np.arange(self.dim + 2)
        filtration_values = [levels[i] + np.random.random(np.append(len(vector_set), self.shape))*(levels[i + 1] - levels[i]) for i, vector_set in enumerate(self.vector_sets)]
        self.assign_filtration(filtration_values)
        return self
    
    def assign_height_filtration(self, filtration_values):
        """
        Assign the filtration values of cubical simplex using height filtration.

        Parameters:
        -----------
        filtration_values: np.array of shape self.shape
            The filtration values in the vertices
        """
        filtration_values = np.asarray(filtration_values)
        if filtration_values.shape != self.shape:
            raise ValueError(f'The filtrtation values matrix should have shape {self.shape}')
        pass
    
    def draw_strong(self, cmap='jet', vmin=None, vmax=None, ax=None, all_borders=True, zorder=0):
        """
        Draw the strong filtration of the complex.
        
        Parameters:
        -----------
        cmap: str, optional
            The colormap to use for the filtration values. Default is 'jet'.
        vmin: float, optional
            The minimum value for the colormap. If None, the minimum value of the filtration values     
        """
        if self.dim != 2:
            return
        if ax is None:
            ax = plt.gca()
        if vmin is None:
            vmin = self.filtration_values[0].min()
        if vmax is None:
            vmax = self.filtration_values[-1].max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        sm = ScalarMappable(cmap=cmap, norm=norm)

        matrix = self.filtration_values[0][0]
        for x0, x1 in itertools.product(np.arange(self.shape[0]), np.arange(self.shape[1])):
            color = cmap(norm(matrix[x0, x1]))
            ax.scatter(x0, x1, color=color, zorder=2)
            x = np.array([x0, x1])
            if x0 == 0 and all_borders:
                ax.scatter(self.shape[0], x1, color=color, zorder=2 + zorder)
            if x1 == 0 and all_borders:
                ax.scatter(x0, self.shape[1], color=color, zorder=2 + zorder)
            if (x0 == 0) and (x1 == 0) and all_borders:
                ax.scatter(self.shape[0], self.shape[1], color=color, zorder=2 + zorder)

        
        for (v0, v1), matrix in zip(self.vector_sets[1], self.filtration_values[1]):
            for x0, x1 in itertools.product(np.arange(self.shape[0]), np.arange(self.shape[1])):
                color = cmap(norm(matrix[x0, x1]))
                ax.plot([x0, x0 + v0], [x1, x1 + v1], color=color, zorder=1 + zorder, linewidth=4)
                if x0 == 0 and all_borders:
                    ax.plot([self.shape[0], self.shape[0]], [x1, x1 + v1], color=color, zorder=1 + zorder, linewidth=4)
                if x1 == 0 and all_borders:
                    ax.plot([x0, x0 + v0], [self.shape[1], self.shape[1]], color=color, zorder=1 + zorder, linewidth=4)
                        
        matrix = self.filtration_values[2][0]
        for x0, x1 in itertools.product(np.arange(self.shape[0]), np.arange(self.shape[1])):
            color = cmap(norm(matrix[x0, x1]))
            ax.fill_between(np.arange(2) + x0, np.zeros(2) + x1, np.zeros(2) + x1 + 1, color=color, zorder=0 + zorder)
        
        return sm
    
    def draw_gradiental(self, cmap='jet', vmin=None, vmax=None, ax=None, method='cubic', n=300):
        """
        Draw the gradiental filtration of the complex.
        
        Parameters:
        -----------
        cmap: str, optional
            The colormap to use for the filtration values. Default is 'jet'.
        vmin: float, optional
            The minimum value for the colormap. If None, the minimum value of the filtration values
        vmax: float, optional
            The maximum value for the colormap. If None, the maximum value of the filtration values
        ax: matplotlib.axes.Axes, optional
            The axes to draw the filtration on. If None, the current axes will be used.
        method: str, optional
            The method to use for interpolation. Default is 'cubic'.
        n: int, optional
            The number of points to use for interpolation in each direction. Default is 300.
        
        Returns:
        --------
        sm: matplotlib.cm.ScalarMappable
            The scalar mappable object for the colormap.
        """
        if self.dim != 2:
            return
        if ax is None:
            ax = plt.gca()
        if vmin is None:
            vmin = self.filtration_values[0].min()
        if vmax is None:
            vmax = self.filtration_values[-1].max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        sm = ScalarMappable(cmap=cmap, norm=norm)

        points0 = np.array(list(itertools.product(np.arange(self.shape[0]), np.arange(self.shape[1]))))
        heights0 = np.concatenate(self.filtration_values[0][0])
        points1 = np.concatenate([points0 + 0.5*np.array(vector) for vector in self.vector_sets[1]])
        heights1 = np.concatenate(np.concatenate(self.filtration_values[1]))
        points2 = points0 + 0.5*np.ones(2)
        heights2 = np.concatenate(self.filtration_values[2][0])
        points = np.concatenate([points0, points1, points2])
        heights = np.concatenate([heights0, heights1, heights2])

        heights_add = heights[(points[:, 0] == 0) | (points[:, 1] == 0)]
        points_add = points[(points[:, 0] == 0) | (points[:, 1] == 0)]
        points_add[points_add[:, 0] == 0, 0] = self.shape[0]
        points_add[points_add[:, 1] == 0, 1] = self.shape[1]

        points_zero = [(self.shape[0], 0), (0, self.shape[1])]
        heights_zero = np.ones(2)*self.filtration_values[0][0][0, 0]

        points = np.concatenate([points, points_add, points_zero])
        heights = np.concatenate([heights, heights_add, heights_zero])

        x, y = points[:, 0], points[:, 1]
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), n),
            np.linspace(y.min(), y.max(), n)
        )
        grid_z = griddata(points, heights, (grid_x, grid_y), method=method)
        
        ax.imshow(grid_z, extent=(x.min(), x.max(), y.min(), y.max()), cmap=cmap, norm=norm, aspect='auto')
        return sm

    def get_order(self, sort_with_filtration=True, map=utils.array_to_tuple, return_filtration=False, return_dims=False):
        """
        Get the order of the cells in the complex.

        Parameters:
        -----------
        sort_with_filtration: bool, optional
            If True, the order will be sorted by the filtration values. Default is True.
        map: function, optional
            A function to map the cells to a tuple. Default is utils.array_to_tuple.
        return_filtration: bool, optional
            If True, the filtration values will be returned. Default is False.
        return_dims: bool, optional
            If True, the dimensions of the cells will be returned. Default is False.

        Returns:
        --------
        order: list of tuples
            The order of the cells in the complex, where each cell is represented as a tuple.
        dims: np.array, optional
            The dimensions of the cells in the complex, if return_dims is True.
        filtration_values: np.array, optional
            The filtration values of the cells in the complex, if return_filtration is True.
        """
        order = []
        
        basis = np.eye(self.dim, dtype=int)
        for celldim, vectorset in enumerate(self.vector_sets):
            for vector in vectorset:
                # Ебать, я тут хуйню наворотил! В общем, это - координаты клетки в нуле.
                zerocell = np.array([np.zeros(self.dim, dtype=int) if np.logical_not(i).all() else basis[vector][list(i)].sum(axis=0) for i in itertools.product([False, True], repeat=celldim)])
                for point in itertools.product(*[np.arange(n) for n in self.shape]):
                    cell = zerocell + np.array(point)
                    cell %= np.array(self.shape)
                    cell = map(cell)
                    order.append(cell)

        dims = np.concatenate([celldim + np.zeros(np.prod(matrix.shape), dtype=int) for celldim, matrix in enumerate(self.filtration_values)])
        values = np.concatenate([matrix.flatten() for matrix in self.filtration_values])
        
        if sort_with_filtration:
            indices = np.lexsort((dims, values))
            dims = dims[indices]
            values = values[indices]
            order = list(np.array(order, dtype=object)[indices])
        
        if (not return_filtration) and (not return_dims):
            return order
        result = [order]
        if return_dims:
            result.append(dims)
        if return_filtration:
            result.append(values)
        return tuple(result)

    def get_border_matrix(self, sort_with_filtration=True, dtype=int):
        """
        Get the border matrix of the complex.

        Parameters:
        -----------
        sort_with_filtration: bool, optional
            If True, the order will be sorted by the filtration values. Default is True.
        
        dtype: data-type, optional
            The data type of the border matrix. Default is int.
        self.essential_cells_dims
        Returns:
        --------
        border_matrix: np.array
            The border matrix of the complex, where the (i, j)-th entry is
            1 if the i-th cell is a face of the j-th cell, and 0 otherwise.
        """
        # Очень неэффективное решение, но должно работать!
        # This works incorrectly if self.shape contains n <= 2
        order, dims = self.get_order(map = lambda s: set(utils.array_to_tuple(s)), return_dims=True, sort_with_filtration=sort_with_filtration)
        order = np.array(order, dtype=object)
        f = np.vectorize(lambda s0, s1: (s0 & s1) == s0)
        matrix = f(order.reshape(-1, 1), order.reshape(1, -1)) & (dims.reshape(-1, 1) - dims.reshape(1, -1) == -1)
        matrix = matrix.astype(dtype)

        return matrix

    def get_depth_poset(self, sort_with_filtration=True) -> depth.DepthPoset:
        """
        Get the depth poset of the complex.

        Parameters:
        -----------
        sort_with_filtration: bool, optional
            If True, the order will be sorted by the filtration values. Default is True.
        
        Returns:
        --------
        dp: depth.DepthPoset
            The depth poset of the complex, where the cells are ordered by their filtration values.
        """
        order, dims, fvals = self.get_order(sort_with_filtration=sort_with_filtration, return_dims=True, return_filtration=True)
        if not sort_with_filtration:
            fvals = np.arange(len(order))
        bm = self.get_border_matrix(sort_with_filtration=sort_with_filtration)
        dp = depth.DepthPoset.from_border_matrix(border_matrix=bm, dims=dims, filter_values=fvals, sources=order)
        return dp

    
@total_ordering
class EssentialCell:
    """
    Represents an essential cell in a cubical torus complex.
    """
    def __init__(self, dim, conds):
        self.dim = int(dim)
        self.conds = np.array(conds, dtype=bool)
    
    def __repr__(self):
        code = ''.join(self.conds.astype(int).astype(str))
        return f'EssentialCell(dim={self.dim}, conds={code})'

    def __str__(self):
        if self.dim == -1:
            return '$\\emptyset$'
        code = ''.join(self.conds.astype(int).astype(str))
        return f'$\\mathfrak{{E}}^{{{self.dim}}}_{{{code}}}$'

    def __hash__(self):
        return hash((self.dim, tuple(self.conds)))

    def __eq__(self, other):
        if isinstance(other, EssentialCell):
            return (self.dim == other.dim) and (self.conds == other.conds).all()
        if isinstance(other, tuple):
            return False
    
    def __lt__(self, other):
        if isinstance(other, EssentialCell):
            if self.dim < other.dim:
                return True
            if self.dim > other.dim:
                return False
            return tuple(self.conds) < tuple(other.conds)
        if isinstance(other, tuple):
            other_dim = round(np.log(len(other))/np.log(2))
            if other_dim < self.dim:
                return False
            if other_dim > self.dim:
                return True
            return False
    
    def has_border(self, cell, dim=None):
        """
        Check if the essential cell has a border with the given cell.

        Parameters:
        -----------
        cell: set
            The cell to check for a border with.

        Returns:
        --------
        bool
            True if the essential cell has a border with the given cell, False otherwise.
        """
        if dim is None:
            dim = int(round(np.log(len(cell))/np.log(2)))
        if self.dim != dim + 1:
            return False
        return (np.array(list(cell))[:, self.conds] == 0).all()
        

class CubicalTorusComplexExtended(CubicalTorusComplex):
    """
    The Cubical Torus Complex with added essential cells and filtration values.
    """
    def __init__(self, shape, dim=None, vector_sets=None):
        super().__init__(shape, dim, vector_sets)

        self.essential_cells = [EssentialCell(-1, np.zeros(self.dim, dtype=bool))]
        for k in range(1, self.dim + 1):
            for cond_indices in itertools.combinations(np.arange(self.dim), k):
                conds = np.zeros(self.dim, dtype=bool)
                conds[list(cond_indices)] = True
                self.essential_cells.append(EssentialCell(k + 1, conds))

    def assign_filtration(self, filtration_values, essential_filtration_values=None):
        """
        """
        super().assign_filtration(filtration_values)

        if essential_filtration_values is None:
            min_val = min([m.min() for m in self.filtration_values])
            max_val = max([m.min() for m in self.filtration_values])
            self.essential_filtration_values = max_val*np.ones(len(self.essential_cells))
            self.essential_filtration_values[0] = min_val
        else:
            self.essential_filtration_values = np.array(essential_filtration_values, dtype=float)
            if len(self.essential_filtration_values) != len(self.essential_cells):
                raise ValueError(f'The essential filtration values should have length 2^{self.dim} = {len(self.essential_cells)}')
        return self
            
    def assign_random_barycentric_filtration(self, essential_filtration_values=None, levels=None):
        """
        Assign random filtration values to the complex using barycentric filtration.

        Parameters:
        -----------
        essential_filtration_values: list of floats, optional
            The filtration values for the essential cells. 
            If None, the default values are used, which are minimum and maximum levels.

        levels: list of floats, optional
            The filtration levels for the complex. If None, the default levels are used, which are
            np.arange(self.dim + 2). The filtration values will be randomly assigned between these levels
            for each vector set.
            If levels are provided, they should be of length self.dim + 2.
            The filtration values will be randomly assigned between these levels for each vector set.
        
        Returns:
        --------
        self: CubicalTorusComplexExtended
            The complex with assigned filtration values.
        """
        if levels is None:
            levels = np.arange(self.dim + 2)
        
        if essential_filtration_values is None:
            min_val = min(levels)
            max_val = max(levels)
            essential_filtration_values = [min_val] + [max_val]*(len(self.essential_cells) - 1)
        elif len(essential_filtration_values) != len(self.essential_cells):
            raise ValueError(f'The essential filtration values should have length 2^{self.dim} = {len(self.essential_cells)}')

        super().assign_random_barycentric_filtration(levels=levels)
        self.assign_filtration(self.filtration_values, essential_filtration_values=essential_filtration_values)
        return self

    def get_order(self, sort_with_filtration=True, map=utils.array_to_tuple, return_filtration=False, return_dims=False):
        """
        Get the order of the cells in the complex.

        Parameters:
        -----------
        sort_with_filtration: bool, optional
            If True, the order will be sorted by the filtration values. Default is True.
        map: function, optional
            A function to map the cells to a tuple. Default is utils.array_to_tuple.
        return_filtration: bool, optional
            If True, the filtration values will be returned. Default is False.
        return_dims: bool, optional
            If True, the dimensions of the cells will be returned. Default is False.

        Returns:
        --------
        order: list of tuples
            The order of the cells in the complex, where each cell is represented as a tuple.
        dims: np.array, optional
            The dimensions of the cells in the complex, if return_dims is True.
        filtration_values: np.array, optional
            The filtration values of the cells in the complex, if return_filtration is True.
        """
        order, dims, values = super().get_order(sort_with_filtration=False, map=map, return_filtration=True, return_dims=True)
        order = np.concatenate([np.array(order, dtype=object), self.essential_cells])
        dims = np.concatenate([dims, np.array([cell.dim for cell in self.essential_cells])])
        filtration_values = np.concatenate([values, self.essential_filtration_values])

        if sort_with_filtration:
            indices = np.lexsort((dims, filtration_values))
            dims = dims[indices]
            filtration_values = filtration_values[indices]
            order = list(np.array(order, dtype=object)[indices])
        
        if (not return_filtration) and (not return_dims):
            return order
        result = [order]
        if return_dims:
            result.append(dims)
        if return_filtration:
            result.append(filtration_values)
        return tuple(result)
    
    def get_border_matrix(self, sort_with_filtration=True, dtype=int):

        """
        Get the border matrix of the complex.

        Parameters:
        -----------
        sort_with_filtration: bool, optional
            If True, the order will be sorted by the filtration values. Default is True.
        
        dtype: data-type, optional
            The data type of the border matrix. Default is int.
        self.essential_cells_dims
        Returns:
        --------
        border_matrix: np.array
            The border matrix of the complex, where the (i, j)-th entry is
            1 if the i-th cell is a face of the j-th cell, and 0 otherwise.
        """
        # Очень неэффективное решение, но должно работать!
        # This works incorrectly if self.shape contains n <= 2
        order, dims = self.get_order(map = lambda s: s if isinstance(s, EssentialCell) else set(utils.array_to_tuple(s)), 
                                     return_dims=True, sort_with_filtration=sort_with_filtration)
        order = np.array(order, dtype=object)
        def f(s0, s1):
            if isinstance(s0, EssentialCell) and isinstance(s1, EssentialCell):
                return False
            elif isinstance(s0, EssentialCell):
                return s0.has_border(s1) or s0.dim in [-1, self.dim + 1]
            elif isinstance(s1, EssentialCell):
                return s1.has_border(s0) or s1.dim in [-1, self.dim + 1]
            else:
                return (s0 & s1) == s0
        matrix = np.vectorize(f)(order.reshape(-1, 1), order.reshape(1, -1)) & (dims.reshape(-1, 1) - dims.reshape(1, -1) == -1)
        matrix = matrix.astype(dtype)

        return matrix

    def get_just_torus(self):
        """
        """
        return CubicalTorusComplex(self.shape).assign_filtration(self.filtration_values)