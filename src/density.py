'''
Here should be callable classes of density measure on R^2.
They could be defined from depth poset and used to define the metric between posets 
by taking L_p distance between density functions.
So these classes also should contain integral method.

Now there is only SquareDensity class is realised, but this gives an unstable metric.
'''


import numpy as np

from src.depth import DepthPoset

from src.planar_geometry import calculate_triangle_areas, is_triangle_containing_point

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable





class Density:
	pass


class SquareDensity(Density):
	def __init__(self, x, y, values):
		"""
		"""
		self.x = np.array(x)
		if (self.x != np.sort(self.x)).any():
			raise ValueError(f"The x-borders should be sorted.")
		if self.x[0] != -np.inf:
			self.x = np.append(-np.inf, self.x)
		if self.x[-1] != np.inf:
			self.x = np.append(self.x, np.inf)
		self.y = np.array(y)
		if (self.y != np.sort(self.y)).any():
			raise ValueError(f"The y-borders should be sorted.")
		if self.y[0] != -np.inf:
			self.y = np.append(-np.inf, self.y)
		if self.y[-1] != np.inf:
			self.y = np.append(self.y, np.inf)
		self.values = np.array(values)
		if self.values.shape != (len(self.x) - 1, len(self.y) - 1):
			raise ValueError(f"values-matrix shape {self.values.shape} does not correspond the borders lengths {len(self.x)} and {len(self.y)}.")

	@classmethod
	def from_rectangles(self, xmin, xmax, ymin, ymax, values=1):
		"""
		"""
		xmin, xmax, ymin, ymax = np.array([xmin, xmax, ymin, ymax])
		values = np.array(values)
		if values.shape == ():
			values = values*np.ones(xmin.shape)
		x = np.unique(np.concatenate([xmin, xmax, [-np.inf, +np.inf]]))
		y = np.unique(np.concatenate([ymin, ymax, [-np.inf, +np.inf]]))

		matrix = np.zeros([len(x) - 1, len(y) - 1])

		x_mid = 0.5*(x[1:] + x[:-1])
		y_mid = 0.5*(y[1:] + y[:-1])
		
		for i in range(len(xmin)):
			x_cond = (x_mid >= xmin[i])&(x_mid <= xmax[i])
			x_cond = x_cond.reshape([len(x_cond), 1])
			y_cond = (y_mid >= ymin[i])&(y_mid <= ymax[i])
			cond = np.logical_and(x_cond*np.ones(matrix.shape), y_cond*np.ones(matrix.shape))
			matrix[cond] += values[i]

		return SquareDensity(x, y, matrix)

	@classmethod
	def from_depth_poset(self, dp: DepthPoset):
		"""
		"""
		if len(dp.edges) == 0:
			xmin, xmax, ymin, ymax = [], [], [], []
		else:
			xmin, xmax, ymin, ymax = np.array([(e1.birth_value, e0.birth_value, e0.death_value, e1.death_value) for e0, e1 in dp.edges]).transpose()
		return SquareDensity.from_rectangles(xmin, xmax, ymin, ymax, values=1)

	def __call__(self, x, y):
		"""
		"""
		x = np.asarray(x)
		y = np.asarray(y)

		if x.shape != y.shape:
			raise ValueError(f'Arrays should have the same shape, but: x.shape={x.shape} and y.shape={y.shape}')
		if len(x.shape) != 1:
			return self(x.ravel(), y.ravel()).reshape(x.shape)

		# Создаем булевы массивы, определяющие, в какие интервалы попадают x и y
		x_mask = (self.x[:-1, None] <= x) & (self.x[1:, None] > x)
		y_mask = (self.y[:-1, None] <= y) & (self.y[1:, None] > y)

		# Находим индексы интервалов
		x_indices = np.argmax(x_mask, axis=0)
		y_indices = np.argmax(y_mask, axis=0)

		# Возвращаем соответствующие значения
		return self.values[x_indices, y_indices]

	def __add__(self, other):
		"""
		"""
		x = np.unique(np.concatenate([self.x, other.x]))
		y = np.unique(np.concatenate([self.y, other.y]))

		x_grid = 0.5*(x[1:] + x[:-1])
		y_grid = 0.5*(y[1:] + y[:-1])
		y_grid, x_grid = np.meshgrid(y_grid, x_grid)
		values = self(x_grid, y_grid) + other(x_grid, y_grid)

		return SquareDensity(x, y, values)

	def __mul__(self, num):
		"""
		"""
		return SquareDensity(self.x, self.y, self.values*num)

	def __sub__(self, other):
		"""
		"""
		return self + (other*-1)

	def __abs__(self):
		"""
		"""
		return SquareDensity(self.x, self.y, abs(self.values))

	def __pow__(self, num):
		"""
		"""
		return SquareDensity(self.x, self.y, self.values**num)

	def min(self):
		"""
		"""
		return self.values.min()

	def max(self):
		"""
		"""
		return self.values.max()

	def show(self, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf, cmap='Blues', alpha=None, vmin=None, vmax=None, ax=None, ignore=0, **kwargs):
		"""
		"""
		if ax is None:
			fig, ax = plt.subplots()

		cmap = plt.get_cmap(cmap)

		if vmin is None:
			vmin = self.min()
		if vmax is None:
			vmax = self.max()
		if vmin == vmax:
			if vmin == 0:
				vmin, vmax = 0, 1
			else:
				vmin, vmax = np.sort([0, vmin])

		x = np.concatenate([[xmin], self.x[(self.x > xmin)&(self.x < xmax)], [xmax]])
		y = np.concatenate([[ymin], self.y[(self.y > ymin)&(self.y < ymax)], [ymax]])

		for ix in range(len(x) - 1):
			for iy in range(len(y) - 1):
				x0, x1, y0, y1 = x[ix], x[ix + 1], y[iy], y[iy + 1]
				xval, yval = 0.5*(x0 + x1), 0.5*(y0 + y1)
				val = self(xval, yval)
				try:
					if not ignore(val):
						val = (val - vmin)/(vmax - vmin)
						col = cmap(val, alpha=alpha)
						ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=col, **kwargs)
				except TypeError:
					if val != ignore:
						val = (val - vmin)/(vmax - vmin)
						col = cmap(val, alpha=alpha)
						ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color=col, **kwargs)

		# create and return mappable object
		norm = Normalize(vmin=vmin, vmax=vmax)
		sm = ScalarMappable(cmap=cmap, norm=norm)
		return sm

	def integral(self, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf):
		"""
		"""
		x = np.unique(np.concatenate([self.x[(self.x >= xmin)&(self.x <= xmax)], [-np.inf, +np.inf]]))
		y = np.unique(np.concatenate([self.y[(self.y >= ymin)&(self.y <= ymax)], [-np.inf, +np.inf]]))

		x_mid, y_mid = np.meshgrid(0.5*(x[1:] + x[:-1]), 0.5*(y[1:] + y[:-1]))
		vals = self(x_mid, y_mid)

		x0, y0 = np.meshgrid(x[:-1], y[:-1])
		x1, y1 = np.meshgrid(x[1:], y[1:])
		squares = (x1 - x0)*(y1 - y0)

		if (vals[squares == np.inf] != 0).any():
			if (vals[squares == np.inf] >= 0).all():
				return +np.inf
			if (vals[squares == np.inf] <= 0).all():
				return -np.inf
			return np.nan

		return (vals[squares != np.inf]*squares[squares != np.inf]).sum()


class TriangulationDensity(Density):
	def __init__(self, vertices, triangles, values, outer_value=0):
		"""
		Parameters:
		-----------
		vertices: np.array shape (n, 2) dtype float
			The cords of n vertices on the plane

		triangles: np.array shape (k, 3) dtype int
			The vertex indices of k triangles

		values: array length k
			The density values in the triangle areas

		outer_value: float
			The density value out of triangles		
		"""
		self.vertices = np.array(vertices)
		self.triangles = np.array(triangles, dtype=int)
		self.values = np.array(values)
		self.outer_value = outer_value

		if self.triangles.max() + 1 > len(self.vertices):
			raise ValueError(f'The triangles contain more vertices than given')

	def __call__(self, x, y):
		"""
		"""
		x = np.asarray(x)
		y = np.asarray(y)
		if x.shape != y.shape:
			raise ValueError(f'x and y shoud be the same dimension, but: x.shape={x.shape}, y.shape={y.shape}')
		original_shape = x.shape
		x = x.reshape(-1)
		y = y.reshape(-1)

		triangles = self.vertices[self.triangles]
		triangle_areas = calculate_triangle_areas(triangles)
		triangles_contain_point = is_triangle_containing_point(triangles, x, y)

		res = np.ones(triangles_contain_point.shape)*self.values*triangle_areas
		res_area = np.ones(triangles_contain_point.shape)*triangle_areas

		res[~triangles_contain_point] = np.nan
		res_area[~triangles_contain_point] = 0

		res = res/res_area.sum(axis=1).reshape([res.shape[0], 1])
		res = np.nansum(res, axis=1)

		res = res.reshape(original_shape)

		return res

	def __add__(self, other):
		"""
		"""
		pass



	def min(self):
		"""
		"""
		return np.append(self.values, self.outer_value).min()

	def max(self):
		"""
		"""
		return np.append(self.values, self.outer_value).max()

	def show(self, cmap='Blues', alpha=None, vmin=None, vmax=None, ax=None, ignore=0, **kwargs):
		"""
		"""
		if ax is None:
			ax = plt.gca()

		cmap = plt.get_cmap(cmap)

		if vmin is None:
			vmin = self.min()
		if vmax is None:
			vmax = self.max()
		if vmin == vmax:
			if vmin == 0:
				vmin, vmax = 0, 1
			else:
				vmin, vmax = np.sort([0, vmin])

		for triangle, value in zip(self.triangles, self.values):
			x, y = self.vertices[np.append(triangle, triangle[0])].transpose()
			try:
				if not ignore(value):
					val = (value - vmin)/(vmax - vmin)
					col = cmap(val, alpha=alpha)
					ax.fill(x, y, color=col, **kwargs)
			except TypeError:
				if value != ignore:
					val = (value - vmin)/(vmax - vmin)
					col = cmap(val, alpha=alpha)
					ax.fill(x, y, color=col, **kwargs)

		# create and return mappable object
		norm = Normalize(vmin=vmin, vmax=vmax)
		sm = ScalarMappable(cmap=cmap, norm=norm)
		return sm
