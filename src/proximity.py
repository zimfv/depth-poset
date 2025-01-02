import numpy as np

from src.depth import DepthPoset

from matplotlib import pyplot as plt

class Proximity:
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
		x = np.unique(np.concatenate([xmin, xmax]))
		y = np.unique(np.concatenate([ymin, ymax]))

		matrix = np.zeros([len(x) - 1, len(y) - 1])
		x_mid = 0.5*(x[1:] + x[:-1])
		y_mid = 0.5*(y[1:] + y[:-1])
		for i in range(len(xmin)):
			x_cond = (x_mid >= xmin[i])&(x_mid <= xmax[i])
			x_cond = x_cond.reshape([len(x_cond), 1])
			y_cond = (y_mid >= ymin[i])&(y_mid <= ymax[i])
			cond = np.logical_and(x_cond*np.ones(matrix.shape), y_cond*np.ones(matrix.shape))
			matrix[cond] += values[i]
		# очень херовое, костыльное решение. It's bas solution, should be rewritten.
		if (np.inf in x) and (-np.inf in x) and (np.inf in y) and (-np.inf in y):
			values = matrix
		else:
			values = np.zeros([len(x) + 1, len(y) + 1])
			values[1:-1, 1:-1] = matrix
		return Proximity(x, y, values)

	@classmethod
	def from_depth_poset(self, dp: DepthPoset):
		"""
		"""
		xmin, xmax, ymin, ymax = np.array([(e1.birth_value, e0.birth_value, e0.death_value, e1.death_value) for e0, e1 in dp.edges]).transpose()
		return Proximity.from_rectangles(xmin, xmax, ymin, ymax, values=1)

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
		x = np.unique(np.concatenate([self.x, other.x]))[1:-1]
		y = np.unique(np.concatenate([self.y, other.y]))[1:-1]

		x_grid = np.append(x - 0.5*(x[1:] - x[:-1]).min(), x[-1] + 1)
		y_grid = np.append(y - 0.5*(y[1:] - y[:-1]).min(), y[-1] + 1)
		y_grid, x_grid = np.meshgrid(y_grid, x_grid)
		values = self(x_grid, y_grid) + other(x_grid, y_grid)

		return Proximity(x, y, values)

	def __mul__(self, num):
		"""
		"""
		return Proximity(self.x, self.y, self.values*num)

	def __sub__(self, other):
		"""
		"""
		return self + (other*-1)

	def __abs__(self):
		"""
		"""
		return Proximity(self.x, self.y, abs(self.values))

	def __pow__(self, num):
		"""
		"""
		return Proximity(self.x, self.y, self.values**num)

	def min(self):
		"""
		"""
		return self.values.min()

	def max(self):
		"""
		"""
		return self.values.max()

	def show(self, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf, cmap='winter', alpha=None, vmin=None, vmax=None, ax=plt, ignore=0, **kwargs):
		"""
		"""
		cmap = plt.get_cmap(cmap)

		if vmin is None:
			vmin = self.min()
		if vmax is None:
			vmax = self.max()

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