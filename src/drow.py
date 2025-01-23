import numpy as np
import gudhi as gh

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_simplex(simplex, points, color, ax=None, zorder=0, 
				 pointwidth=1, marker='o', linewidth=2, linestyle='-', 
				 hatch=None, facecolor='none', 
				 label=None):
	"""
	Plot the simplex on 2-dimensional plane.
	"""
	if ax is None:
		ax = plt.gca()

	if len(simplex) == 1:
		x, y = np.transpose(points[simplex]*np.ones([1, 2]))
		ax.scatter(x, y, color=color, zorder=2 + zorder, 
				   linewidth=pointwidth, marker=marker, 
				   label=label)
	if len(simplex) == 2:
		x, y = np.transpose(points[simplex]*np.ones([2, 2]))
		ax.plot(x, y, color=color, zorder=1 + zorder, 
				linewidth=linewidth, linestyle=linestyle, 
				label=label)
	if len(simplex) == 3:
		x, y = np.transpose(points[simplex]*np.ones([3, 2]))
		if hatch is None:
			ax.fill(x, y, color=color, zorder=0 + zorder, 
					label=label)
		else:
			ax.fill(x, y, zorder=0 + zorder, 
					facecolor=facecolor, hatch=hatch, edgecolor=color, 
					linestyle=linestyle, linewidth=linewidth,
					label=label)

def plot_filtred_complex2d(stree: gh.SimplexTree, points, cmap='viridis', ax=None, vmin=None, vmax=None, zorder=0, 
						   pointwidth=1, linewidth=2):
	cmap = plt.get_cmap(cmap)

	if vmin is None:
		vmin = min([value for simplex, value in stree.get_filtration()])
	if vmax is None:
		vmax = max([value for simplex, value in stree.get_filtration()])

	norm = Normalize(vmin=vmin, vmax=vmax)
	sm = ScalarMappable(cmap=cmap, norm=norm)
	
	for simplex, value in stree.get_filtration():
		color = cmap(norm(value))
		plot_simplex(simplex, points, color, ax=ax, zorder=zorder, 
					 pointwidth=pointwidth, linewidth=linewidth, 
					 hatch=None)
	return sm