import itertools
import numpy as np


def calculate_triangle_areas(triangles):
	"""
	Returns the array of triangle areas

	Paamaters:
	----------
	triangles: np.array shape (shape, 3, 2)
		The cords of the triangles

	Returns:
	--------
	areas : np.array shape shape, dtype float
		The areas of the triangles
	"""
	# Extract coordinates
	x = triangles[..., :, 0]  # Shape: (shape, 3)
	y = triangles[..., :, 1]  # Shape: (shape, 3)

	# Compute signed area
	areas = 0.5 * np.abs(
		x[..., 0] * (y[..., 1] - y[..., 2]) +
		x[..., 1] * (y[..., 2] - y[..., 0]) +
		x[..., 2] * (y[..., 0] - y[..., 1])
	)
	return areas

def is_triangle_containing_point(triangles, x, y):
	"""
	Determines whether each point (x, y) lies inside each triangle.
	
	Parameters:
	-----------
	triangles: np.ndarray of shape (..., 3, 2)
		Triangles defined by three (x, y) vertices.
	
	x: np.ndarray of shape (...)
		X-coordinates of points to test.

	y: np.ndarray of shape (...)
		Y-coordinates of points to test.

	Returns:
	--------
	inside : np.ndarray of shape (..., ...)
		Boolean array indicating if each point is in each triangle.
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	
	# reshaping to vectors
	if x.shape != y.shape:
		raise ValueError(f'x and y should have same dimension, but x.shape={x.shape}, y.shape={y.shape}')
	shape_p = x.shape
	x = x.reshape(-1)
	y = y.reshape(-1)

	shape_tri = triangles.shape
	triangles = triangles.reshape(-1, 3, 2)
	
	# Points array of shape (n_p, 2)
	points = np.stack([x, y], axis=-1)
	
	# Number of triangles (n_tri) and points (n_p)
	n_tri = triangles.shape[0]
	n_p = len(x)
	
	# Reshape triangles to (n_tri, 3, 2)
	A = triangles[:, 0, :]
	B = triangles[:, 1, :]
	C = triangles[:, 2, :]
	
	# Vectors from points to triangle vertices (n_p, n_tri, 2)
	v0 = B - A
	v1 = C - A
	v2 = points[:, np.newaxis, :] - A  # Shape (n_p, 1, 2) then broadcast to (n_p, n_tri, 2)

	# Compute the dot products using broadcasting
	d00 = np.sum(v0 * v0, axis=-1)  # (n_tri)
	d01 = np.sum(v0 * v1, axis=-1)  # (n_tri)
	d11 = np.sum(v1 * v1, axis=-1)  # (n_tri)
	d20 = np.sum(v2 * v0, axis=-1)  # (n_p, n_tri)
	d21 = np.sum(v2 * v1, axis=-1)  # (n_p, n_tri)
	
	# Compute the barycentric coordinates
	denom = d00 * d11 - d01 * d01  # (n_tri)
	v = (d11 * d20 - d01 * d21) / denom#[:, np.newaxis]  # (n_p, n_tri)
	w = (d00 * d21 - d01 * d20) / denom#[:, np.newaxis]  # (n_p, n_tri)
	u = 1 - v - w  # (n_p, n_tri)

	# Point is inside triangle if all barycentric coordinates are non-negative
	inside = (u >= 0) & (v >= 0) & (w >= 0)

	# reshaping to original shape
	new_shape = np.concatenate([shape_p, shape_tri[:-2]]).astype(int)
	inside = inside.reshape(new_shape)

	return inside

def get_lines_intersections(a_points, a_directions, b_points, b_directions):
	"""
	Returns the intersections of lines.
	
	Parameters:
	-----------
	a_points : np.array shape (n, 2)
		The points, defining the lines a
		
	a_directions : np.array shape (n, 2)
		The direction of the lines a
		
	b_points : np.array shape (n, 2)
		The points, defining the lines b
		
	b_directions : np.array shape (n, 2)
		The direction of the lines b

	Returns:
	--------
	c : np.array shape (n, 2)
		The intersections of line pairs a, b
	"""
	a_points = np.array(a_points)
	b_points = np.array(b_points)
	a_directions = np.array(a_directions)
	b_directions = np.array(b_directions)
	if (a_points.shape != b_points.shape) or (a_points.shape != a_directions.shape) or (a_directions.shape != b_directions.shape):
		raise ValueError(f'The perameter dimensions should be equal, but:\na_points.shape={a_points.shape}; a_directions.shape={a_directions.shape}\nb_points.shape={b_points.shape}; b_directions.shape={b_directions.shape}')
	original_shape = a_points.shape
	a_points = a_points.reshape(-1, 2)
	b_points = b_points.reshape(-1, 2)
	a_directions = a_directions.reshape(-1, 2)
	b_directions = b_directions.reshape(-1, 2)

	# Differences between points
	delta_points = b_points - a_points
	
	# Build the determinant matrix for each line pair
	det_matrix = np.stack((a_directions, -b_directions), axis=2)
	det_values = np.linalg.det(det_matrix)  # Determinants for all line pairs
	
	# Find indices where lines are not parallel (det ≠ 0)
	non_parallel = ~np.isclose(det_values, 0)
	
	# Initialize result with NaN
	c = np.full_like(a_points, np.nan, dtype=float)
	
	# Solve for t where lines are not parallel
	det_inv = np.linalg.pinv(det_matrix)  # Pseudo-inverse for solving systems
	params = np.einsum('ijk,ik->ij', det_inv, delta_points)  # Parametric solutions
	
	# Compute intersection points for valid cases
	c[non_parallel] = a_points[non_parallel] + params[non_parallel, 0:1] * a_directions[non_parallel]
	
	c = c.reshape(original_shape)

	return c
	
def segments_intersect(a0, a1, b0, b1):
	"""
	Determines if pairs of line segments intersect.

	Parameters:
	-----------
	a0 : np.array shape (n, 2)
		The starting points of segment a.
		
	a1 : np.array shape (n, 2)
		The ending points of segment a.
		
	b0 : np.array shape (n, 2)
		The starting points of segment b.
		
	b1 : np.array shape (n, 2)
		The ending points of segment b.

	Returns:
	--------
	c : np.array shape (n,) dtype bool
		True if the segment pairs a, b intersect, otherwise False.
	"""
	a0 = np.array(a0)
	a1 = np.array(a1)
	b0 = np.array(b0)
	b1 = np.array(b1)
	if (a0.shape != a1.shape) or (a0.shape != b0.shape) or (b0.shape != b1.shape):
		raise ValueError(f'The perameter dimensions should be equal, but:\na0.shape={a0.shape}; a1.shape.shape={a1.shape.shape}\nb0.shape.shape={b0.shape.shape}; b1.shape.shape={b1.shape.shape}')
	original_shape = a0.shape
	a0 = a0.reshape(-1, 2)
	a1 = a1.reshape(-1, 2)
	b0 = b0.reshape(-1, 2)
	b1 = b1.reshape(-1, 2)

	# Vectorized helper functions
	def cross(v1, v2):
		"""Compute the 2D cross product."""
		return v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

	def is_between(p, q, r):
		"""Check if point q is on the segment pr."""
		cross_product = cross(q - p, r - p)
		dot_product = np.sum((q - p) * (r - p), axis=1)
		squared_length = np.sum((r - p) ** 2, axis=1)
		return np.isclose(cross_product, 0) & (dot_product >= 0) & (dot_product <= squared_length)

	# Direction vectors for each segment
	d1 = a1 - a0
	d2 = b1 - b0

	# Compute cross products to determine relative orientation
	d0_cross_d2 = cross(b0 - a0, d2)
	d1_cross_d2 = cross(d1, d2)

	# Handle parallel and collinear cases
	parallel = np.isclose(d1_cross_d2, 0)
	collinear = parallel & np.isclose(d0_cross_d2, 0)

	# For collinear segments, check overlap
	collinear_intersects = (
		collinear &
		(is_between(a0, b0, a1) | is_between(a0, b1, a1) | is_between(b0, a0, b1) | is_between(b0, a1, b1))
	)

	# For non-parallel segments, compute intersection parameters
	t = d0_cross_d2 / d1_cross_d2
	u = cross(b0 - a0, d1) / d1_cross_d2

	# Check if the intersection parameters fall within segment bounds
	proper_intersects = ~parallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

	# Combine results
	c = proper_intersects | collinear_intersects

	c = c.reshape(original_shape[:-1])

	return c

def segment_contains_point(a0, a1, p, include_ends=False):
	"""
	Returns does the given segment contain the point

	Parameters:
	-----------
	a0: np.array shape (..., 2)
		The ends of the segments

	a1: np.array shape (..., 2)
		The ends of the segments

	p: np.array shape (..., 2)
		The cords of the points

	include_ends: bool
		Include segment ends if this is True

	Returns:
	--------
	res: np.array shape (..., ...) dtype bool
	"""
	a0 = np.asarray(a0)
	a1 = np.asarray(a1)
	p = np.asarray(p)

	if (a0.shape != a1.shape):
		msg = f'The segment ends dimensions should be equal, but:\na0.shape={a0.shape}; a1.shape.shape={a1.shape.shape}'
		raise ValueError(msg)
	if (a0.shape[-1] != 2) or (p.shape[-1] != 2):
		msg = f'The points and segment ends should be on 2-dimensional plane (i.e. the last dimension should be equal 2), but a0.shape={a0.shape}, a1.shape={a1.shape}, p.shape={p.shape}'
		raise ValueError(msg)
	a_shape = a0.shape
	p_shape = p.shape
	a0 = a0.reshape(-1, 2)
	a1 = a1.reshape(-1, 2)
	p = p.reshape(-1, 2)

	a_n = len(a0)
	p_n = len(p)

	a0 = np.ones([a_n, p_n, 2])*a0.reshape([a_n, 1, 2])
	a1 = np.ones([a_n, p_n, 2])*a1.reshape([a_n, 1, 2])
	p = np.ones([a_n, p_n, 2])*p

	# I suppose this is not the most accurate solution
	res = (np.linalg.norm(a0 - p, axis=2) + np.linalg.norm(a1 - p, axis=2)) == np.linalg.norm(a0 - a1, axis=2) 
	if not include_ends:
		res = res & (a0 != p).any(axis=2) & (a1 != p).any(axis=2)
	res = res.reshape(np.concatenate([a_shape[:-1], p_shape[:-1], ]).astype(int))
	return res


def get_uncrossed_segments(points, segments, decimals=12):
	"""
	For given points and segments on plane returns new pack of points and segments,
	such that there the crosses devide segments to new 4.

	P.s.: This should work only for generic cases.
	
	Parameters:
	-----------
	points: np.array shape (n, 2)
		Points coordinates

	segments: np.array shape (k, 2) dtype=int
		Indices of segments ends

	decimals: int
		Number of decimal places to round to (default: 12). If decimals is negative, 
		it specifies the number of positions to the left of the decimal point.

	Returns:
	--------
	new_points: np.array shape (new_n, 2)
		Points coordinates

	new_segments: np.array shape (new_k, 2) dtype=int
		Indices of segments ends
	"""
	# define pairs
	segment_pairs = np.array(list(itertools.combinations(segments, 2)))
	segment_pairs_ends = points[segment_pairs]

	# find crossing segments, which crosses are not their ends
	nontrivial_cross = segments_intersect(segment_pairs_ends[:, 0, 0], segment_pairs_ends[:, 0, 1], 
										  segment_pairs_ends[:, 1, 0], segment_pairs_ends[:, 1, 1])
	nontrivial_cross = nontrivial_cross & (segment_pairs[:, 0, 0] != segment_pairs[:, 1, 0])
	nontrivial_cross = nontrivial_cross & (segment_pairs[:, 0, 0] != segment_pairs[:, 1, 1])
	nontrivial_cross = nontrivial_cross & (segment_pairs[:, 0, 1] != segment_pairs[:, 1, 0])
	nontrivial_cross = nontrivial_cross & (segment_pairs[:, 0, 1] != segment_pairs[:, 1, 1])

	# find new points - add crosses
	crossing_segment_pairs = segment_pairs[nontrivial_cross]
	points_a0 = points[crossing_segment_pairs[:, 0, 0]]
	points_b0 = points[crossing_segment_pairs[:, 1, 0]]
	points_b1 = points[crossing_segment_pairs[:, 1, 1]]
	points_a1 = points[crossing_segment_pairs[:, 0, 1]]
	
	new_points = get_lines_intersections(points_a0, points_a1 - points_a0, points_b0, points_b1 - points_b0)
	new_points = np.concatenate([points, new_points])

	# define which new point indices devide old segments
	changing_segments = {}
	for edge in np.concatenate(crossing_segment_pairs):
		changing_segments.update({tuple(np.sort(edge)): []})
	for i, (edge0, edge1) in enumerate(crossing_segment_pairs):
		changing_segments[tuple(np.sort(edge0))].append(i + len(points))
		changing_segments[tuple(np.sort(edge1))].append(i + len(points))

	def sort_by_distances(i0, i1, i_between=[]):
		# sort list of indices s.t. the coresponding points are consistently located between points with indices i0 and i1
		if len(i_between) == 0:
			return [i0, i1]
		i_min = i_between[0]
		for i in i_between:
			if np.linalg.norm(new_points[i0] - new_points[i]) < np.linalg.norm(new_points[i0] - new_points[i_min]):
				i_min = i
		return np.append(i0, sort_by_distances(i0=i_min, i1=i1, i_between=list(set(i_between) - {i_min})))

	# reorder list of segment dividers
	for i0, i1 in changing_segments.keys():
		changing_segments[(i0, i1)] = sort_by_distances(i0, i1, i_between=changing_segments[(i0, i1)])
	
	# find new segments
	if len(changing_segments) == 0:
		segments_to_remove = np.zeros([0, 2])
		segments_to_add = np.zeros([0, 2])
	else:
		segments_to_remove = np.sort(list(changing_segments.keys()), axis=1)
		segments_to_add = [np.transpose([value[:-1], value[1:]]) for value in changing_segments.values()]
		segments_to_add = np.concatenate(segments_to_add)
		segments_to_add = np.sort(segments_to_add, axis=1)

	# filter and sort new segments
	new_segments = np.sort(segments, axis=1)
	new_segments = new_segments[~np.any(np.all(new_segments[:, None] == segments_to_remove, axis=2), axis=1)]
	new_segments = np.concatenate([new_segments, segments_to_add], axis=0)
	
	# reindexing
	new_points = new_points.round(decimals) # rounding
	new_points_reindexed = np.unique(new_points, axis=0)
	
	reindex_dict = {i: np.where((new_points == new_points_reindexed[[i]]).all(axis=1))[0] for i in range(len(new_points_reindexed))}
	new_segments_reindexed = np.empty(new_segments.shape, dtype=int)
	for new_index in reindex_dict:
		for old_index in reindex_dict[new_index]:
			new_segments_reindexed[new_segments == old_index] = new_index

	#return new_points, new_segments
	return new_points_reindexed, new_segments_reindexed