import numpy as np

from scipy.spatial import Delaunay
import triangle


def triangulate_complex_numbers(zs, fix_perimeter=False):
    """
    Returns the triangulation of the set of complex numbers on the complex plane

    Parameters:
    -----------
    zs: array[complex]
        The ordered boundary

    fix_perimeter: bool
        Adds the constrain, that there should be edges, connecting neihbour numbers in the order, if it's True
        If Flase - just compute Delaunay triangulation
        We shoud fix the perimeter, but this can kill the kernel, so we refused to do that and set the parameter False.

    Returns:
    --------
    tri: list(tuple)
        The triples of indices, which are the triangle vertices
    """
    points = np.transpose([np.real(zs), np.imag(zs)])
    if not fix_perimeter:
        tri = Delaunay(points)
        tri = tri.simplices
    else:
        segments = np.transpose([np.arange(len(zs)), (np.arange(len(zs)) + 1)%len(zs)])
        segments = np.sort(segments, axis=1)
        tri = triangle.triangulate(dict(vertices=points, segments=segments), "p")
        assert (tri['vertices'] == points).all()
        tri = tri['triangles']
    return tri


def get_triangle_representation(z, z1, z2, z3):
    """
    Represents the complex number z as linear combination of 3 given numbers z1, z2, z3

    Parameters:
    -----------
    z: complex array
        A complex number or an array of complex numbers to be represented in the triangle
    
    z1, z2, z3: complex numbers
        Vertices of the triangle
    
    Returns:
    --------
    t: np.array
        An array of shape (z.shape) + (3,) with coefficients for the representation of `z` in the triangle defined by `z1`, `z2`, `z3`
    """
    z = np.asarray(z)
    a1 = np.real(z2) - np.real(z1)
    b1 = np.real(z3) - np.real(z1)
    c1 = np.real(z) - np.real(z1)
    a2 = np.imag(z2) - np.imag(z1)
    b2 = np.imag(z3) - np.imag(z1)
    c2 = np.imag(z) - np.imag(z1)
    det = a1 * b2 - a2 * b1
    if np.any(det == 0):
        raise ValueError("The triangle is degenerate")
    t3 = (a1*c2 - a2*c1) / det
    t2 = (b2*c1 - b1*c2) / det
    t1 = 1 - t2 - t3
    return np.array([t1, t2, t3]).transpose(tuple(np.arange(z.ndim) + 1) + (0, ))


def map_triangle(z, z1_input, z2_input, z3_input, z1_output, z2_output, z3_output):
    """
    Represents the number z as linear combination of input triangle and linearly map this as represented in the output triangle

    Parameters:
    -----------
    z: complex or complex array
        The complex argument
    
    z1_input, z2_input, z3_input: complex
        The vertices of the input triangle
    
    z1_output, z2_output, z3_output: complex
        The vertices of the output triangle

    Returns:
    --------

    """
    t = get_triangle_representation(z, z1_input, z2_input, z3_input)
    z_res = t@np.array([z1_output, z2_output, z3_output], dtype='complex')
    return z_res


def get_map_boundary_to_unit_circle(zs, angles=None):
    """
    Returns the function from complex to complex maping the given boundary to the unit cicrle (circumscribed unit circle polygon).
    This triangulates the boundary. Maps each vertex of the boundary to the unit circle, 
    and then maps numbers in these triangles as their linear representations 

    Parameters:
    -----------
    zs: array[complex]
        The ordered boundary
    
    angles: array[float]
        The angles coresponding the points of boundary.
        Should be the same length as zs
        This will be defined automatically, if it's not given

    Returns:
    --------
    map_boundary_to_unit_circle: function
        Maps the boundary to a unit circle in the complex space 
    """
    if angles is None:
        angles = 2*np.pi*np.arange(len(zs))/len(zs)
    angles %= 2*np.pi

    triangles = triangulate_complex_numbers(zs)
    triangle_inputs = np.array([zs[tri] for tri in triangles])
    triangle_outputs = np.array([np.exp(1j*angles[tri]) for tri in triangles])

    def map_boundary_to_unit_circle(z):
        z = np.asarray(z)
        z_work = z.reshape(-1)
        z_res = np.nan*z_work
        for (z1_input, z2_input, z3_input), (z1_output, z2_output, z3_output) in zip(triangle_inputs, triangle_outputs):
            m = get_triangle_representation(z_work, z1_input, z2_input, z3_input)
            m = ((m >= 0) & (m <= 1)).all(axis=-1)
            z_res[m] = map_triangle(z_work[m], z1_input, z2_input, z3_input, z1_output, z2_output, z3_output)
        z_res = z_res.reshape(z.shape)
        
        return z_res
    return map_boundary_to_unit_circle


def get_map_changing_center(z0, alpha=0):
    """
    Returns the complex function, changing the center of the unit disk, but preserving the boundary unit circle:

    .. math::

        f(z) = (1-t(z))\cdot(z - z_0)\cdot e^{i\alpha} + t(z)\cdot z
    
    where the parameter $t(z) = \frac{|z - z_0|}{|e^{i\cdot \arg(z - z_0)} - z_0|}$.

    Parameters:
    -----------
    z0: complex
        The new center of the unit disk
    
    alpha: float
        The angle to rotate the neighbourhood arround the center

    Returns:
    --------
    map_changing_center: function
        The complex function changing the center of the unit disk
    """
    def map_changing_center(z):
        z = np.asarray(z)
        t = np.abs(z - z0)/np.abs(np.exp(1j*np.angle(z - z0)) - z0)
        res = t*z + (1 - t)*(z - z0)*np.exp(1j*alpha)
        return res
    return map_changing_center


def get_map_disk_to_unit(z0, zs, angles=None):
    """
    Returns the function maping disk to the unit disk and seting the center

    Parameters:
    -----------
    z0: complex
        The new center of the unit disk
    
    zs: array[complex]
        The ordered boundary
    
    angles: array[float]
        The angles coresponding the points of boundary.
        Should be the same length as zs
        This will be defined automatically, if it's not given

    Returns:
    --------
    map_disk_to_unit: function
    """
    if angles is None:
        angles = 2*np.pi*np.arange(len(zs))/len(zs)

    alpha = np.angle(zs - z0)[np.argmin(angles%(2*np.pi))]
    f1 = get_map_boundary_to_unit_circle(zs, angles)    
    f2 = get_map_changing_center(f1(z0), alpha)
    def map_disk_to_unit(z):
        return f2(f1(z))
    return map_disk_to_unit