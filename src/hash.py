import hashlib
import base64
import numpy as np


def boundary_matrix_to_tuples_list(delta):
    """
    """
    delta = np.asarray(delta, dtype=bool)
    return [tuple(map(int, np.flatnonzero(col))) for col in delta.transpose()]

def hash_tuples_list(arr):
    digest = hashlib.sha256(repr(tuple(arr)).encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

def hash_boundary_matrix(delta):
    """
    """
    return hash_tuples_list(boundary_matrix_to_tuples_list(delta))
