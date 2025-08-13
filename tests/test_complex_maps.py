import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from src.cplane import get_map_disk_to_unit


from skimage import measure


def extract_ordered_boundary(mask, return_countor: bool=False):
    """
    Exclude the ordered boundary as list of tuples 

    Parameters:
    -----------
    mask: np.array ndim=2
        The binary mask
    
    return_countor: bool:
        Returns the float coordinates between pixels if True
        Returns the coordinates of the pixesl if False

    Returns:
    --------
    ordered_boundary: list[tuple]
        coordinates of the boundary
    """
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("Could not find a border")
    
    # Take the longest border if there are many
    contour = max(contours, key=len)
    if return_countor:
        return contour

    # Choose the border pixels
    ordered_boundary = [(int(round(y)), int(round(x))) for y, x in contour]
    ordered_boundary = [(y, x) for y, x in contour]
    ordered_boundary = []
    for y, x, in contour:
        if y%1 == 0:
            y, x = int(y), int(x)
            if not mask[y, x]:
                x += 1
        else:
            y, x = int(y), int(x)
            if not mask[y, x]:
                y += 1
        if (y, x) not in ordered_boundary:
            ordered_boundary.append((y, x))
    

    return ordered_boundary


def test1():
    zs = np.array([-2j, 2-1j, 3+1j, 2+3j, 2j, -2+3j, -1+1j, -2-1j])
    z_center = 1 - 1j

    f1 = get_map_disk_to_unit(z_center, zs)


    x = np.linspace(-4, 4, 101)
    y = np.linspace(-4, 4, 101)
    z0 = x.reshape(1, -1) + 1j*y.reshape(-1, 1)
    z1 = f1(z0)

    zx = np.append(zs, zs[0]).real
    zy = np.append(zs, zs[0]).imag
    zx = np.searchsorted(x, zx)
    zy = np.searchsorted(y, zy)


    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for (i_col, repr), (i_row, z) in itertools.product(enumerate([np.real, np.imag, np.abs, np.angle]), enumerate([z0, z1])):
        axs[i_row, i_col].set_title(f'{repr.__name__}')
        img = repr(z)
        cmap = plt.get_cmap().copy()
        cmap.set_bad(color='gray')
        sm = axs[i_row, i_col].imshow(img, cmap=cmap, origin='lower')
        plt.colorbar(sm, ax=axs[i_row, i_col])

        axs[i_row, i_col].plot(zx, zy, color='red', linestyle='--')
        axs[i_row, i_col].set_xticks(np.linspace(0, len(x)-1, 5).astype(int), x[np.linspace(0, len(x)-1, 5).astype(int)])
        axs[i_row, i_col].set_yticks(np.linspace(0, len(y)-1, 5).astype(int), y[np.linspace(0, len(y)-1, 5).astype(int)])
        axs[i_row, i_col].grid(True)

    plt.tight_layout()
    plt.show()


def test2():
    mask = np.zeros([16, 12])
    mask[2, 3:6] = 1
    mask[3, 2:9] = 1
    mask[4, 3:10] = 1
    mask[5:8, 8] = 1
    mask[5:11, 4:8] = 1
    mask[8:11, 3] = 1
    mask[11, 2:9] = 1
    mask[12, 3:10] = 1
    mask[13, 6:9] = 1

    mask = ndimage.zoom(mask, (4, 4), order=1, grid_mode=True).astype(bool).astype(int)
    #mask = ndimage.zoom(mask, (8, 8), order=1, grid_mode=True).astype(bool).astype(int)


    center_x, center_y = np.argwhere(mask)[np.random.randint(mask.sum())]

    boundary = extract_ordered_boundary(mask, return_countor=True)
    boundary_x, boundary_y = np.transpose(boundary)
    boundary_angles = np.arange(len(boundary))/len(boundary) * 2*np.pi


    x = np.arange(mask.shape[0])
    y = np.arange(mask.shape[1])
    z = x.reshape(1, -1) + 1j*y.reshape(-1, 1)

    zs = boundary_x + 1j*boundary_y
    z_center = center_x + 1j*center_y
    
    zx = np.append(zs, zs[0]).real
    zy = np.append(zs, zs[0]).imag
    zx = np.searchsorted(x, zx)
    zy = np.searchsorted(y, zy)

    f = get_map_disk_to_unit(z_center, zs)

    cmap = plt.get_cmap().copy()
    cmap.set_bad(color='gray')

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for (i_col, repr), (i_row, z) in itertools.product(enumerate([np.real, np.imag, np.abs, np.angle]), enumerate([z, f(z)])):
        axs[i_row, i_col].set_title(f'{repr.__name__}')
        img = repr(z)
        
        sm = axs[i_row, i_col].imshow(img, cmap=cmap, origin='lower')
        plt.colorbar(sm, ax=axs[i_row, i_col])

        axs[i_row, i_col].plot(zx, zy, color='red', linestyle='--', zorder=1)
        axs[i_row, i_col].scatter(center_x, center_y, color='red', marker='*', zorder=2)
        axs[i_row, i_col].set_xticks(np.linspace(0, len(x)-1, 5).astype(int), x[np.linspace(0, len(x)-1, 5).astype(int)])
        axs[i_row, i_col].set_yticks(np.linspace(0, len(y)-1, 5).astype(int), y[np.linspace(0, len(y)-1, 5).astype(int)])
        axs[i_row, i_col].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #test1()
    test2() # there is some core issue