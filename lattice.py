import numpy as np

def create_lattice(shape, concs_1, concs_2):
    """
    Create a lattice to be used for diffusion. At the moment only 3 species
    are supported.

    Parameters
    ----------
    shape: tuple
        shape (L, l) of the lattice. Typically you want a lot more columns
        than rows so that you can average the concentration over the columns
        axis. L should be a multiple of 2.
    concs_1: array_like of length 3
        concentrations of the upper half
    concs_2: array_like of length 3
        concentrations of the lower half
    """
    L, l = shape
    concs_1_cum = np.cumsum(concs_1)
    concs_2_cum = np.cumsum(concs_2)
    # initial network
    grid_1 = np.random.random((l*L)/2).reshape((L/2, l))
    lattice_1 = np.empty_like(grid_1, dtype=np.uint8)
    lattice_1[grid_1 < concs_1_cum[0]] = 0
    lattice_1[np.logical_and(grid_1>=concs_1_cum[0], grid_1<concs_1_cum[1])] = 1
    lattice_1[grid_1 >= concs_1_cum[1]] = 2
    grid_2 = np.random.random((l*L)/2).reshape((L/2, l))
    lattice_2 = np.empty_like(grid_2, dtype=np.uint8)
    lattice_2[grid_2 < concs_2_cum[0]] = 0
    lattice_2[np.logical_and(grid_2>=concs_2_cum[0], grid_2<concs_2_cum[1])] = 1
    lattice_2[grid_2 >= concs_2_cum[1]] = 2
    lattice = np.vstack((lattice_1, lattice_2))
    return lattice
