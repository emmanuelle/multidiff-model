import numpy as np
cimport numpy as np
cimport cython

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef inline int clip_warp_plus(int x, int L): return x if x<L else 0 
cdef inline int clip_warp_minus(int x, int L): return x if x>=0 else (L-1) 

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evolve_lattice(np.ndarray[np.uint8_t, ndim=2] lattice, 
                   np.ndarray[np.float64_t, ndim=2] rates, int nb_steps):
    """
    Evolution of a lattice of particles with different species undergoing
    a random walk, where the jump probability depends on the species of the
    neighbor of a particle. 

    Parameters
    ----------

    lattice: 2-D ndarray of uint8
        array of labels representing the particles types, of shape (L, l)
    rates : 2-D ndarray
        symmetric array of exchange rates between species: rates[i, j] is
        the exchange rate between species i and j. All coefficients should
        be smaller than 1 (interpreted as a probability).
    nb_steps : int
        number of diffusion steps to perform

    Returns
    -------
    save_lattices: 3-D array of uint8, of shape (nb_steps, L, l)
        lattice at all intermediate times.
    """
    cdef int L = lattice.shape[0]
    cdef int l = lattice.shape[1]
    cdef int step_index, perm_index
    cdef int i, j, i_move, j_move, at_1, at_2, direction
    cdef np.ndarray[np.uint8_t, ndim=3] save_lattices = np.empty((nb_steps, 
                                                    L, l), dtype=np.uint8)
    cdef np.ndarray[np.int64_t, ndim=1] perm
    cdef np.ndarray[np.float64_t, ndim=1] trans_rate
    cdef np.ndarray[np.uint8_t, ndim=2] directions = np.random.randint(0, 4,
                            L*l * nb_steps).reshape((nb_steps, L*l)).astype(np.uint8)
    cdef np.ndarray[np.int64_t, ndim=1] vert_move = np.array([0, 0, -1, 1],
                                                        dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] horiz_move = np.array([-1, 1, 0, 0],
                                                        dtype=np.int64)
    for step_index in range(nb_steps):
        save_lattices[step_index] = lattice
        perm = np.random.permutation(L*l)
        trans_rate = np.random.random(L*l)
        for perm_index in perm:
            i = perm_index / l
            j = perm_index - i * l
            direction = directions[step_index, perm_index]
            i_move = (i + vert_move[direction])
            i_move = clip_warp_plus(i_move, L)
            i_move = clip_warp_minus(i_move, L)
            j_move = (j + horiz_move[direction])
            j_move = clip_warp_plus(j_move, l)
            j_move = clip_warp_minus(j_move, l)
            at_1 = lattice[i, j]
            at_2 = lattice[i_move, j_move]
            if trans_rate[perm_index] < rates[at_1, at_2]:
                lattice[i, j] = at_2
                lattice[i_move, j_move] = at_1
    return save_lattices
