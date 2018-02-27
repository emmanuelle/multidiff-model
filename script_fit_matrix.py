import numpy as np
from _lattice import evolve_lattice
from lattice import create_lattice
import matplotlib.pyplot as plt
from multidiff import compute_diffusion_matrix

def symmetrize(arr):
    L = len(arr)
    new_arr = np.copy(arr[L/4:3*L/4])
    new_arr[:L/4] += arr[:L/4][::-1]
    new_arr[L/4:] += arr[3*L/4:][::-1]
    return new_arr/2.

# exchange_rates
rates =  np.array([[0.05, 0.01, 0.9],
                  [0.01, 0.03, 0.1],
                  [0.9, 0.1, 0.1]])

L = 200
l = 20000

base_conc = np.array([0.5, 0.25, 0.25])
alpha = 0.03
exchanges = np.array([[0, -1, 1],
                      [1, -1, 0],
                      [1, 0, -1]])

all_concs = []
x_point = np.arange(L/2) - L/4
x_points = (x_point,)*3

for exch in exchanges:
    concs_1 = base_conc + alpha * exch
    concs_2 = base_conc - alpha * exch
    lattice = create_lattice((L, l), concs_1, concs_2)
    nb_steps = 200
    save_lattices = evolve_lattice(lattice, rates, nb_steps=nb_steps)
    si = symmetrize((save_lattices[-1] == 0).mean(axis=1))
    al = symmetrize((save_lattices[-1] == 1).mean(axis=1))
    ca = symmetrize((save_lattices[-1] == 2).mean(axis=1))
    all_concs.append(np.vstack((si, al, ca)))


diags_init = np.array([1, 1])
P_init = np.eye(2)
diags_res, eigvecs, _, _, _ = compute_diffusion_matrix((diags_init, P_init), 
                               x_points,
                               all_concs, plot=True,
                               labels=['Si', 'Al', 'Ca'])

