To start using this code, first you need to compile the cython file

python setup.py build_ext --inplace

For fitting the diffusion matrix, you need to install the (SVI-made)
multidiff package :

pip install multidiff

Then you can execute one of the scripts: in IPython / jupyter:

run script_lattice.py for plotting concentration profiles
(you don't need multidiff for this one)

run script_fit_matrix to compute the diffusion matrix

