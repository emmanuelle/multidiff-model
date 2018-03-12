from diffusion_model import diffusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import delayed, Parallel


rates = np.array([[0.05, 0.01, 0.9],
                  [0.01, 0.03, 0.1],
                  [0.9, 0.1, 0.1]])

conc_interv = np.linspace(0.1, 0.9, 20)
concs = []
n = len(conc_interv)
for i in range(n):
    for j in range(n):
        a = conc_interv[i]
        b = conc_interv[j]
        if a + b >= 0.95:
            continue
        concs.append([a, b, 1 - (a + b)])

concs = np.array(concs)

diags_all = []
eigvecs_all = []

res = Parallel(n_jobs=4)(delayed(diffusion_matrix)(base_conc, rates)
                                 for base_conc in concs)

diags = np.vstack([resi[0] for resi in res])
np.save('diags.npy', diags)
eigvecs = np.hstack([resi[1] for resi in res])
np.save('eigvecs.npy', eigvecs)

all_results = {}
all_results['rates'] = rates
all_results['concs'] = concs
all_results['diags'] = diags
all_results['eigvecs'] = eigvecs
import pickle
f = open('dump_scrip_diffusion_model.pkl', 'w')
pickle.dump(all_results, f)
f.close()
