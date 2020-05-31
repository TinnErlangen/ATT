from cnx_utils import load_sparse, phi, TriuSparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from cnx_utils import pw_cor_dist
import pickle
plt.ion()

proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

freq = "gamma_2"
conds = ["rest", "audio", "visual", "visselten"]
n_jobs = 8
epo_avg = True

dPTEs = []
sub_inds = []
cond_inds = []
for sub in subjs:
    for cond in conds:
            dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                               cond, freq))
            if epo_avg:
                dPTE = dPTE.mean(axis=0, keepdims=True)
            dPTEs.append(dPTE)
            sub_inds += [sub for idx in range(len(dPTE))]
            cond_inds += [cond for idx in range(len(dPTE))]
labels = {"sub":sub_inds, "cond":cond_inds}
dPTE = np.vstack(dPTEs)
mat_n = len(dPTE)
print("Creating and saving pairwise distance matrices...")
# make the indices of the chunks used for parallel
comb = np.array(np.triu_indices(mat_n,k=1),dtype="int32").T
a = np.arange(0,len(comb),len(comb)//n_jobs).astype("int32")
b = a + len(comb)//n_jobs
b[-1] += len(comb)%n_jobs
inds = tuple(zip(a,b))

#send them off for parallel processing
results = Parallel(n_jobs=n_jobs, verbose=100)(delayed(pw_cor_dist)
                  (dPTE,comb[idx[0]:idx[1]]) for idx in inds)
results = np.hstack(results)
triu_inds = np.triu_indices(mat_n,k=1)
out_mat = np.zeros((mat_n,mat_n),dtype="float32")
out_mat[triu_inds[0],triu_inds[1]] = results
del results
out_mat = TriuSparse(out_mat)
out_mat.save("{}{}_dist.sps".format(mat_dir,freq))
with open("{}{}_dist.labels".format(mat_dir,freq), "wb") as f:
    pickle.dump(labels, f)
