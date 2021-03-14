import hdbscan
import numpy as np
import pickle
from cnx_utils import load_sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

precomputed = False
epo_avg = False
proc_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
conds = ["rest", "audio", "visual", "visselten"]
freq = "alpha_1"

if precomputed:
    metric = "precomputed"
    distmat = load_sparse("{}{}_dist.sps".format(proc_dir,filename))
    distmat += distmat.T
    distmat = 1 - distmat
    inmat = distmat
    with open("{}{}_dist.labels".format(proc_dir,filename),"rb") as f:
        labels = pickle.load(f)
else:
    metric = "euclidean"
    dPTEs = []
    sub_inds = []
    cond_inds = []
    for sub in subjs:
        for cond in conds:
                dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                   cond, freq))
                if epo_avg:
                    dPTE = dPTE.mean(axis=0, keepdims=True)
                triu_inds = np.triu_indices(dPTE.shape[-1],k=1)
                dPTE = dPTE[...,triu_inds[0],triu_inds[1]]
                dPTEs.append(dPTE)
                sub_inds += [sub for idx in range(len(dPTE))]
                cond_inds += [cond for idx in range(len(dPTE))]
    labels = {"sub":sub_inds, "cond":cond_inds}
    dPTE = np.vstack(dPTEs)
    dPTE -= 0.5
    inmat = dPTE

hdbs = hdbscan.HDBSCAN(metric=metric, core_dist_n_jobs=8, min_cluster_size=50)
hdbs.fit(inmat)
