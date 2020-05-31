from sklearn.manifold import SpectralEmbedding as SE
import numpy as np
import pickle
from cnx_utils import load_sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

proc_dir = "/home/jeff/ATT_dat/proc/"
comp_num = 6
alpha = 1
precomputed = False
epo_avg = False

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
conds = ["rest", "audio", "visual", "visselten"]
freq = "alpha_1"

if precomputed:
    affinity = "precomputed"
    distmat = load_sparse("{}{}_dist.sps".format(proc_dir,filename))
    distmat += distmat.T
    distmat = 1 - distmat
    inmat = distmat
    with open("{}{}_dist.labels".format(proc_dir,filename),"rb") as f:
        labels = pickle.load(f)
else:
    affinity = "nearest_neighbors"
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

cond_array = np.array(labels["cond"])
sub_array = np.array(labels["sub"])
conds = list(np.unique(cond_array))
subs = list(np.unique(sub_array))

colors = ["blue","red","green","cyan"]

se = SE(n_components=comp_num, affinity=affinity, n_jobs=8)
y = se.fit_transform(inmat)

fig = plt.figure()
if comp_num == 2:
    ax = fig.add_subplot(111)
    for cond,col in zip(conds,colors):
        ax.scatter(y[cond_array==cond,0],y[cond_array==cond,1],color=col,
                   alpha=alpha, marker=".")
        ax.set_xlim((np.quantile(y[:,0],0.01),
                     np.quantile(y[:,0],0.99)))
        ax.set_ylim((np.quantile(y[:,1],0.01),
                     np.quantile(y[:,1],0.99)))
elif comp_num > 2:
    ax = fig.add_subplot(111, projection='3d')
    for cond,col in zip(conds,colors):
        ax.scatter(y[cond_array==cond,0],y[cond_array==cond,1],
                   zs=y[cond_array==cond,2],color=col, alpha=alpha, marker=".")

if comp_num > 5:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cond,col in zip(conds,colors):
        ax.scatter(y[cond_array==cond,3],y[cond_array==cond,4],
                   zs=y[cond_array==cond,5],color=col, alpha=alpha, marker=".")

fig, axes = plt.subplots(comp_num,1)
bins = 20
inds = list(range(comp_num))
for idx_idx,idx in enumerate(inds):
    axes[idx_idx].hist([y[cond_array==cond,idx] for cond in conds], alpha=alpha,
                        bins=bins, color=colors[:len(conds)])

if comp_num > 5:
    fig, axes = plt.subplots(3,1)
    bins = 20
    inds = [3,4,5]
    for idx_idx,idx in enumerate(inds):
        axes[idx_idx].hist([y[cond_array==cond,idx] for cond in conds],
                            alpha=0.5,bins=bins,
                            color=colors[:len(conds)])

np.save("{}se{}_{}.npy".format(proc_dir,comp_num,filename),y)
