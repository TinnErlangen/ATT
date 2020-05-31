from mne.stats import f_mway_rm, summarize_clusters_stc, f_threshold_mway_rm
from cnx_utils import dPTE_to_undirected, load_sparse, phi, Graph, cnx_cluster
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mne.stats import fdr_correction
plt.ion()

def cnx_permute():
    data = dPTEs.copy()
    cond_slices = np.arange(dPTEs.shape[1])
    for sub_idx in range(data.shape[0]):
        np.random.shuffle(cond_slices)
        data[sub_idx,] = data[sub_idx,cond_slices,]
    f_vals, p_vals = f_mway_rm(data, factor_levels, effects=effects)
    return np.sum(f_vals[p_vals<comp_p_thresh])
    #return np.max(f_vals[effect_idx,])

proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
freq = "alpha_1"
conds = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
factor_levels = [len(conds), len(wavs)]
effects = ["A","B","A:B"]
effect_idx = 0
p_thresh = 0.05
comp_p_thresh = 0.05
perm_num = 10000
n_jobs = 8
cluster_idx = 0
f_thresh = f_threshold_mway_rm(len(subjs),factor_levels,effects=effects,
                               pvalue=p_thresh)

# get constraint info
net_names = ["rest-{}_{}_c{}".format(cond,freq,cluster_idx) for cond in conds]
net_nets = []
for net_n in net_names:
    net_nets += [tuple(x) for x in list(np.load("{}{}.npy".format(proc_dir,net_n)))]
net_nets = list(set(net_nets))
constrain_inds = tuple(zip(*net_nets))

dPTEs = [[] for c in conds for w in wavs]
for sub in subjs:
    idx = 0
    for cond in conds:
        for wav in wavs:
            dPTE = load_sparse("{}nc_{}_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                  cond, wav,
                                                                  freq))
            dPTEs[idx].append(dPTE)
            idx += 1

dPTEs = [[d.mean(axis=0) for d in dpte] for dpte in dPTEs]
dPTEs = np.array(dPTEs)
dPTEs = np.swapaxes(dPTEs,0,1)

# get only data under constraints and vectorise
dPTEs = dPTEs[...,np.array(constrain_inds[0]),np.array(constrain_inds[1])]
# move centre from 0.5 to 0
dPTEs -= 0.5

cnx_n = dPTE.shape[-1]
f_vals, p_vals = f_mway_rm(dPTEs, factor_levels, effects=effects)
fresult = np.max(f_vals[effect_idx,])
fresult = np.sum(f_vals[p_vals<comp_p_thresh])
print(fresult)
reject_fdr, p_vals_fdr = fdr_correction(p_vals[effect_idx,],alpha=0.05)

# permute shuffled data
results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cnx_permute)() for i in range(perm_num))
results = np.array(results)
plt.hist(results,bins=200)
result_thresh = np.quantile(results,1-comp_p_thresh)
print("{} threshold is {}".format(p_thresh, result_thresh))
if fresult>result_thresh:
    print("Signficant at at least {}".format(comp_p_thresh))
else:
    print("No significant components.")
