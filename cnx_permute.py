from mne.stats import f_mway_rm, summarize_clusters_stc, f_threshold_mway_rm
from cnx_utils import dPTE_to_undirected, load_sparse, phi, Graph, cnx_cluster
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.ion()

def cnx_permute():
    data = dPTEs.copy()
    cond_slices = np.arange(dPTEs.shape[1])
    for sub_idx in range(data.shape[0]):
        np.random.shuffle(cond_slices)
        data[sub_idx,] = data[sub_idx,cond_slices,]
    f_vals, p_vals = f_mway_rm(data, factor_levels, effects=effects)
    fsums, edges = cnx_cluster(f_vals, p_vals, cnx_n, p_thresh=p_thresh)
    return np.array(fsums).max()

proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
freqs = ["gamma_2"]
conds = ["rest", "visselten"]
if not (len(conds) == 1 or len(freqs) == 1):
    raise ValueError("Conds or freqs must have length 1.")
factor_levels = [max([len(conds),len(freqs)])]
effects = "A"
p_thresh = 0.05
comp_p_thresh = 0.05
perm_num = 10000
n_jobs = 8
f_thresh = f_threshold_mway_rm(len(subjs),factor_levels,effects=effects,
                               pvalue=p_thresh)

dPTEs = [[] for c in conds for f in freqs]
for sub in subjs:
    idx = 0
    for cond in conds:
        for freq in freqs:
            dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                               cond, freq))
            dPTEs[idx].append(dPTE)
            idx += 1

#trial_min = np.array([len(d) for dpte in dPTEs for d in dpte]).min()
#pruned_dPTE = [[d[np.random.randint(0,len(d),trial_min)] for d in dpte] for dpte in dPTEs]
vec_dPTE = [[phi(d.mean(axis=0),k=1) for d in dpte] for dpte in dPTEs]

dPTEs = np.swapaxes(np.array(vec_dPTE),0,1)
dPTEs -= 0.5

cnx_n = dPTE.shape[-1]
f_vals, p_vals = f_mway_rm(dPTEs, factor_levels, effects=effects)
fsums, edges = cnx_cluster(f_vals, p_vals, cnx_n, p_thresh=p_thresh)
print(fsums)

# # permute shuffled data
# results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(cnx_permute)() for i in range(perm_num))
# results = np.array(results)
# plt.hist(results,bins=200)
# fsum_thresh = np.quantile(results,1-comp_p_thresh)
# print("{} threshold is {}".format(p_thresh, fsum_thresh))
# sig_comps = np.where([f > fsum_thresh for f in fsums])[0]
# if len(sig_comps):
#     for comp_idx in np.nditer(sig_comps):
#         edge_out = np.array(edges[comp_idx])
#         if len(conds) > 1:
#             np.save("{}{}-{}_{}_c{}.npy".format(proc_dir,conds[0],conds[1],freqs[0],comp_idx),edge_out)
#         if len(freqs) > 1:
#             np.save("{}{}-{}_{}_c{}.npy".format(proc_dir,freqs[0],freqs[1],conds[0],comp_idx),edge_out)
# else:
#     print("No significant components.")
