from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_undirected_cnx, plot_directed_cnx, load_sparse
import mne
import pickle
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jeff/ATT_dat/lmm_dics/"
band = "alpha_1"
node_n = 70
threshold = 0.2
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
mat_n = len(labels)
calc_aic = True
top_cnx = 250

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"]

if calc_aic:
    aics = {mod:np.empty(node_n) for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}.pickle".format(proc_dir,band,mod,n_idx))
            aics[mod][n_idx] = this_mod.aic

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.empty(node_n)
    aic_comps["winner_margin"] = np.empty(node_n)
    for n_idx in range(node_n):
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array
        aic_prob = np.exp((aic_array.min()-aic_array)/2)
        aic_comps["probs"][n_idx,] = aic_prob
        aic_order = np.argsort(aic_prob)
        aic_comps["order"][n_idx,] = aic_order
        aic_comps["winner_ids"][n_idx] = np.where(aic_order==len(models)-1)[0][0]
        aic_comps["winner_margin"][n_idx] = np.sort(aic_array)[1] - aic_array.min()
        aic_threshed = aic_prob.copy()
        aic_threshed[aic_threshed<threshold] = 0
        aic_threshed[aic_threshed>0] = 1
        aic_comps["threshed"][n_idx,] = aic_threshed

    with open("{}{}/aic.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

plt.hist(aic_comps["winner_ids"])
