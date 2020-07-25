from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_undirected_cnx, plot_directed_cnx, load_sparse
import mne
import pickle
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jeff/ATT_dat/lmm/"
sps_dir = "/home/jeff/ATT_dat/proc/"
band = "alpha_1"
node_n = 2415
threshold = 0.001
cond_threshold = 0.05
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
mat_n = len(labels)
calc_aic = False
top_cnx = 100
bot_cnx = None

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"]
var_base = "C(Block, Treatment('rest'))"
conds = ["rest","audio","visual","visselten","zaehlen"]
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]]

if calc_aic:
    aics = {mod:np.empty(node_n) for mod in models}
    aics_pvals = {mod:[] for mod in models}
    aics_params = {mod:[] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}.pickle".format(proc_dir,band,mod,n_idx))
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod].append(this_mod.pvalues)
            aics_params[mod].append(this_mod.params)

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.empty(node_n)
    aic_comps["winner_margin"] = np.empty(node_n)
    aic_comps["single_winner_ids"] = np.empty(node_n)
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    for n_idx in range(node_n):
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array
        aic_prob = np.exp((aic_array.min()-aic_array)/2)
        aic_comps["probs"][n_idx,] = aic_prob
        aic_order = np.argsort(aic_prob)
        aic_comps["order"][n_idx,] = aic_order
        aic_comps["winner_ids"][n_idx] = np.where(aic_order==len(models)-1)[0][0]
        aic_comps["winner_margin"][n_idx] = np.sort(aic_prob.copy())[len(models)-2] - aic_array.min()
        #aic_comps["winner_margin"][n_idx] = np.max(1-aic_prob[aic_prob!=1])
        aic_threshed = aic_prob.copy()
        aic_threshed[aic_threshed<threshold] = 0
        aic_threshed[aic_threshed>0] = 1
        aic_comps["threshed"][n_idx,] = aic_threshed
        if aic_comps["threshed"][n_idx].sum() == 1:
            winner_idx = aic_comps["winner_ids"][n_idx]
            aic_comps["single_winner_ids"][n_idx] = winner_idx
            if aic_comps["threshed"][n_idx][2] == 1:
                for stat_cond_idx,stat_cond in enumerate(stat_conds):
                    if aics_pvals["cond"][n_idx][stat_cond] < cond_threshold:
                        aic_comps["sig_params"][n_idx][stat_cond_idx] = aics_params["cond"][n_idx][stat_cond]
        else:
            aic_comps["single_winner_ids"][n_idx] = None

    with open("{}{}/aic.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

plt.hist(aic_comps["single_winner_ids"])

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = {mod:np.zeros((mat_n,mat_n)) for mod in models}
cnx_params = {stat_cond:np.zeros((mat_n,mat_n)) for stat_cond in stat_conds}
brains = []
colors = [(1,0,0),(0,1,0),(0,0,1)]
models, colors = ["cond"], [(0,0,1)]
for color, mod in zip(colors, models):
    mod_idx = aic_comps["models"].index(mod)
    for n_idx in range(node_n):
        if aic_comps["single_winner_ids"][n_idx] == mod_idx:
            cnx_masks[mod][triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1
        if mod == "cond":
            for stat_cond_idx,stat_cond in enumerate(stat_conds):
                if aic_comps["sig_params"][n_idx][stat_cond_idx]:
                    cnx_params[stat_cond][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["sig_params"][n_idx][stat_cond_idx]

all_params = np.abs(np.array([cnx_params[stat_cond] for stat_cond in stat_conds]).flatten())
all_params.sort()
alpha_max, alpha_min = all_params[-1:], all_params[-top_cnx].min()
params_brains = []
for stat_cond,cond in zip(stat_conds,["audio","visual","visselten","zaehlen"]):
    params_brains.append(plot_directed_cnx(cnx_params[stat_cond],labels,parc,
                         alpha_min=alpha_min,alpha_max=alpha_max,
                         ldown_title=cond, top_cnx=top_cnx))

# now load up dPTEs
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

# average by subject, hold resting state separate because it was baseline
dPTEs = [[] for cond in conds]
for sub in subjs:
    idx = 0
    for cond in conds:
        dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(sps_dir, sub,
                                                           cond, band))
        dPTEs[idx].append(dPTE.mean(axis=0))
        idx += 1
dPTEs = np.mean(dPTEs,axis=1)

all_dPTEs = dPTEs.mean(axis=0)
task_dPTEs = np.array([dPTEs[0,],dPTEs[1:,].mean(axis=0)])
cond_dPTEs = dPTEs[1:,]
