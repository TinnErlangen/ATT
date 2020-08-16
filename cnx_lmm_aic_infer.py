from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_undirected_cnx, plot_directed_cnx, load_sparse
import mne
import pickle
from collections import Counter
import matplotlib.pyplot as plt
plt.ion()

'''
Here we want to load up the results calculated in cnx_lmm_compare, infer
significance with the AIC, and visualise results
'''

proc_dir = "/home/jeff/ATT_dat/lmm/"
sps_dir = "/home/jeff/ATT_dat/proc/"
band = "alpha_1"
node_n = 2415
threshold = 0.001 # threshold for AIC comparison
cond_threshold = 0.05 # theshold for condition p values
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)
calc_aic = False
top_cnx = 70
bot_cnx = None

ROI = "L3969-lh"  # M1 central
ROI = "L3395-lh"  # M1 superior
# ROI = "L8143_L7523-lh" # M1 dorsal
# ROI = "L4557-lh"  # superior-parietal posterior
# ROI = "L7491_L4557-lh"  # left sup-parietal anterior
# ROI = None

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"] # these will form the main keys of aic_comps dictionary below
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
conds = ["rest","audio","visual","visselten","zaehlen"]
#conds = ["rest","audio","visual","visselten"]
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names

if calc_aic:
    aics = {mod:np.zeros(node_n) for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            try:
                this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}.pickle".format(proc_dir,band,mod,n_idx))
            except:
                continue
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod][n_idx] = this_mod.pvalues
            aics_params[mod][n_idx] = this_mod.params
            aics_confint[mod][n_idx] = this_mod.conf_int()

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.empty(node_n)
    aic_comps["winner_margin"] = np.empty(node_n)
    aic_comps["single_winner_ids"] = np.empty(node_n)
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["confint_params"] = np.zeros((node_n,len(stat_conds),2))
    for n_idx in range(node_n):
        for mod in models:
            if not aics[mod][n_idx]:
                aic_comps["single_winner_ids"][n_idx] = None
                continue
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array # store raw AIC values
        aic_prob = np.exp((aic_array.min()-aic_array)/2) # convert AIC to p-values; less than threshold indicates they are different than the mininum (best) fit
        aic_comps["probs"][n_idx,] = aic_prob
        aic_order = np.argsort(aic_prob) # model indices sorted by fit from best to worst
        aic_comps["order"][n_idx,] = aic_order
        aic_comps["winner_ids"][n_idx] = np.where(aic_order==len(models)-1)[0][0] # model index of best fit model
        aic_comps["winner_margin"][n_idx] = np.sort(aic_prob.copy())[len(models)-2] - aic_array.min() # distance between best and 2nd best fit model
        #aic_comps["winner_margin"][n_idx] = np.max(1-aic_prob[aic_prob!=1])
        aic_threshed = aic_prob.copy()
        aic_threshed[aic_threshed<threshold] = 0
        aic_threshed[aic_threshed>0] = 1
        aic_comps["threshed"][n_idx,] = aic_threshed # 0,1 indicator of statistical inference between models: best fit model or not significantly different from best fit are 1, otherwise 0
        if aic_comps["threshed"][n_idx].sum() == 1: # all other models significantly different than best model
            winner_idx = aic_comps["winner_ids"][n_idx]
            aic_comps["single_winner_ids"][n_idx] = winner_idx # mark 1 if all other models significantly different than best model
            if aic_comps["threshed"][n_idx][2] == 1: # if the best model was "cond," than find out which conditions were significantly different than rest
                for stat_cond_idx,stat_cond in enumerate(stat_conds):
                    if aics_pvals["cond"][n_idx][stat_cond] < cond_threshold:
                        aic_comps["sig_params"][n_idx][stat_cond_idx] = aics_params["cond"][n_idx][stat_cond]
                        aic_comps["confint_params"][n_idx][stat_cond_idx] = (aics_confint["cond"][n_idx].loc[stat_cond][0], aics_confint["cond"][n_idx].loc[stat_cond][1])
        else:
            aic_comps["single_winner_ids"][n_idx] = None

    with open("{}{}/aic.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

# plt.hist(aic_comps["single_winner_ids"])
# plt.title("Single winner IDs")

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

if ROI:
    ROI_idx = label_names.index(ROI)
    for stat_cond in stat_conds:
        mask_mat = np.zeros(cnx_params[stat_cond].shape)
        mask_mat[:,ROI_idx] = np.ones(cnx_params[stat_cond].shape[0])
        mask_mat[ROI_idx,:] = np.ones(cnx_params[stat_cond].shape[1])
        cnx_params[stat_cond] *= mask_mat

all_params = np.abs(np.array([cnx_params[stat_cond] for stat_cond in stat_conds]).flatten())
all_params.sort()
alpha_max, alpha_min = all_params[-1:], all_params[-top_cnx].min()
alpha_max, alpha_min = None, None
params_brains = []
for stat_cond,cond in zip(stat_conds,["audio","visual","visselten","zaehlen"]):
    params_brains.append(plot_directed_cnx(cnx_params[stat_cond],labels,parc,
                         alpha_min=alpha_min,alpha_max=alpha_max,
                         ldown_title=cond, top_cnx=top_cnx))

# sig_combs = []
# for sig_idx in range(len(aic_comps["sig_params"])):
#     row_sigs = tuple(np.where(aic_comps["sig_params"][sig_idx,])[0])
#     sig_combs.append(row_sigs)
# counts = dict(Counter(sig_combs))
# del counts[()]
# CnxOI = [(0,1,2),(0,1,2,3),(0,2),(0,),(1,2)]
# #CnxOI = [(0,2),(0,),(1,2)]
# cnx_oi = {coi:np.zeros((mat_n,mat_n)) for coi in CnxOI}
# coi_brains = []
# for coi in CnxOI:
#     for n_idx in range(node_n):
#         if sig_combs[n_idx] == coi:
#             cnx_oi[coi][triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1
#     coi_brains.append(plot_undirected_cnx(cnx_oi[coi],labels,parc,
#                       ldown_title=str(coi), top_cnx=counts[coi],
#                       uniform_weight=True))


# # now load up dPTEs
# subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
#          "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
#          "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
#          "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
#
# # average by subject, hold resting state separate because it was baseline
# dPTEs = [[] for cond in conds]
# for sub in subjs:
#     idx = 0
#     for cond in conds:
#         dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(sps_dir, sub,
#                                                            cond, band))
#         dPTEs[idx].append(dPTE.mean(axis=0))
#         idx += 1
# dPTEs = np.mean(dPTEs,axis=1)
#
# all_dPTEs = dPTEs.mean(axis=0)
# task_dPTEs = np.array([dPTEs[0,],dPTEs[1:,].mean(axis=0)])
# cond_dPTEs = dPTEs[1:,]
