from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import (plot_undirected_cnx, plot_directed_cnx, plot_rgba_cnx,
                       load_sparse, write_brain_image)
import mne
import pickle
import pandas as pd
from collections import Counter
from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

'''
Here we want to load up the results calculated in cnx_lmm_compare, infer
significance with the AIC, and visualise results
'''

proc_dir = "/home/jev/ATT_dat/lmm/"
sps_dir = "/home/jev/ATT_dat/proc/"
band = "alpha_1"
node_n = 2415
threshold = 0.001 # threshold for AIC comparison
cond_threshold = 0.05 # theshold for condition p values
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)
calc_aic = False
top_cnx = 100
bot_cnx = None
write_images = True
conds = ["rest","audio","visual","visselten","zaehlen"]
z_name = ""
no_Z = True
if no_Z:
    z_name = "no_Z"
    conds = ["rest","audio","visual","visselten"]

ROI = "L3969-lh"  # M1 central
ROI = "L3395-lh"  # M1 superior
# ROI = "L8143_L7523-lh" # M1 dorsal
# ROI = "L4557-lh"  # superior-parietal posterior
# ROI = "L7491_L4557-lh"  # left sup-parietal anterior
ROI = None

views = {"left":{"view":"lateral","distance":800,"hemi":"lh"},
         "right":{"view":"lateral","distance":800,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":900}
}

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"] # these will form the main keys of aic_comps dictionary below
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format

stat_conds = ["Intercept"] + [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names

if calc_aic:
    aics = {mod:np.zeros(node_n) for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    aics_predicted = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            try:
                this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}{}.pickle".format(proc_dir,band,mod,n_idx,z_name))
            except:
                continue
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod][n_idx] = this_mod.pvalues
            aics_params[mod][n_idx] = this_mod.params
            aics_confint[mod][n_idx] = this_mod.conf_int()
            predicts = pd.Series({en:0 for en in this_mod.model.exog_names}, dtype=float)
            for en_idx, en in enumerate(this_mod.model.exog_names):
                vector = np.zeros(len(this_mod.model.exog_names))
                vector[0] = 1
                vector[en_idx] = 1
                predicts[en] = this_mod.model.predict(this_mod.params, vector)
            aics_predicted[mod][n_idx] = predicts

    aic_comps = {var:np.zeros((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.zeros(node_n)
    aic_comps["winner_margin"] = np.zeros(node_n)
    aic_comps["single_winner_ids"] = np.zeros(node_n)
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["confint_params"] = np.zeros((node_n,len(stat_conds),2))
    aic_comps["dual_winner"] = np.zeros(node_n)
    aic_comps["simple_sig_params"] = np.zeros((node_n, 2))
    aic_comps["simple_confint_params"] = np.zeros((node_n,2,2))
    aic_comps["predicted"] = aics_predicted
    aic_comps["stat_conds"] = stat_conds
    aic_comps["conds"] = conds
    aic_comps["cond_dict"] = {k:v for k,v in zip(conds, stat_conds)}
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

        if np.array_equal(aic_comps["threshed"][n_idx], np.array([0,1,1])): # simple and cond model significantly better than null model
            aic_comps["dual_winner"][n_idx] = 1 # mark 1 if all other models significantly different than best model
            if aics_pvals["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"] < cond_threshold:
                aic_comps["simple_sig_params"][n_idx][1] = aics_params["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"]
                aic_comps["simple_sig_params"][n_idx][0] = aics_params["simple"][n_idx]["Intercept"]
                aic_comps["simple_confint_params"][n_idx] = (aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][0],
                                                             aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][1])
        else:
            aic_comps["dual_winner"][n_idx] = 0

    with open("{}{}/aic{}.pickle".format(proc_dir,band,z_name), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic{}.pickle".format(proc_dir,band,z_name), "rb") as f:
        aic_comps = pickle.load(f)

# plt.hist(aic_comps["winner_ids"])
# plt.title("Winner IDs")

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = {mod:np.zeros((mat_n,mat_n)) for mod in models}
cnx_params = {stat_cond:np.zeros((mat_n,mat_n)) for stat_cond in stat_conds}
cnx_params["simple_rest"] = np.zeros((mat_n,mat_n))
cnx_params["simple_task"] = np.zeros((mat_n,mat_n))

for n_idx in range(node_n):
    for stat_cond_idx,stat_cond in enumerate(stat_conds):
        if aic_comps["sig_params"][n_idx][stat_cond_idx]:
            cnx_params[stat_cond][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["sig_params"][n_idx][stat_cond_idx]
    if aic_comps["dual_winner"][n_idx]:
        cnx_params["simple_rest"][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["simple_sig_params"][n_idx][0]
        cnx_params["simple_task"][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["simple_sig_params"][n_idx][1]

# center intercepts around 0
for cp in ["Intercept", "simple_rest"]:
    inds = cnx_params[cp] != 0
    cnx_params[cp][inds] -= 0.5

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
alpha_max, alpha_min = 0.015, 0.001
alpha_max, alpha_min = None, None
params_brains = []
for stat_cond,cond in zip(stat_conds,conds):
    params_brains.append(plot_directed_cnx(cnx_params[stat_cond],labels,parc,
                         alpha_min=alpha_min,alpha_max=alpha_max,
                         ldown_title=cond, top_cnx=top_cnx))
    if write_images:
        write_brain_image(cond, views, params_brains[-1], "../images/")

params_brains.append(plot_directed_cnx(cnx_params["simple_task"],labels,parc,
                     alpha_min=None,alpha_max=None,
                     ldown_title="Simple (task)", top_cnx=top_cnx))
if write_images:
    write_brain_image("simple_task", views, params_brains[-1], "../images/")

params_brains.append(plot_directed_cnx(cnx_params["simple_rest"],labels,parc,
                     alpha_min=None,alpha_max=None,
                     ldown_title="Simple (rest)", top_cnx=top_cnx))
if write_images:
    if write_images:
        write_brain_image("simple_rest", views, params_brains[-1], "../images/")

# make 4D matrix with RGBA
mat_rgba = np.zeros((mat_n, mat_n, 4))
idx = 0
for stat_cond in stat_conds[1:]:
    if "zaehlen" in stat_cond:
        continue
    mat_rgba[...,idx] = abs(cnx_params[stat_cond])
    idx += 1
rgba_norm = np.linalg.norm(mat_rgba[...,:3],axis=2)
for idx in range(3):
    nonzero = np.where(rgba_norm)
    for x,y in zip(*nonzero):
        mat_rgba[x,y,idx] /= rgba_norm[x,y]
#rgba_norm = (rgba_norm-rgba_norm[rgba_norm>0].min())/(rgba_norm.max()-rgba_norm[rgba_norm>0].min())
mat_rgba[...,-1] = rgba_norm
params_brains.append(plot_rgba_cnx(mat_rgba.copy(), labels, parc,
                     ldown_title="Rainbow", top_cnx=top_cnx))
params_brains[-1]._renderer.plotter.add_legend([["Audio","r"],["Visual","g"],
                                               ["Visselten","b"]],
                                               bcolor=(0,0,0))
if write_images:
    write_brain_image("rainbow", views, params_brains[-1],
                "../images/")
