from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import (plot_undirected_cnx, plot_directed_cnx, plot_rgba_cnx,
                       load_sparse, make_brain_image, annotated_matrix)
import mne
import pickle
import pandas as pd
from collections import Counter
from os import listdir
from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
#matplotlib.rcParams['figure.dpi'] = 1200

def pval_from_perms(perms, val):
    perms.sort()
    loc = abs(perms - val).argmin()
    pval = loc / len(perms)
    return pval

'''
Here we want to load up the results calculated in cnx_lmm_compare, infer
significance with the AIC and permutations, and visualise results
'''

proc_dir = "/home/jev/ATT_dat/proc/"
lmm_dir = "/home/jev/ATT_dat/lmm/"
band = "alpha_1"
node_n = 2415
perm_n = 1024
threshold = 0.05 # threshold for AIC comparison
cond_threshold = 0.05 # theshold for condition p values
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)
calc_aic = False
background = (1,1,1)
text_color = (0,0,0)
top_cnx = 150
figsize = 1920
bot_cnx = None
write_images = False
conds = ["rest", "audio", "visual", "visselten", "zaehlen"]
z_name = ""
no_Z = False
if no_Z:
    z_name = "no_Z"
    conds = ["rest", "audio", "visual", "visselten"]

# ROI = "L3969-lh"  # M1 central
# ROI = "L3395-lh"  # M1 superior
# ROI = "L8143_L7523-lh" # M1 dorsal
# ROI = "L4557-lh"  # superior-parietal posterior
# ROI = "L7491_L4557-lh"  # left sup-parietal anterior
ROI = None

views = {"left":{"view":"lateral", "distance":900, "hemi":"lh"},
         "right":{"view":"lateral", "distance":900, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":900},
         "caudal":{"view":"caudal", "distance":800}
}

region_dict = {"occipital":["L2340", "L2685", "L4236_L1933", "L10017",
                            "L2340_L1933"],
               "parietal":["L4557", "L4557_L2996", "L5037", "L7491_L4557",
                           "L7491_L5037", "L8143", "L8729_L7491", "L928"],
               "temporal":["L5106_L2688", "L5511_L4359", "L7049", "L7097_L4359",
                           "L7097_L5106", "L7755", "L2235"],
               "central":["L1154_L1032", "L3395", "L3969", "L7550_L3015",
                          "L8143_L7523", "L1032"],
               "frontal":["L1869", "L4118", "L4118_L2817", "L6412",
                          "L6412_L4118", "L6698_L1154", "L8983_L3015",
                          "L9249_L6698", "L2817"]
               }

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "winner"] # these will form the main keys of aic_comps dictionary below
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
stat_conds = ["Intercept"] + [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names

if calc_aic:
    # get permutations
    perm_dir = "{}{}/cnx_perm/".format(lmm_dir, band)
    perm_file_list = listdir(perm_dir)
    file_n = len(perm_file_list)
    perms = {"null":np.zeros((node_n, perm_n)), "simple":np.zeros((node_n, 0)),
             "cond":np.zeros((node_n, 0))}
    for pf in perm_file_list:
        if ".pickle" not in pf:
            continue
        with open("{}{}".format(perm_dir, pf), "rb") as f:
            this_perm = pickle.load(f)
        perms["simple"] = np.hstack((perms["simple"], this_perm["simple"]))
        perms["cond"] = np.hstack((perms["cond"], this_perm["cond"]))

    aics = {mod:np.zeros(node_n) for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    aics_predicted = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            #try:
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}{}.pickle".format(lmm_dir,band,mod,n_idx,z_name))
            # except:
            #     continue
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

    # calculate the AIC delta thresholds from the permutations
    null_tile = np.tile(np.expand_dims(aics["null"],1), (1,1024))
    perm_simp_diff = perms["simple"] - null_tile
    perm_simp_maxima, perm_simp_minima = (np.nanmax(perm_simp_diff, axis=1),
                                          np.nanmin(perm_simp_diff, axis=1))
    simp_thresh = np.quantile(perm_simp_minima, threshold/2)

    perm_avg_tile = np.tile(np.nanmean(perms["simple"], axis=1, keepdims=True),
                            (1, 1024))
    perm_cond_diff = perms["cond"] - perm_avg_tile
    perm_cond_maxima, perm_cond_minima = (np.nanmax(perm_cond_diff, axis=1),
                                          np.nanmin(perm_cond_diff, axis=1))
    cond_thresh = np.quantile(perm_cond_minima, threshold/2)

    aic_comps = {var:np.zeros((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["confint_params"] = np.zeros((node_n,len(stat_conds),2))
    aic_comps["simple_sig_params"] = np.zeros((node_n, 2))
    aic_comps["simple_confint_params"] = np.zeros((node_n,2,2))
    aic_comps["null_intercept"] = np.zeros((node_n, 1))
    aic_comps["predicted"] = aics_predicted
    aic_comps["stat_conds"] = stat_conds
    aic_comps["conds"] = conds
    aic_comps["cond_dict"] = {k:v for k,v in zip(conds, stat_conds)}
    for n_idx in range(node_n):
        for mod in models:
            if not aics[mod][n_idx]:
                continue
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array # store raw AIC values
        simp_delta = aic_array[1] - aic_array[0]
        cond_delta = aic_array[2] - aic_array[1]
        winners = np.array([1,0,0])
        if simp_delta < simp_thresh:
            winners = np.array([0,1,0])
        if cond_delta < cond_thresh:
            winners = np.array([0,0,1])

        aic_comps["winner"][n_idx,] = winners # 0,1 indicator of statistical inference between models: best fit model or not significantly different from best fit are 1, otherwise 0
        if aic_comps["winner"][n_idx][2] == 1: # if the best model was "cond," than find out which conditions were significantly different than rest
            for stat_cond_idx, stat_cond in enumerate(stat_conds):
                if aics_pvals["cond"][n_idx][stat_cond] < cond_threshold:
                    aic_comps["sig_params"][n_idx][stat_cond_idx] = aics_params["cond"][n_idx][stat_cond]
                    aic_comps["confint_params"][n_idx][stat_cond_idx] = (aics_confint["cond"][n_idx].loc[stat_cond][0], aics_confint["cond"][n_idx].loc[stat_cond][1])
        elif aic_comps["winner"][n_idx][1] == 1: # simple model wins
            if aics_pvals["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"] < cond_threshold:
                aic_comps["simple_sig_params"][n_idx][1] = aics_params["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"]
                aic_comps["simple_sig_params"][n_idx][0] = aics_params["simple"][n_idx]["Intercept"]
                aic_comps["simple_confint_params"][n_idx] = (aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][0],
                                                             aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][1])
        elif aic_comps["winner"][n_idx][0] == 1: # null model wins
            aic_comps["null_intercept"][n_idx][0] = aics_params["null"][n_idx]["Intercept"]

    with open("{}/{}/aic_perm{}.pickle".format(lmm_dir,band,z_name), "wb") as f:
        pickle.dump(aic_comps, f)
else:
    with open("{}/{}/aic_perm{}.pickle".format(lmm_dir, band, z_name), "rb") as f:
        aic_comps = pickle.load(f)

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = {mod:np.zeros((mat_n,mat_n)) for mod in models}
cnx_params = {stat_cond:np.zeros((mat_n,mat_n)) for stat_cond in stat_conds}
cnx_params["task"] = np.zeros((mat_n,mat_n))

for n_idx in range(node_n):
    if np.array_equal(aic_comps["winner"][n_idx], [0,0,1]):
        for stat_cond_idx, stat_cond in enumerate(stat_conds):
            if aic_comps["sig_params"][n_idx][stat_cond_idx]:
                cnx_params[stat_cond][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["sig_params"][n_idx][stat_cond_idx]
    elif np.array_equal(aic_comps["winner"][n_idx], [0,1,0]):
        cnx_params["Intercept"][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["simple_sig_params"][n_idx][0]
        cnx_params["task"][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["simple_sig_params"][n_idx][1]
    elif np.array_equal(aic_comps["winner"][n_idx], [1,0,0]):
        cnx_params["Intercept"][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["null_intercept"][n_idx][0]

# center intercepts around 0
inds = cnx_params["Intercept"] != 0
cnx_params["Intercept"][inds] -= 0.5

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
params_brains = {}
for stat_cond, cond in zip(stat_conds, conds):

    ### temp: take this out later
    if stat_cond != "Intercept":
        continue
    ####

    params_brains[cond] = plot_directed_cnx(cnx_params[stat_cond], labels,parc,
                                            alpha_min=alpha_min,
                                            alpha_max=alpha_max,
                                            ldown_title="", top_cnx=top_cnx,
                                            figsize=figsize,
                                            background=background,
                                            text_color=text_color)
    if write_images:
        make_brain_figure(views, params_brains[-1])

# params_brains["task"] = plot_directed_cnx(cnx_params["task"],
#                                           labels, parc, alpha_min=None,
#                                           alpha_max=None,
#                                           ldown_title="",
#                                           top_cnx=top_cnx,
#                                           figsize=figsize,
#                                           background=background,
#                                           text_color=text_color)
# if write_images:
#     make_brain_figure(views, params_brains[-1])

# make figure for manuscripts

# rearrange matrices by region, ant-pos, left/right


fontsize=165
img_rat = 12

pans = ["A", "B"]
pads = ["Z", "Y"]

mos_str = ""
do_pad = True
for pan, pad in zip(pans, pads):
    for idx in range(img_rat):
        mos_str += pan
    if do_pad:
        mos_str += pad

# mos_figsize = np.array([len(pans)*figsize, len(views)*figsize])/100
# fig, axes = plt.subplot_mosaic(mos_str, figsize=mos_figsize)
# for pan in pans:
#     # axes[pan].set_title("{}".format(desc),
#     #                     fontsize=fontsize)
#     axes[pan].axis("off")
# for pad in pads:
#     axes[pad].axis("off")
#
# # rest cnx by brainview
# img = make_brain_image(views, params_brains["rest"], text="",
#                        text_loc="lup", text_pan=0, orient="square")
# axes["A"].imshow(img)
#

# cnx conditions by matrix
m_fig, m_axes = plt.subplots(4, 3, figsize=(17.2, 20))
annot_labels = [{"col_key":{1:"tab:orange", 2:"tab:pink"},
                 "labels":np.concatenate((np.ones(35), np.ones(35)*2))},
                 {"col_key":{1:(.5,.5,.5, 0.2), 2:(.5,.5,.5,0)},
                  "labels":np.concatenate((np.ones(10), np.ones(60)*2))}]
img = annotated_matrix(cnx_params["Intercept"], label_names, annot_labels,
                       annot_vert_pos="left", annot_hor_pos="bottom",
                       overlay=True, annot_height=6)

#m_axes[0][0].imshow(img)
m_axes = [ax for axe in m_axes for ax in axe]
for ma in m_axes:
    ma.imshow(img)
    ma.axis("off")
plt.tight_layout()
#plt.savefig("../images/cnx_figure{}.png".format(sup_str))
