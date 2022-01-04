from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_rgba, plot_parc_compare, make_brain_image
import mne
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize, ListedColormap
#plt.ion()

def param_norm(params):
    param_abs = abs(params)
    pa_min, pa_max, = param_abs[param_abs!=0].min(), param_abs.max()
    param_norms = ((param_abs - pa_min) / (pa_max - pa_min)) * np.sign(params)
    return param_norms

def params_to_rgba(params):
    alphas = np.abs(params)
    rgba = np.zeros((len(alphas), 4))
    rgba[params>0, 0] = 1
    rgba[params<0, 2] = 1
    rgba[params!=0, 3] = alphas[alphas!=0]
    return rgba

proc_dir = "/home/jev/ATT_dat/lmm_dics/"
band = "alpha_1"
node_n = 70
threshold = 0.05
cond_threshold = 0.05
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage", parc)
mat_n = len(labels)
calc_aic = True
cmap = "seismic"

# import os
# os.environ["QT_API"] = "pyqt5"

views = {"left":{"view":"lateral","distance":500,"hemi":"lh"},
         "right":{"view":"lateral","distance":500,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":500}}
parc_ov_views = {"left":{"view":"lateral","distance":500,"hemi":"lh"},
                 "right":{"view":"lateral","distance":500,"hemi":"rh"},
                 "upper":{"view":"dorsal","distance":500}
}

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"]
conds = ["rest","audio","visual","visselten","zaehlen"]
conds = ["rest","audio","visual","visselten"]
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]]

if calc_aic:
    # get permutations
    perms = np.load("{}{}/dics_byresp_perm_aics.npy".format(proc_dir, band))
    aics = {mod:np.empty(node_n) for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_byresp_{}.pickle".format(proc_dir,band,mod,n_idx))
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod][n_idx] = this_mod.pvalues
            aics_params[mod][n_idx] = this_mod.params
            aics_confint[mod][n_idx] = this_mod.conf_int()

    # calculate the AIC delta thresholds from the permutations
    null_tile = np.tile(np.expand_dims(aics["null"], 1), (1,1024))
    perm_simp_diff = perms[...,0] - null_tile
    perm_simp_maxima, perm_simp_minima = (perm_simp_diff.max(axis=1),
                                          perm_simp_diff.min(axis=1))
    simp_thresh = np.quantile(perm_simp_minima, threshold/2)

    perm_avg_tile = np.tile(np.mean(perms[...,0], axis=1, keepdims=True),
                            (1, 1024))
    perm_cond_diff = perms[...,1] - perm_avg_tile
    perm_cond_maxima, perm_cond_minima = (perm_cond_diff.max(axis=1),
                                          perm_cond_diff.min(axis=1))
    cond_thresh = np.quantile(perm_cond_minima, threshold/2)

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["confint_params"] = np.zeros((node_n,len(stat_conds),2))
    aic_comps["simp_params"] = np.zeros((node_n, 2))
    aic_comps["simp_confint_params"] = np.zeros((node_n, 2, 2))
    for n_idx in range(node_n):
        for mod in models:
            if not aics[mod][n_idx]:
                continue
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array # store raw AIC values
        simp_diff = aics["simple"][n_idx] - aics["null"][n_idx]
        cond_diff = aics["cond"][n_idx] - aics["simple"][n_idx]
        aic_threshed = np.array([1,0,0])
        if simp_diff < simp_thresh:
            aic_threshed = np.array([0,1,0])
        if cond_diff < cond_thresh:
            aic_threshed = np.array([0,0,1])

        if np.where(aic_threshed)[0] == 2: # if the best model was "cond," than find out which conditions were significantly different than rest
            for stat_cond_idx,stat_cond in enumerate(stat_conds):
                if aics_pvals["cond"][n_idx][stat_cond] < cond_threshold:
                    aic_comps["sig_params"][n_idx][stat_cond_idx] = aics_params["cond"][n_idx][stat_cond]
                    aic_comps["confint_params"][n_idx][stat_cond_idx] = (aics_confint["cond"][n_idx].loc[stat_cond][0], aics_confint["cond"][n_idx].loc[stat_cond][1])
        if np.where(aic_threshed)[0] == 1: # simple model wins
            if aics_pvals["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"] < cond_threshold:
                aic_comps["simp_params"][n_idx][1] = aics_params["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"]
                aic_comps["simp_params"][n_idx][0] = aics_params["simple"][n_idx]["Intercept"]
                aic_comps["simp_confint_params"][n_idx] = (aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][0],
                                                             aics_confint["simple"][n_idx].loc["C(Block, Treatment('rest'))[T.task]"][1])

    with open("{}{}/aic.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

brains = []

## Region70 with aparc
# brains.append(plot_parc_compare("aparc", "RegionGrowing_70"))
# make_brain_figure(parc_ov_views, brains[-1])
figsize = 2160
fontsize=150
img_rat = 8
pans = ["A", "B", "C", "D"]
pads = ["Z", "Y", "X", "W"]
descs = ["General", "Auditory", "Visual", "Aud. distraction"]
mos_str = ""
pad = True
for pan, pad in zip(pans, pads):
    for idx in range(img_rat):
        mos_str += pan+"\n"
    if pad:
        mos_str += pad+"\n"
mos_str += "E"

fig, axes = plt.subplot_mosaic(mos_str, figsize=(64.80, 90.00))
for desc, pan in zip(descs, pans):
    axes[pan].set_title("{} task".format(desc),
                        fontsize=fontsize)
    axes[pan].axis("off")
for pad in pads:
    axes[pad].axis("off")

params = aic_comps["sig_params"].copy()
simp_params = np.expand_dims(aic_comps["simp_params"].copy()[:,1], 1)
params = np.concatenate((params, simp_params), axis=1)
params_n = param_norm(params)
vmin = -abs(params).max()
vmax = abs(params).max()

param_n = params_n[:,-1]
rgba = params_to_rgba(param_n)
brains.append(plot_rgba(rgba, labels, parc, figsize=figsize))
img = make_brain_image(views, brains[-1], text="A", text_loc="lup", text_pan=0)
ax = axes["A"]
ax.imshow(img)

# changes from rest for each individual task
# audio
param_n = params_n[:,0]
rgba = params_to_rgba(param_n)
brains.append(plot_rgba(rgba, labels, parc, figsize=figsize))
img = make_brain_image(views, brains[-1], text="B", text_loc="lup", text_pan=0)
ax = axes["B"]
ax.imshow(img)

# visual
param_n = params_n[:,1]
rgba = params_to_rgba(param_n)
brains.append(plot_rgba(rgba, labels, parc, figsize=figsize))
img = make_brain_image(views, brains[-1], text="C", text_loc="lup", text_pan=0)
ax = axes["C"]
ax.imshow(img)

# visselten
param_n = params_n[:,2]
rgba = params_to_rgba(param_n)
brains.append(plot_rgba(rgba, labels, parc, figsize=figsize))
img = make_brain_image(views, brains[-1], text="D", text_loc="lup", text_pan=0)
ax = axes["D"]
ax.imshow(img)

ax = axes["E"]
norm = Normalize(vmin, vmax)
scalmap = cm.ScalarMappable(norm, cmap)
plt.colorbar(scalmap, cax=ax, orientation="horizontal")
ax.tick_params(labelsize=fontsize)
ax.set_xlabel("DICS power (NAI normalised)",
               fontsize=fontsize*0.8)
plt.suptitle("Estimated power change from resting state", fontsize=fontsize)
plt.savefig("../images/dics_figure.png")
# # rainbow
# this_rgba = np.zeros((len(labels), 4))
# params = np.abs(aic_comps["sig_params"])
# param_norms = np.linalg.norm(params, axis=1)
# for idx in range(len(params)):
#     if param_norms[idx]:
#         params[idx,] /= param_norms[idx]
# this_rgba[:,:3] = params
# param_norms = param_norm(param_norms)
# this_rgba[:,3] = param_norms
# brains.append(plot_rgba(abs(this_rgba), labels, parc, lup_title="Rainbow"))
# brains[-1]._renderer.plotter.add_legend([["Audio","r"],["Visual","g"],
#                                          ["Visselten","b"]], bcolor=(0,0,0))
# make_brain_figure(views, brains[-1])
