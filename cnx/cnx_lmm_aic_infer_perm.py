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
import io
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
plt.ion()

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

ROI = None

band_name = {"theta_0":"Theta (4-8Hz)", "alpha_0":"Low alpha (8-10Hz)",
             "alpha_1":"High alpha (10-13Hz)", "beta_0":"Beta (13-30Hz)",
             "gamma_0":"Gamma (31-48Hz)"}

views = {"left":{"view":"lateral", "distance":625, "hemi":"lh"},
         "right":{"view":"lateral", "distance":625, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":650,
                  "focalpoint":(-.77, 3.88, -21.53)},
         "caudal":{"view":"caudal", "distance":600}
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

    # ## temp: take this out later
    # if stat_cond != "Intercept":
    #     continue
    # ###

    params_brains[cond] = plot_directed_cnx(cnx_params[stat_cond], labels,parc,
                                            alpha_min=alpha_min,
                                            alpha_max=alpha_max,
                                            ldown_title="", top_cnx=top_cnx,
                                            figsize=figsize,
                                            background=background,
                                            text_color=text_color)

params_brains["task"] = plot_directed_cnx(cnx_params["task"],
                                          labels, parc, alpha_min=None,
                                          alpha_max=None,
                                          ldown_title="",
                                          top_cnx=top_cnx,
                                          figsize=figsize,
                                          background=background,
                                          text_color=text_color)

###### make figure for manuscripts


# rest cnx by brainview
brain_img = make_brain_image(views, params_brains["rest"], text="",
                             text_loc="lup", text_pan=0, orient="horizontal")


# cnx conditions by matrix

# rearrange by region, hemisphere, ant/pos
inds = []
reg_arranged = []
hemi_arranged = []
for lobe, lobe_regs in region_dict.items():
    for hemi in ["lh", "rh"]:
        these_label_names = ["{}-{}".format(lr, hemi) for lr in lobe_regs]
        these_labels = [x for x in labels if x.name in these_label_names]
        # order based on ant-pos location
        these_ypos = np.array([x.pos[:,1].mean() for x in these_labels])
        these_label_names = [these_label_names[x] for x in these_ypos.argsort()]
        inds.extend([label_names.index(x) for x in these_label_names])
        reg_arranged.extend([lobe for x in these_label_names])
        hemi_arranged.extend([hemi for x in these_label_names])
## rearrange matrix indices
# we still want them in old format for later
old_cnx_params = {k:v.copy() for k,v in cnx_params.items()}
inds = np.array(inds)
for k in cnx_params.keys():
    cnx_params[k] += cnx_params[k].T * -1
    cnx_params[k] = cnx_params[k][inds, :]
    cnx_params[k] = cnx_params[k][:, inds]

# annotation specifications
annot_labels = [{"col_key":{"occipital":"tab:orange", "parietal":"gold",
                            "temporal":"tab:pink", "central":"lime",
                            "frontal":"tab:cyan"}, "labels":reg_arranged},
                 {"col_key":{"lh":(.5, .5, .5, 0.35), "rh":(.5, .5, .5, 0.5)},
                  "labels":hemi_arranged}]

# estimated parameters
mos_str = """
          AAAABBBBCCCCX
          DDDDEEEEFFFFY
          """
ax_names = ["A", "B", "C", "D", "E", "F"]
fx_fig, fx_axes = plt.subplot_mosaic(mos_str, figsize=(76.8, 38.4))
# get universal vmin, vmax
all_vals = np.concatenate(list(cnx_params.values()))
vmin, vmax = np.min(all_vals), np.max(all_vals)
conds = ["Resting state", "Audio task", "Visual task",
         "Visual w/distraction", "Counting backwards", "General task"]
stat_conds += ["task"]
for ax_n, cond, stat_cond in zip(ax_names, conds, stat_conds):
    img = annotated_matrix(cnx_params[stat_cond], label_names, annot_labels,
                           annot_vert_pos="left", annot_hor_pos="bottom",
                           overlay=True, annot_height=6, vmin=vmin, vmax=vmax)
    fx_axes[ax_n].imshow(img)
    fx_axes[ax_n].set_title(cond, fontsize=84)
    fx_axes[ax_n].axis("off")

# colorbar in Y
disp_vmin, disp_vmax = np.around(vmin, decimals=3), np.around(vmax, decimals=3)
cbar = plt.colorbar(ScalarMappable(norm=Normalize(vmin=disp_vmin, vmax=disp_vmax),
                    cmap="seismic"), cax=fx_axes["X"])
cbar.ax.set_yticks([disp_vmin, 0, disp_vmax])
cbar.ax.set_yticklabels([disp_vmin, 0, disp_vmax], fontsize=54)
cbar.ax.text(0.25, 1.01, "X \u2192 Y", fontsize=60, transform=cbar.ax.transAxes,
             weight="bold")
cbar.ax.text(0.25, -.05, "Y \u2192 X", fontsize=60, transform=cbar.ax.transAxes,
             weight="bold")
cbar.ax.text(-.2, 0.4, "dPTE - 0.5", rotation="vertical", fontsize=54,
             transform=cbar.ax.transAxes)

# custom legend in Y
fx_axes["Y"].set_xlim(0, 2)
fx_axes["Y"].set_ylim(0, 12)
fx_axes["Y"].axis("off")
fs = 48
bar_w, bar_h = 0.6, 0.5
reg_key = annot_labels[0]["col_key"]
hemi_key = annot_labels[1]["col_key"]
y = 11
for rk, rv in reg_key.items():
    for hk, hv in hemi_key.items():
        rect = Rectangle((0.1, y), bar_w, bar_h, color=rv)
        fx_axes["Y"].add_patch(rect)
        rect = Rectangle((0.1, y), bar_w, bar_h, color=hv)
        fx_axes["Y"].add_patch(rect)
        txt = "{}-{}".format(rk, hk)
        fx_axes["Y"].text(0.1+bar_w+0.1, y+bar_h/2, txt, fontsize=fs)
        y -= 1

plt.tight_layout()
# consolidate as single image in numpy format
io_buf = io.BytesIO()
fx_fig.savefig(io_buf, format='raw', dpi=100)
io_buf.seek(0)
fx_img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fx_fig.bbox.bounds[3]),
                               int(fx_fig.bbox.bounds[2]), -1))
io_buf.close()
plt.close(fx_fig)

#####

fig, axes = plt.subplots(2, 1, figsize=(21.6, 21.6))
axes[0].imshow(brain_img)
axes[1].imshow(fx_img)

axes[0].set_title("A|  Resting state, top {} connections".format(top_cnx),
                  fontsize=36, pad=60, loc="left")
axes[1].set_title("B|  Estimated dPTE by condition", pad=60, loc="left",
                  fontsize=36)

axes[0].set_anchor("W")
axes[1].set_anchor("W")

for ax in axes:
    ax.axis("off")
plt.suptitle("{} directed connectivity".format(band_name[band]),
             fontsize=42)
plt.tight_layout()

plt.savefig("../images/cnx_{}.png".format(band))
plt.savefig("../images/cnx_{}.tif".format(band))


### pick out brain views for later construction of fig 2 in fig2_assemble.py
if band == "theta_0":
    brain_img = make_brain_image(views, params_brains["task"], text="",
                                 text_loc="lup", text_pan=0,
                                 orient="horizontal")
    np.save("../images/theta0_task.npy", brain_img)
if band == "alpha_0":
    brain_img = make_brain_image(views, params_brains["task"], text="",
                                 text_loc="lup", text_pan=0,
                                 orient="horizontal")
    np.save("../images/alpha0_task.npy", brain_img)
if band == "alpha_1":
    # average motor response tasks, make a Brain out of them
    mot_mat = np.stack([old_cnx_params[x] for x in stat_conds[1:4]]).mean(axis=0)
    motor_brain = plot_directed_cnx(mot_mat, labels, parc, alpha_min=None,
                                    alpha_max=None, ldown_title="",
                                    top_cnx=top_cnx, figsize=figsize,
                                    background=background,
                                    text_color=text_color)
    brain_img = make_brain_image(views, motor_brain, text="", text_loc="lup",
                                 text_pan=0, orient="horizontal")
    np.save("../images/alpha1_motor.npy", brain_img)

    brain_img = make_brain_image(views, params_brains["task"], text="",
                                 text_loc="lup", text_pan=0,
                                 orient="horizontal")
    np.save("../images/alpha1_task.npy", brain_img)

    brain_img = make_brain_image(views, params_brains["zaehlen"], text="",
                                 text_loc="lup", text_pan=0,
                                 orient="horizontal")
    np.save("../images/alpha1_zaehlen.npy", brain_img)
