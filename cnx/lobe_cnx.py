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
import seaborn as sns
plt.ion()

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
label_names_nohemi = [lab[:-3] for lab in label_names]
mat_n = len(labels)
models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "winner"] # these will form the main keys of aic_comps dictionary below
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
conds = ["rest", "audio", "visual", "visselten", "zaehlen"]
stat_conds = ["Intercept"] + [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names
node_n = 2415

band = "alpha_0"
lmm_dir = "/home/jev/ATT_dat/lmm/"

region_dict = {"Occipital":["L2340", "L2685", "L4236_L1933", "L10017",
                            "L2340_L1933"],
               "Parietal":["L4557", "L4557_L2996", "L5037", "L7491_L4557",
                           "L7491_L5037", "L8143", "L8729_L7491", "L928"],
               "Temporal":["L5106_L2688", "L5511_L4359", "L7049", "L7097_L4359",
                           "L7097_L5106", "L7755", "L2235"],
               "Central":["L1154_L1032", "L3395", "L3969", "L7550_L3015",
                          "L8143_L7523", "L1032"],
               "Frontal":["L1869", "L4118", "L4118_L2817", "L6412",
                          "L6412_L4118", "L6698_L1154", "L8983_L3015",
                          "L9249_L6698", "L2817"]
               }



with open("{}/{}/aic_perm.pickle".format(lmm_dir, band), "rb") as f:
    aic_comps = pickle.load(f)

# build up cnx_params
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
for k in cnx_params.keys():
    cnx_params[k] += cnx_params[k].T * -1

# calculate lobe connectivity
lobe = {}
for l_k, l_v in region_dict.items():
    lobe[l_k] = {}
    for cond, stat_cond in zip(conds, stat_conds):
        lobe[l_k][cond] = {}
        # within lobe connectivity
        lobe[l_k][cond]["within"] = {}
        for hemi in ["lh", "rh"]:
            total = {"pos":[], "neg":[]}
            for src_reg in l_v:
                src_idx = label_names.index("{}-{}".format(src_reg, hemi))
                for dest_reg in l_v:
                    dest_idx = label_names.index("{}-{}".format(dest_reg, hemi))
                    val = cnx_params[stat_cond][src_idx, dest_idx]
                    if val > 0:
                        total["pos"].append(val)
                    elif val < 0:
                        total["neg"].append(np.abs(val))
            for tot_k, tot_v in total.items():
                mu = np.mean(tot_v)
                mu = 0 if np.isnan(mu) else mu
                total[tot_k] = mu
            lobe[l_k][cond]["within"][hemi] = total
        # homologue connectivity
        total = {"pos":[], "neg":[]}
        for src_reg in l_v:
            src_idx = label_names.index("{}-lh".format(src_reg))
            for dest_reg in l_v:
                dest_idx = label_names.index("{}-rh".format(dest_reg))
                val = cnx_params[stat_cond][src_idx, dest_idx]
                if val > 0:
                    total["pos"].append(val)
                elif val < 0:
                    total["neg"].append(np.abs(val))
        for tot_k, tot_v in total.items():
            mu = np.mean(tot_v)
            mu = 0 if np.isnan(mu) else mu
            total[tot_k] = mu
        lobe[l_k][cond]["in lh-rh"] = total
        # without lobe connectivity
        lobe[l_k][cond]["without"] = {}
        for hemi_src in ["lh", "rh"]:
            for hemi_dest in ["lh", "rh"]:
                total = {"pos":[], "neg":[]}
                for src_reg in l_v:
                    src_idx = label_names.index("{}-{}".format(src_reg, hemi_src))
                    for dest_reg in label_names_nohemi:
                        if dest_reg in l_v:
                            continue
                        dest_idx = label_names.index("{}-{}".format(dest_reg, hemi_dest))
                        val = cnx_params[stat_cond][src_idx, dest_idx]
                        if val > 0:
                            total["pos"].append(val)
                        elif val < 0:
                            total["neg"].append(np.abs(val))
                for tot_k, tot_v in total.items():
                    mu = np.mean(tot_v)
                    mu = 0 if np.isnan(mu) else mu
                    total[tot_k] = mu
                lobe[l_k][cond]["without"]["{}-{}".format(hemi_src, hemi_dest)] = total

fig, axes = plt.subplots(len(region_dict), len(conds), figsize=(38.4, 21.6))
x = ["in lh", "in rh", "in lh-rh", "out lh-lh", "out lh-rh",
     "out rh-rh", "out rh-lh"]
posneg = ["pos" for xx in x] + ["neg" for xx in x]
x *= 2

for reg_idx, reg in enumerate(region_dict.keys()):
    for cond_idx, cond in enumerate(conds):
        val_dict = lobe[reg][cond]
        y = np.array([val_dict["within"]["lh"]["pos"],
                      val_dict["within"]["rh"]["pos"],
                      val_dict["in lh-rh"]["pos"],
                      val_dict["without"]["lh-lh"]["pos"],
                      val_dict["without"]["lh-rh"]["pos"],
                      val_dict["without"]["rh-lh"]["pos"],
                      val_dict["without"]["rh-rh"]["pos"],
                      val_dict["within"]["lh"]["neg"],
                      val_dict["within"]["rh"]["neg"],
                      val_dict["in lh-rh"]["neg"],
                      val_dict["without"]["lh-lh"]["neg"],
                      val_dict["without"]["lh-rh"]["neg"],
                      val_dict["without"]["rh-lh"]["neg"],
                      val_dict["without"]["rh-rh"]["neg"]
                     ])
        sns.barplot(x, y, hue=posneg, ax=axes[reg_idx][cond_idx],
                    palette=["red", "blue"])
        axes[reg_idx][cond_idx].set_title("{} {}".format(reg, cond))
        axes[reg_idx][cond_idx].set_ylim(0, 0.0085)
fig.suptitle(band)
