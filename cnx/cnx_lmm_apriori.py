import numpy as np
import mne
import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import warnings
from os.path import isdir
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.patches import Rectangle

def load_sparse(filename,convert=True,full=False,nump_type="float32"):
    with open(filename,"rb") as f:
        result = pickle.load(f)
    if convert:
        full_mat = np.zeros(result["mat_sparse"].shape[:-1] + \
          (result["mat_res"],result["mat_res"])).astype(nump_type)
        full_mat[...,result["mat_inds"][0],result["mat_inds"][1]] = \
          result["mat_sparse"]
        result = full_mat
    return result

def dpte_bar(val_dict, xlim=(0.49, 0.51), bar_h=0.3, colors=None):
    fig, ax = plt.subplots(1, figsize=(12.8, 8))
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = [.5, 1.5]
    ax.set_yticks(yticks, labels=[])
    xticks = np.arange(xlim[0], xlim[1], 0.005)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=24)

    task_color = "tab:orange"

    for idx, (k, v) in enumerate(val_dict.items()):
        rest_CIs = v["Rest_CIs"]
        rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2),
                          rest_CIs[1]-rest_CIs[0], bar_h, color="gray",
                          alpha=0.5)
        ax.add_patch(rect)
        ax.vlines(v["Rest"], yticks[idx]-bar_h/2, yticks[idx]+bar_h/2,
                  color="gray", alpha=0.9)

        task_CIs = v["Task_CIs"] + v["Rest"]
        rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2),
                          task_CIs[1]-task_CIs[0], bar_h, color=task_color,
                          alpha=0.5)
        ax.add_patch(rect)
        ax.vlines(v["Task"] + v["Rest"], yticks[idx]-bar_h/2,
                  yticks[idx]+bar_h/2, color=task_color, alpha=0.9)

        ax.text(xlim[0] + 0.0005, yticks[idx], k, fontsize=28, va="center")

        ax.vlines(0.5, 0, C, color="black")

    return fig, ax


# get command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--band', type=str, required=True)
opt = parser.parse_args()

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)

paths = {}
paths["theta_0"] = {"Left motor":{"from":["L3395-lh"], "to":["all"]},
                    "Right frontal pole":{"from":["L1869-rh"], "to":["all"]}
                   }

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33", "ATT_34",
         "ATT_35", "ATT_36", "ATT_37"]

if isdir("/home/jev"):
    root_dir = "/home/jev/ATT_dat/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir+"proc/"

# parameters and setup
proc_dir = root_dir + "proc/"
lmm_dir = root_dir + "lmm/"
conds = ["rest", "audio", "visual", "visselten", "zaehlen"]
band = opt.band

with open("{}{}/cnx_params_{}.pickle".format(lmm_dir, band, band), "rb") as f:
    cnx_params = pickle.load(f)

data = []
predictor_vars = ("Subj","Block")
dm_simple = pd.DataFrame(columns=predictor_vars)
dm_cond = dm_simple.copy()
group_id = []
for sub_idx,sub in enumerate(subjs):
    for cond_idx,cond in enumerate(conds):
        # we actually only need the dPTE to get the number of trials
        data_temp = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                cond, band))
        for epo_idx in range(data_temp.shape[0]):
            c = cond if cond == "rest" else "task"
            dm_simple = dm_simple.append({"Subj":sub, "Block":c}, ignore_index=True)
            dm_cond = dm_cond.append({"Subj":sub, "Block":cond}, ignore_index=True)
            data.append(data_temp[epo_idx,])
            group_id.append(sub_idx)
data = np.array(data)
group_id = np.array(group_id)

formula = "Brain ~ C(Block, Treatment('rest'))"
if band == "theta_0":
    these_data = data.copy()
    df = dm_simple.copy()
    path = paths[band]
    these_data *= cnx_params["task"].astype(bool) # mask
    these_data[these_data==0] = np.nan # so we can use nanmean to ignore 0s
    mod_ests = {}
    for k, v in path.items():
        if v["from"][0] == "all":
            from_inds = np.arange(mat_n)
        else:
            from_inds = np.array([label_names.index(x) for x in v["from"]])
        from_mat = these_data[:, from_inds,]
        from_mat = np.nanmean(from_mat, axis=1)

        if v["to"][0] == "all":
            to_inds = np.arange(mat_n)
        else:
            to_inds = np.array([label_names.index(x) for x in v["to"]])
        to_mat = from_mat[:, to_inds]
        quant = np.nanmean(to_mat, axis=1)

        df["Brain"] = quant
        model = MixedLM.from_formula(formula, df, groups=group_id)
        mod_fit = model.fit(reml=False)
        print(mod_fit.summary())
        stat_cond = "C(Block, Treatment('rest'))[T.task]"
        CIs = mod_fit.conf_int()
        mod_ests[k] = {"Rest":mod_fit.params["Intercept"],
                       "Task":mod_fit.params[stat_cond],
                       "Rest_CIs":np.array([CIs[0]["Intercept"],
                                            CIs[1]["Intercept"]]),
                       "Task_CIs":np.array([CIs[0][stat_cond],
                                            CIs[1][stat_cond]])
                      }
