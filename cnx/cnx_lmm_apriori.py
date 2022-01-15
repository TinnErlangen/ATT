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
import io

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

def dpte_bar(val_dict, xlim=(0.488, 0.512), bar_h=0.2, colors=None,
             task="Task", task_name={"Task":"Task"},
             figsize=(19.2, 19.2), leg_loc="upper right"):
    fig, ax = plt.subplots(1, figsize=figsize)
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = np.arange((1/C), C+(1/C))
    if C == 1:
        yticks = [.5]
    ax.set_yticks([])
    xticks = np.arange(xlim[0], xlim[1], 0.005)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=50)
    ax.set_xlabel("estimated regional mean dPTE", fontsize=50)

    task_color = "tab:orange"
    fs = 52

    for pure_idx, (k, v) in enumerate(val_dict.items()):
        idx = C - pure_idx - 1
        rest_CIs = v["Rest_CIs"]
        rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2),
                          rest_CIs[1]-rest_CIs[0], bar_h, color="gray",
                          alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v["Rest"], yticks[idx]-bar_h/2, yticks[idx]+bar_h/2,
                  color="gray", alpha=0.9)

        task_CIs = v["{}_CIs".format(task)] + v["Rest"]
        rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2),
                          task_CIs[1]-task_CIs[0], bar_h, color=task_color,
                          alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v[task] + v["Rest"], yticks[idx]-bar_h/2,
                  yticks[idx]+bar_h/2, color=task_color, alpha=0.9)

        if max((v[task] + v["Rest"]), v["Rest"]) > 0.5:
            ax.text(xlim[0] + 0.0005, yticks[idx], k,
                    va="center", size=fs)
        else:
            ax.text(xlim[1] - 0.0005, yticks[idx], k,
                    va="center", ha="right", size=fs)

    ax.vlines(0.5, 0, C, color="black")
    leg_lines = [plt.Line2D([0], [0], color="gray", lw=10),
                 plt.Line2D([0], [0], color=task_color, lw=10)]
    ax.legend(leg_lines, ["Rest", task_name[task]], fontsize=fs, loc=leg_loc)

    return fig, ax

def dpte_bar_multi(val_dict, conds, cond_names, xlim=(0.488, 0.512),
                   bar_h=0.1, colors=None, figsize=(19.2, 19.2),
                   leg_loc="upper right"):
    fig, ax = plt.subplots(1, figsize=figsize)
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = [0.25, 1.5][:C]
    ax.set_yticks([])
    xticks = np.arange(xlim[0], xlim[1], 0.005)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=50)
    ax.set_xlabel("estimated regional mean dPTE", fontsize=50)

    rest_bar_h = bar_h * len(conds)
    colors = ["tab:green", "tab:purple", "tab:pink", "tab:cyan"]
    fs = 52

    for pure_idx, (k, v) in enumerate(val_dict.items()):
        idx = C - pure_idx - 1
        rest_CIs = v["Rest_CIs"]
        rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2),
                          rest_CIs[1]-rest_CIs[0], rest_bar_h, color="gray",
                          alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v["Rest"], yticks[idx]-bar_h/2,
                  yticks[idx]-bar_h/2 + rest_bar_h, color="gray", alpha=0.9)

        task_dptes = []
        for m_idx, (cond, color) in enumerate(zip(conds, colors)):
            task_CIs = v["{}_CIs".format(cond)] + v["Rest"]
            rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2+bar_h*m_idx),
                              task_CIs[1]-task_CIs[0], bar_h, color=color,
                              alpha=0.3)
            ax.add_patch(rect)
            ax.vlines(v[cond] + v["Rest"],
                      yticks[idx]-bar_h/2+bar_h*m_idx,
                      yticks[idx]-bar_h/2+bar_h*m_idx+bar_h, color=color,
                      alpha=0.9)
            task_dptes.append(v[cond] + v["Rest"])

        max_amp = task_dptes[np.abs(task_dptes).argmax()]

        if max(max_amp, v["Rest"]) > 0.5:
            ax.text(xlim[0] + 0.0005, yticks[idx]-bar_h/2+rest_bar_h/2, k,
                    fontsize=fs, va="center")
        else:
            ax.text(xlim[1] - 0.0005, yticks[idx]-bar_h/2+rest_bar_h/2, k,
                    fontsize=fs, va="center", ha="right")

    leg_lines = [plt.Line2D([0], [0], color="gray", lw=10)]
    colors.reverse()
    leg_lines += [plt.Line2D([0], [0], color=c, lw=10) for c in colors]
    ax.legend(leg_lines, ["Rest"] + cond_names, fontsize=fs, loc=leg_loc)
    ax.vlines(0.5, 0, C, color="black")

    return fig, ax

# get command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--band', type=str, required=True)
parser.add_argument('--path', type=str, default="same")
opt = parser.parse_args()

path = opt.band if opt.path == "same" else opt.path

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)

paths = {}
paths["theta_0"] = {"LH motor":{"from":["L3395-lh"], "to":["all"]},
                    "RH frontal pole":{"from":["L1869-rh"], "to":["all"]}
                   }
paths["alpha_0"] = {"LH Hub":{"from":["L2340-lh"], "to":["all"]},
                    "Other occipital":{"from":["L4236_L1933-lh",
                                               "L4236_L1933-rh",
                                               "L2340_L1933-lh",
                                               "L2340_L1933-rh",
                                               "L2685-lh", "L2685-rh",
                                               "L7097_L5106-lh",
                                               "L7097_L5106-rh",
                                               "L10017-lh", "L10017-rh"],
                                               "to":["all"]
                                              }
                   }
paths["alpha_1_t"] = {"LH parietal":{"from":["L4557-lh"], "to":["all"]}}
paths["alpha_1_m"] = {"LH parietal":{"from":["L4557-lh"], "to":["all"]},
                      "LH motor":{"from":["L3395-lh", "L8143_L7523-lh",
                                          "L8143-lh"],
                                  "to":["all"]}}
paths["alpha_1_z"] = {
                      "LH TPJ":{"from":["L8729_L7491-lh",
                                        "L7491_L5037-lh"], "to":["all"]},
                      "RH TPJ":{"from":["L5037-rh"], "to":["all"]},
                      "LH parietal":{"from":["L4557-lh"], "to":["all"]}
                     }
paths["alpha_1_LA1"] = {"LH P\u2192LH A1":{"from":["L4557-lh"], "to":["L2235-lh"]},
                       "LH M\u2192LH A1":{"from":["L3395-lh", "L8143_L7523-lh",
                                                  "L8143-lh"],
                                          "to":["L2235-lh"]}}
paths["alpha_1_LV1"] = {"LH P\u2192LH V1":{"from":["L4557-lh"], "to":["L2340_L1933-lh"]},
                       "LH M\u2192LH V1":{"from":["L3395-lh", "L8143_L7523-lh",
                                                  "L8143-lh"],
                                          "to":["L2340_L1933-lh"]}}


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
for k,v in cnx_params.items():
    cnx_params[k] = v + v.T * -1

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
mod_ests = {}
if path in ["theta_0", "alpha_0", "alpha_1_t"]:
    these_data = data.copy()
    triu_inds, tril_inds = np.triu_indices(mat_n, k=1), np.tril_indices(mat_n, k=-1)
    these_data[:, tril_inds[0], tril_inds[1]] = \
      (these_data[:, triu_inds[0], triu_inds[1]] - 0.5) * -1 + 0.5
    df = dm_simple.copy()
    this_path = paths[path]
    these_data *= cnx_params["task"].astype(bool) # mask
    these_data[these_data==0] = np.nan # so we can use nanmean to ignore 0s
    for k, v in this_path.items():
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
    fig, ax = dpte_bar(mod_ests)
else:
    these_data = data.copy()
    triu_inds, tril_inds = np.triu_indices(mat_n, k=1), np.tril_indices(mat_n, k=-1)
    these_data[:, tril_inds[0], tril_inds[1]] = \
      (these_data[:, triu_inds[0], triu_inds[1]] - 0.5) * -1 + 0.5
    df = dm_cond.copy()
    this_path = paths[path]
    # masking
    if path in ["alpha_1_m", "alpha_1_LA1", "alpha_1_LV1", "alpha_1_RA1",
                "alpha_1_RV1"]:
        mask_names = ["C(Block, Treatment('rest'))[T.audio]",
                      "C(Block, Treatment('rest'))[T.visual]",
                      "C(Block, Treatment('rest'))[T.visselten]",
                      "C(Block, Treatment('rest'))[T.zaehlen]"]
        mask = sum([cnx_params[mn].astype(bool) for mn in mask_names]).astype(bool)
    elif path == "alpha_1_z":
        mask = cnx_params["C(Block, Treatment('rest'))[T.zaehlen]"].astype(bool)
    these_data *= mask
    these_data[these_data==0] = np.nan # so we can use nanmean to ignore 0s
    for k, v in this_path.items():
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
        CIs = mod_fit.conf_int()
        mod_ests[k] = {"Rest":mod_fit.params["Intercept"],
                       "Rest_CIs":np.array([CIs[0]["Intercept"],
                                            CIs[1]["Intercept"]]),

                       "audio":mod_fit.params["C(Block, Treatment('rest'))[T.audio]"],
                       "audio_CIs":np.array([CIs[0]["C(Block, Treatment('rest'))[T.audio]"],
                                            CIs[1]["C(Block, Treatment('rest'))[T.audio]"]]),

                       "visual":mod_fit.params["C(Block, Treatment('rest'))[T.visual]"],
                       "visual_CIs":np.array([CIs[0]["C(Block, Treatment('rest'))[T.visual]"],
                                            CIs[1]["C(Block, Treatment('rest'))[T.visual]"]]),

                       "visselten":mod_fit.params["C(Block, Treatment('rest'))[T.visselten]"],
                       "visselten_CIs":np.array([CIs[0]["C(Block, Treatment('rest'))[T.visselten]"],
                                                 CIs[1]["C(Block, Treatment('rest'))[T.visselten]"]]),

                       "zaehlen":mod_fit.params["C(Block, Treatment('rest'))[T.zaehlen]"],
                       "zaehlen_CIs":np.array([CIs[0]["C(Block, Treatment('rest'))[T.zaehlen]"],
                                           CIs[1]["C(Block, Treatment('rest'))[T.zaehlen]"]]),
                      }
        conds = ["zaehlen", "visselten", "visual", "audio"]
        cond_names = ["Audio", "Visual", "Vis. w/distr.", "Counting"]
    if path == "alpha_1_z":
        fig, ax = dpte_bar(mod_ests, task="zaehlen", task_name={"zaehlen":"Counting"})
    else:
        fig, ax = dpte_bar_multi(mod_ests, conds, cond_names, leg_loc=(0, 0.35))

# consolidate as single image in numpy format
io_buf = io.BytesIO()
fig.savefig(io_buf, format='raw', dpi=100)
io_buf.seek(0)
img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
io_buf.close()

np.save("../images/params_bar_{}.npy".format(path), img_arr)
