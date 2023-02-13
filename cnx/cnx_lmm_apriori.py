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

def CI_arr(CIs, cond):
    return np.array([CIs[0][cond], CIs[1][cond]])

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

def dpte_bar(val_dict, xlim=(-0.2, 0.2), bar_h=0.2, colors=None,
             task="task", task_name={"task":"Task"},
             figsize=(19.2, 19.2), leg_loc="upper right"):
    fig, ax = plt.subplots(1, figsize=figsize)
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = np.arange((1/C), C+(1/C))
    if C == 1:
        yticks = [.5]
    ax.set_yticks([])
    xticks = np.linspace(xlim[0], xlim[1]+0.0001, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:.1f}".format(xt) for xt in xticks], fontsize=50)
    ax.set_xlabel("estimated regional total dPTE-0.5", fontsize=50)

    task_color = "tab:orange"
    fs = 52

    for pure_idx, (k, v) in enumerate(val_dict.items()):
        idx = C - pure_idx - 1
        rest_CIs = v["Rest_{}_CIs".format(task)]
        rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2),
                          rest_CIs[1]-rest_CIs[0], bar_h, color="gray",
                          alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v["Rest_{}".format(task)],
                  yticks[idx]-bar_h/2, yticks[idx]+bar_h/2,
                  color="gray", alpha=0.9)

        task_CIs = (v["{}_CIs".format(task)] + v["Rest_{}".format(task)])[0]
        rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2),
                          task_CIs[1]-task_CIs[0], bar_h, color=task_color,
                          alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v[task] + v["Rest_{}".format(task)], yticks[idx]-bar_h/2,
                  yticks[idx]+bar_h/2, color=task_color, alpha=0.9)

        if max((v[task] + v["Rest_{}".format(task)]),
                v["Rest_{}".format(task)]) > 0.:
            ax.text(xlim[0] + 0.0005, yticks[idx]+bar_h, k,
                    va="center", size=fs)
        else:
            ax.text(xlim[1] - 0.0005, yticks[idx]+bar_h, k,
                    va="center", ha="right", size=fs)

    ax.vlines(0., 0, C, color="black")
    leg_lines = [plt.Line2D([0], [0], color="gray", lw=10),
                 plt.Line2D([0], [0], color=task_color, lw=10)]
    ax.legend(leg_lines, ["Rest", task_name[task]], fontsize=fs, loc=leg_loc)

    return fig, ax

def dpte_bar_multi(val_dict, conds, cond_names, xlim=(-0.3, 0.3),
                   bar_h=0.1, colors=None, figsize=(19.2, 19.2),
                   leg_loc="upper right"):
    conds = conds.copy()
    conds.reverse()
    fig, ax = plt.subplots(1, figsize=figsize)
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = [0.25, 1.5][:C]
    ax.set_yticks([])
    xticks = np.linspace(xlim[0], xlim[1]+0.0001, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:.2f}".format(xt) for xt in xticks], fontsize=50)
    ax.set_xlabel("estimated regional total dPTE-0.5", fontsize=50)

    colors = ["tab:green", "tab:purple", "tab:pink", "tab:cyan"]
    fs = 40

    for pure_idx, (k, v) in enumerate(val_dict.items()):
        idx = C - pure_idx - 1

        task_dptes = []
        for m_idx, (cond, color) in enumerate(zip(conds, colors)):
            rest_CIs = v["Rest_{}_CIs".format(cond)]
            rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2+bar_h*m_idx),
                              rest_CIs[1]-rest_CIs[0], bar_h, color="gray",
                              alpha=0.3)
            ax.add_patch(rect)
            ax.vlines(v["Rest_{}".format(cond)],
                      yticks[idx]-bar_h/2+bar_h*m_idx,
                      yticks[idx]-bar_h/2+bar_h*m_idx+bar_h,
                      color="gray", alpha=0.9)

            task_CIs = (v["{}_CIs".format(cond)] + v["Rest_{}".format(cond)])[0]
            rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2+bar_h*m_idx),
                              task_CIs[1]-task_CIs[0], bar_h, color=color,
                              alpha=0.3)
            ax.add_patch(rect)
            ax.vlines(v[cond] + v["Rest_{}".format(cond)],
                      yticks[idx]-bar_h/2+bar_h*m_idx,
                      yticks[idx]-bar_h/2+bar_h*m_idx+bar_h, color=color,
                      alpha=0.9)
            task_dptes.append(v[cond] + v["Rest_{}".format(cond)])

        max_amp = task_dptes[np.abs(task_dptes).argmax()]

        if max(max_amp, v["Rest_{}".format(cond)]) > 0.:
            ax.text(xlim[0] + 0.0005, yticks[idx]+bar_h*len(conds)/2-bar_h/2+bar_h,
                    k, fontsize=fs, va="center")
        else:
            ax.text(xlim[1] - 0.0005, yticks[idx]+bar_h*len(conds)/2-bar_h/2+bar_h,
                    k, fontsize=fs, va="center", ha="right")

    leg_lines = [plt.Line2D([0], [0], color="gray", lw=10)]
    colors.reverse()
    leg_lines += [plt.Line2D([0], [0], color=c, lw=10) for c in colors]
    ax.legend(leg_lines, ["Rest"] + cond_names, fontsize=fs, loc=leg_loc)
    ax.vlines(0., 0, C, color="black")

    return fig, ax

def dpte_bar_multi_1rest(val_dict, conds, cond_names, xlim=(-0.3, 0.3),
                         bar_h=0.1, colors=None, figsize=(19.2, 19.2),
                         leg_loc="upper right"):
    conds = conds.copy()
    conds.reverse()
    fig, ax = plt.subplots(1, figsize=figsize)
    C = len(val_dict)
    ax.set_ylim(0, C)
    ax.set_xlim(xlim)
    yticks = [0.25, 1.5][:C]
    ax.set_yticks([])
    xticks = np.linspace(xlim[0], xlim[1]+0.0001, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:.2f}".format(xt) for xt in xticks], fontsize=50)
    ax.set_xlabel("estimated regional total dPTE-0.5", fontsize=50)

    colors = ["tab:green", "tab:purple", "tab:pink"]
    fs = 40

    for pure_idx, (k, v) in enumerate(val_dict.items()):
        idx = C - pure_idx - 1
        rest_CIs = v["Rest_CIs"]
        rect = Rectangle((rest_CIs[0], yticks[idx]-bar_h/2),
                          rest_CIs[1]-rest_CIs[0], bar_h*len(conds),
                          color="gray", alpha=0.3)
        ax.add_patch(rect)
        ax.vlines(v["Rest"],
                  yticks[idx]-bar_h/2,
                  yticks[idx]-bar_h/2+bar_h*len(conds),
                  color="gray", alpha=0.9)

        task_dptes = []
        for m_idx, (cond, color) in enumerate(zip(conds, colors)):
            task_CIs = (v["{}_CIs".format(cond)])
            rect = Rectangle((task_CIs[0], yticks[idx]-bar_h/2+bar_h*m_idx),
                              task_CIs[1]-task_CIs[0], bar_h, color=color,
                              alpha=0.3)
            ax.add_patch(rect)
            ax.vlines(v[cond],
                      yticks[idx]-bar_h/2+bar_h*m_idx,
                      yticks[idx]-bar_h/2+bar_h*m_idx+bar_h, color=color,
                      alpha=0.9)
            task_dptes.append(v[cond])

        max_amp = task_dptes[np.abs(task_dptes).argmax()]

        if max(max_amp, v["Rest"]) > 0.:
            ax.text(xlim[0] + 0.0005, yticks[idx]+bar_h*len(conds)/2-bar_h/2+bar_h,
                    k, fontsize=fs, va="center")
        else:
            ax.text(xlim[1] - 0.0005, yticks[idx]+bar_h*len(conds)/2-bar_h/2+bar_h,
                    k, fontsize=fs, va="center", ha="right")

    leg_lines = [plt.Line2D([0], [0], color="gray", lw=10)]
    colors.reverse()
    leg_lines += [plt.Line2D([0], [0], color=c, lw=10) for c in colors]
    ax.legend(leg_lines, ["Rest"] + cond_names, fontsize=fs, loc=leg_loc)
    ax.vlines(0., 0, C, color="black")

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
paths["theta_0_t"] = {"LH motor":{"from":["L3395-lh"], "to":["all"]},
                    "RH frontal pole":{"from":["L1869-rh"], "to":["all"]}
                     }
paths["theta_0_c"] = {"RH dorsal parietal":{"from":["L4557_L2996-rh"],
                      "to":["all"]}
                     }
paths["alpha_0_t"] = {"LH occipital hub":{"from":["L2340-lh"], "to":["all"]},
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
paths["alpha_0_c"] = {"LH mid-temporal":{"from":["L7097_L4359-lh"], "to":["all"]},
                      "RH V1":{"from":["L2340_L1933-rh"], "to":["all"]}
                      }
paths["alpha_1_t"] = {"LH parietal":{"from":["L4557-lh"], "to":["all"]}}
paths["alpha_1_m"] = {"LH Motor/\nParietal":{"from":["L3395-lh", "L8143-lh",
                                             "L7491_L4557-lh", "L4557-lh"],
                                       "to":["all"]}
                     }
paths["alpha_1_z"] = {"LH TPJ":{"from":["L8729_L7491-lh",
                                        "L7491_L5037-lh"], "to":["all"]},
                      "RH TPJ":{"from":["L5037-rh"], "to":["all"]},
                      "LH parietal":{"from":["L4557-lh"], "to":["all"]}
                     }
paths["alpha_1_LA1"] = {"LH M/P\u2192LH A1":{"from":["L3395-lh", "L8143-lh",
                                             "L7491_L4557-lh", "L4557-lh"],
                                             "to":["L2235-lh"]}
                       }
paths["alpha_1_LV1"] = {"LH M/P\u2192LH V1":{"from":["L3395-lh", "L8143-lh",
                                             "L7491_L4557-lh", "L4557-lh"],
                                             "to":["L2340_L1933-lh"]}
                       }
paths["alpha_1_m_AV"] = {"LH M/P\u2192LH V1":{"from":["L3395-lh", "L8143-lh",
                                             "L7491_L4557-lh", "L4557-lh"],
                                             "V_to":["L2340_L1933-lh"],
                                             "A_to":["L2235-lh", "L7755-lh"]}
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
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
stat_conds = ["Intercept"] + [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names
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
            data_temp[epo_idx, data_temp[epo_idx]!=0] -= 0.5
            data_temp[epo_idx, ] += data_temp[epo_idx, ].T * -1
            data.append(data_temp[epo_idx,])
            group_id.append(sub_idx)
data = np.array(data)
group_id = np.array(group_id)

formula = "Brain ~ C(Block, Treatment('rest'))"
mod_ests = {}
if path in ["theta_0_t", "alpha_0_t", "alpha_1_t"]:
    these_data = data.copy()
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
        if v["to"][0] == "all":
            to_inds = np.arange(mat_n)
        else:
            to_inds = np.array([label_names.index(x) for x in v["to"]])
        to_mat = from_mat[:, :, to_inds]

        quant = np.nansum(to_mat.reshape(len(to_mat), -1), axis=1)

        df["Brain"] = quant
        model = MixedLM.from_formula(formula, df, groups=group_id)
        mod_fit = model.fit(reml=False)
        print(mod_fit.summary())
        stat_cond = "C(Block, Treatment('rest'))[T.task]"
        CIs = mod_fit.conf_int()
        mod_ests[k] = {"Rest_task":mod_fit.params["Intercept"],
                       "task":mod_fit.params[stat_cond],
                       "Rest_task_CIs":np.array([CIs[0]["Intercept"],
                                            CIs[1]["Intercept"]]),
                       "task_CIs":np.array([CIs[0][stat_cond],
                                            CIs[1][stat_cond]])
                      }
    if path == "theta_0_t":
        xlim = [-0.2, 0.2]
    elif path == "alpha_0_t":
        xlim = [-0.5, 2]
    elif path == "alpha_1_t":
        xlim = [-0.5, 0.5]
    fig, ax = dpte_bar(mod_ests, xlim=xlim)
elif path == "alpha_1_m_AV":
    formula = "Brain ~ C(Block, Treatment('rest'))*C(Dest, Treatment('A'))"
    df = dm_cond.copy()
    blocks = df["Block"].values
    these_data = data.copy()
    valid_inds = (blocks != "zaehlen")
    these_data = these_data[valid_inds,]

    df = df[valid_inds]
    v = list(paths[path].values())[0]
    if v["from"][0] == "all":
        from_inds = np.arange(mat_n)
    else:
        from_inds = np.array([label_names.index(x) for x in v["from"]])
    from_mat = these_data[:, from_inds,]

    dfs = []
    for to_idx, this_to in enumerate(["A_to", "V_to"]):
        this_df = df.copy()
        if v[this_to][0] == "all":
            to_inds = np.arange(mat_n)
        else:
            to_inds = np.array([label_names.index(x) for x in v[this_to]])
        to_mat = from_mat[:, :, to_inds]
        quant = np.nansum(to_mat.reshape(len(to_mat), -1), axis=1)
        this_df["Brain"] = quant
        dest = np.empty(len(quant), dtype=str)
        dest[:] = this_to
        this_df["Dest"] = dest
        dfs.append(this_df)
    big_df = pd.concat(dfs)
    groups = big_df["Subj"].values
    model = MixedLM.from_formula(formula, big_df, groups=groups)
    fit = model.fit()
    CIs = fit.conf_int()

    me = {}
    me["Rest"] = fit.params["Intercept"]
    me["audio"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.audio]"]
    me["visual"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.visual]"]
    me["visselten"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.visselten]"]
    me["Rest_CIs"] = CI_arr(CIs, "Intercept")
    me["audio_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.audio]") + me["Rest"]
    me["visual_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.visual]") + me["Rest"]
    me["visselten_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.visselten]") + me["Rest"]
    mod_ests["LH M/P\u2192LH A1"] = me.copy()

    me = {}
    v_fx = fit.params["C(Dest, Treatment('A'))[T.V]"]
    CI_base = fit.params["Intercept"] + v_fx
    me["Rest"] = fit.params["Intercept"] + fit.params["C(Dest, Treatment('A'))[T.V]"]
    me["audio"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.audio]"] + v_fx + fit.params["C(Block, Treatment('rest'))[T.audio]:C(Dest, Treatment('A'))[T.V]"]
    me["visual"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.visual]"] + v_fx + fit.params["C(Block, Treatment('rest'))[T.visual]:C(Dest, Treatment('A'))[T.V]"]
    me["visselten"] = fit.params["Intercept"] + fit.params["C(Block, Treatment('rest'))[T.visselten]"] + v_fx + fit.params["C(Block, Treatment('rest'))[T.visselten]:C(Dest, Treatment('A'))[T.V]"]
    me["Rest_CIs"] = CI_arr(CIs, "Intercept") + fit.params["C(Dest, Treatment('A'))[T.V]"]
    me["audio_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.audio]:C(Dest, Treatment('A'))[T.V]") + CI_base + fit.params["C(Block, Treatment('rest'))[T.audio]"]
    me["visual_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.visual]:C(Dest, Treatment('A'))[T.V]") + CI_base + fit.params["C(Block, Treatment('rest'))[T.visual]"]
    me["visselten_CIs"] = CI_arr(CIs, "C(Block, Treatment('rest'))[T.visselten]:C(Dest, Treatment('A'))[T.V]") + CI_base + fit.params["C(Block, Treatment('rest'))[T.visselten]"]
    mod_ests["LH M/P\u2192LH V1"] = me

else:
    this_path = paths[path]
    for k, v in this_path.items():
        mod_ests[k] = {}
        for cond, stat_cond in zip(conds[1:], stat_conds[1:]):
            df = dm_cond.copy()
            groups = group_id.copy()
            blocks = df["Block"].values
            these_data = data.copy()
            valid_inds = (blocks == "rest") | (blocks == cond)
            these_data = these_data[valid_inds,]
            groups = groups[valid_inds]
            df = df[valid_inds]

            # masking
            these_data *= cnx_params[stat_cond].astype(bool) # mask
            these_data[these_data==0] = np.nan # so we can use nanmean to ignore 0s

            if v["from"][0] == "all":
                from_inds = np.arange(mat_n)
            else:
                from_inds = np.array([label_names.index(x) for x in v["from"]])
            from_mat = these_data[:, from_inds,]
            if v["to"][0] == "all":
                to_inds = np.arange(mat_n)
            else:
                to_inds = np.array([label_names.index(x) for x in v["to"]])
            to_mat = from_mat[:, :, to_inds]

            quant = np.nansum(to_mat.reshape(len(to_mat), -1), axis=1)
            df["Brain"] = quant
            model = MixedLM.from_formula(formula, df, groups=groups)
            if path == "theta_0_c":
                reml = False
            else:
                reml = True
            mod_fit = model.fit(reml=reml)
            print(mod_fit.summary())
            CIs = mod_fit.conf_int()
            mod_ests[k]["Rest_{}".format(cond)] = mod_fit.params["Intercept"]
            mod_ests[k]["Rest_{}_CIs".format(cond)] = np.array([CIs[0]["Intercept"],
                                                                CIs[1]["Intercept"]])
            mod_ests[k][cond] = mod_fit.params[stat_cond]
            mod_ests[k]["{}_CIs".format(cond)] = np.array([CIs[0][stat_cond],
                                                           CIs[1][stat_cond]]),

conds = ["audio", "visual", "visselten", "zaehlen"]
cond_names = ["Audio", "Visual", "Vis. w/distr.", "Counting"]
if path == "alpha_1_z":
    fig, ax = dpte_bar(mod_ests, task="zaehlen",
                       task_name={"zaehlen":"Counting"},
                       xlim=(-0.2, 0.5))
elif path == "alpha_1_m":
    fig, ax = dpte_bar_multi(mod_ests, conds, cond_names, leg_loc=(0.65, 0.65),
                             xlim=(-0.5, 1.2))
elif path == "theta_0_c":
    fig, ax = dpte_bar_multi(mod_ests, conds, cond_names, leg_loc=(0.55, 0.6))
elif path == "alpha_0_c":
    fig, ax = dpte_bar_multi(mod_ests, conds, cond_names, leg_loc=(0, 0.35))
elif path == "alpha_1_LA1" or path == "alpha_1_LV1":
    fig, ax = dpte_bar_multi(mod_ests, conds, cond_names, leg_loc=(0.65, 0.65),
                             xlim=(-0.06, 0.06))
elif path == "alpha_1_m_AV":
    conds = ["audio", "visual", "visselten"]
    cond_names = ["Audio", "Visual", "Vis. w/distr."]
    fig, ax = dpte_bar_multi_1rest(mod_ests, conds, cond_names, leg_loc=(0., 0.35),
                             xlim=(-0.1, 0.1))
    ax.set_title("Motor/Parietal connectivity to A1 and V1", fontsize=38)
    plt.savefig("../images/fig_5.tif")
    plt.savefig("../images/fig_5.png")

# consolidate as single image in numpy format
io_buf = io.BytesIO()
fig.savefig(io_buf, format='raw', dpi=100)
io_buf.seek(0)
img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
io_buf.close()

np.save("../images/params_bar_{}.npy".format(path), img_arr)
