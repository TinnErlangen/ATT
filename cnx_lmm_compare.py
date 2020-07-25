import numpy as np
import mne
from cnx_utils import load_sparse, phi
import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def mass_uv_mixedlmm(formula, data, uv_data, group_id, re_formula=None):
    mods = []
    for d_idx in range(uv_data.shape[1]):
        print("{} of {}".format(d_idx, uv_data.shape[1]), end="\r")
        data_temp = data.copy()
        data_temp["Brain"] = uv_data[:,d_idx]
        model = MixedLM.from_formula(formula, data_temp, groups=group_id)
        try:
            mod_fit = model.fit(reml=False)
        except:
            mods.append(None)
            continue
        mods.append(mod_fit)
    return mods

parser = argparse.ArgumentParser()
parser.add_argument('--band', type=str, required=True)
opt = parser.parse_args()

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33", "ATT_34",
         "ATT_35", "ATT_36", "ATT_37"]

band_info = {}
band_info["theta_0"] = {"freqs":list(np.arange(3,7)),"cycles":3}
band_info["alpha_0"] = {"freqs":list(np.arange(7,10)),"cycles":5}
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
band_info["beta_0"] = {"freqs":list(np.arange(13,22)),"cycles":9}
band_info["beta_1"] = {"freqs":list(np.arange(22,31)),"cycles":9}
band_info["gamma_0"] = {"freqs":list(np.arange(31,41)),"cycles":9}
band_info["gamma_1"] = {"freqs":list(np.arange(41,60)),"cycles":9}
band_info["gamma_2"] = {"freqs":list(np.arange(60,91)),"cycles":9}

# parameters and setup
root_dir = "/home/jeff/ATT_dat/"
#root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir + "proc/"
spacing = "ico4"
conds = ["rest","audio","visual","visselten","zaehlen"]
#conds = ["rest","audio","visual","visselten"]
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
band = opt.band

data = []
predictor_vars = ("Subj","Block")
dm_simple = pd.DataFrame(columns=predictor_vars)
dm_cond = dm_simple.copy()
group_id = []
for sub_idx,sub in enumerate(subjs):
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        data_temp = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                cond, band))
        for epo_idx in range(data_temp.shape[0]):
            c = cond if cond == "rest" else "task"
            dm_simple = dm_simple.append({"Subj":sub, "Block":c}, ignore_index=True)
            dm_cond = dm_cond.append({"Subj":sub, "Block":cond}, ignore_index=True)
            data.append(phi(data_temp[epo_idx,], k=1))
            group_id.append(sub_idx)

data = np.array(data)
group_id = np.array(group_id)

formula = "Brain ~ 1"
mods_null = mass_uv_mixedlmm(formula, dm_simple, data, group_id)
for mod_idx,mod in enumerate(mods_null):
    if mod == None:
        continue
    mod.save("{}{}/null_reg70_lmm_{}.pickle".format(proc_dir,opt.band,mod_idx))

formula = "Brain ~ C(Block, Treatment('rest'))"
mods_simple = mass_uv_mixedlmm(formula, dm_simple, data, group_id)
for mod_idx,mod in enumerate(mods_simple):
    if mod == None:
        continue
    mod.save("{}{}/simple_reg70_lmm_{}.pickle".format(proc_dir,opt.band,mod_idx))

formula = "Brain ~ C(Block, Treatment('rest'))"
mods_cond = mass_uv_mixedlmm(formula, dm_cond, data, group_id)
for mod_idx,mod in enumerate(mods_cond):
    if mod == None:
        continue
    mod.save("{}{}/cond_reg70_lmm_{}.pickle".format(proc_dir,opt.band,mod_idx))
