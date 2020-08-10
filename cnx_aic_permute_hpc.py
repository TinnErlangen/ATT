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
    print("\n")
    return mods

parser = argparse.ArgumentParser()
parser.add_argument('--perm', type=int, default=500)
parser.add_argument('--band', type=str, required=True)
parser.add_argument('--iter', type=int, default=0)
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
root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir + "proc/"
spacing = "ico4"
conds = ["rest","audio","visual","visselten","zaehlen"]
band = opt.band
perm_n = opt.perm
node_n = 2415
models = ["null","simple","cond"]

'''
build up the dataframes and group_id which will eventually be passed to
mass_uv_lmm. We will build two models here. "Simple" will make only one contrast:
resting state and task. "Cond" makes distinctions for the different conditions.
'''
print("Building tables...", end="")
data = []
predictor_vars = ("Subj","Block")
dm_simple = pd.DataFrame(columns=predictor_vars)
dm_cond = dm_simple.copy()
group_id = []
idx_borders = []
idx_border = 0
for sub_idx,sub in enumerate(subjs):
    idx_borders.append([idx_border])
    for cond_idx,cond in enumerate(conds):
        # we actually only need the dPTE to get the number of trials
        data_temp = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                cond, band))
        for epo_idx in range(data_temp.shape[0]):
            c = cond if cond == "rest" else "task"
            dm_simple = dm_simple.append({"Subj":sub, "Block":c}, ignore_index=True)
            dm_cond = dm_cond.append({"Subj":sub, "Block":cond}, ignore_index=True)
            data.append(phi(data_temp[epo_idx,], k=1)) # flatten upper diagonal of connectivity matrix, add it to the list
            group_id.append(sub_idx)
            idx_border += 1
    idx_borders[-1].append(idx_border)
data = np.array(data)
group_id = np.array(group_id)
print("done.")

aics = {mod:np.zeros((node_n,perm_n)) for mod in models}

if opt.iter == 0:
    formula = "Brain ~ 1"
    mods_null = mass_uv_mixedlmm(formula, dm_simple, data, group_id)
    for n_idx in range(node_n):
        aics["null"][n_idx,] = np.broadcast_to(mods_null[n_idx].aic, perm_n)
    del mods_null

# permute
for perm_idx in range(perm_n):
    print("Permutation {} of {}".format(perm_idx, perm_n))
    print("Simple")
    formula = "Brain ~ C(Block, Treatment('rest'))"
    dm_perm = dm_simple.copy()
    col_idx = dm_perm.columns.get_loc("Block")
    for idx_border in idx_borders:
        temp_slice = dm_perm["Block"][idx_border[0]:idx_border[1]].copy()
        temp_slice = temp_slice.sample(frac=1)
        dm_perm.iloc[idx_border[0]:idx_border[1],col_idx] = temp_slice.values
    mods_simple = mass_uv_mixedlmm(formula, dm_perm, data, group_id)
    for n_idx in range(node_n):
        aics["simple"][n_idx,perm_idx] = mods_simple[n_idx].aic
    del mods_simple
    print("Cond")
    formula = "Brain ~ C(Block, Treatment('rest'))"
    dm_perm = dm_cond.copy()
    col_idx = dm_perm.columns.get_loc("Block")
    for idx_border in idx_borders:
        temp_slice = dm_perm["Block"][idx_border[0]:idx_border[1]].copy()
        temp_slice = temp_slice.sample(frac=1)
        dm_perm.iloc[idx_border[0]:idx_border[1],col_idx] = temp_slice.values
    mods_cond = mass_uv_mixedlmm(formula, dm_perm, data, group_id)
    for n_idx in range(node_n):
        aics["cond"][n_idx,perm_idx] = mods_cond[n_idx].aic
    del mods_cond

with open("{}{}/perm_aic_{}.pickle".format(proc_dir,band,opt.iter), "wb") as f:
    pickle.dump(aics,f)
