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
            mod_fit = model.fit()
        except:
            mods.append(None)
            continue
        mods.append(mod_fit)
    return mods

parser = argparse.ArgumentParser()
parser.add_argument('--perm', type=int, default=500)
parser.add_argument('--iter', type=int, default=0)
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
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
band = opt.band
indep_var = None
perm_n = opt.perm

df_laut = pd.read_pickle(root_dir+"behave/laut")
df_ang = pd.read_pickle(root_dir+"behave/ang")

predictor_vars = ["Laut","Subj","Block","Wav"]
dm_laut = df_laut.copy()[predictor_vars]

predictor_vars = ["Angenehm","Subj","Block","Wav"]
predictor_vars = ["Subj","Block"]
dm_ang = df_ang.copy()[predictor_vars]

if indep_var == "Angenehm":
    dm = dm_ang
elif indep_var == "Laut":
    dm = dm_laut
else:
    dm = None

data = []
dm_new = pd.DataFrame(columns=predictor_vars)
idx_borders = []
idx_border = 0
group_id = []
for sub_idx,sub in enumerate(subjs):
    idx_borders.append([idx_border])
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        data_temp = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir,
                                                                sub, cond,
                                                                band))
        for epo_idx in range(data_temp.shape[0]):
            if indep_var:
                sel_inds = (dm["Block"]==cond) & (dm["Subj"]==sub)
                dm_new = dm_new.append(dm[sel_inds])
            else:
                dm_new = dm_new.append({"Subj":sub, "Block":cond}, ignore_index=True)
            data.append(phi(data_temp[epo_idx,], k=1))
            group_id.append(sub_idx)
            idx_border += 1
    idx_borders[-1].append(idx_border)
data = np.array(data)
group_id = np.array(group_id)
if indep_var:
    col_idx = dm_new.columns.get_loc(indep_var)
else:
    col_idx = 1

if indep_var:
    formula = "Brain ~ {} + Block".format(indep_var)
else:
    formula = "Brain ~ C(Block, Treatment('rest'))"

# # get main result
# mods = mass_uv_mixedlmm(formula, dm_new, data, group_id)
# for mod_idx,mod in enumerate(mods):
#     mod.save("{}{}/reg70_lmm_{}.pickle".format(proc_dir,opt.band,mod_idx))

# permute
all_perm_tvals = [[],[],[],[]]
for i in range(perm_n):
    print("Permutation {} of {}".format(i, perm_n))
    dm_perm = dm_new.copy()
    for idx_border in idx_borders:
        if indep_var:
            temp_slice = dm_perm[indep_var][idx_border[0]:idx_border[1]].copy()
        else:
            temp_slice = dm_perm["Block"][idx_border[0]:idx_border[1]].copy()
        temp_slice = temp_slice.sample(frac=1)
        dm_perm.iloc[idx_border[0]:idx_border[1],col_idx] = temp_slice.values
    perm_mods = mass_uv_mixedlmm(formula, dm_perm, data, group_id)
    for pm in perm_mods:
        all_perm_tvals[0].append(pm.tvalues.get("C(Block, Treatment('rest'))[T.audio]"))
        all_perm_tvals[1].append(pm.tvalues.get("C(Block, Treatment('rest'))[T.visselten]"))
        all_perm_tvals[2].append(pm.tvalues.get("C(Block, Treatment('rest'))[T.visual]"))
        all_perm_tvals[3].append(pm.tvalues.get("C(Block, Treatment('rest'))[T.zaehlen]"))
all_perm_tvals = np.array(all_perm_tvals)
np.save("{}cnx_{}_{}_perm_{}_{}.npy".format(proc_dir, indep_var, band, perm_n, opt.iter),
        all_perm_tvals)
