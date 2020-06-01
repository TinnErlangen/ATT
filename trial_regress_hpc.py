import numpy as np
import mne

import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
from mne.stats.cluster_level import _setup_connectivity, _find_clusters, \
    _reshape_clusters
import pandas as pd

def mass_uv_mixedlmm(formula, data, uv_data, group_id, re_formula=None, exclude=[]):
    tvals = []
    coeffs = []
    for d_idx in range(uv_data.shape[1]):
        if d_idx in exclude:
            tvals.append(0)
            coeffs.append(0)
            continue
        data_temp = data.copy()
        data_temp["Brain"] = uv_data[:,d_idx]
        model = MixedLM.from_formula(formula, data_temp, groups=group_id)
        mod_fit = model.fit()
        tvals.append(mod_fit.tvalues.get(indep_var))
        coeffs.append(mod_fit.params.get(indep_var))
    tvals, coeffs = np.array(tvals), np.array(coeffs)
    return tvals, coeffs

parser = argparse.ArgumentParser()
parser.add_argument('--perm', type=int, default=500)
parser.add_argument('--band', type=str, required=True)
opt = parser.parse_args()

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

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
subjects_dir = "/scratch/jeffhanna/freesurfer/subjects/"
proc_dir = "/scratch/jeffhanna/ATT_dat/proc/"
spacing = "ico4"
conds = ["audio","visselten","visual"]
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
band = opt.band
indep_var = "Angenehm"
n_freqs = 1
n_srcs = 5124
n_subjs = len(subjs)
perm_n = opt.perm

# setup connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",
                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)
del fs_src
connectivity = _setup_connectivity(cnx, n_srcs, n_freqs)
exclude = np.load("{}fsaverage_{}_exclude.npy".format(proc_dir,spacing))
include = np.ones(cnx.shape[0],dtype="bool")
include[exclude] = 0

# threshold for clustering
threshold = dict(start=0, step=0.2)
#random_state = 42
random = np.random.RandomState()

df_laut = pd.read_pickle("/scratch/jeffhanna/ATT_dat/behave/laut")
df_ang = pd.read_pickle("/scratch/jeffhanna/ATT_dat/behave/ang")

predictor_vars = ["Laut","Subj","Block","Wav"]
dm_laut = df_laut.copy()[predictor_vars]

predictor_vars = ["Angenehm","Subj","Block","Wav"]
dm_ang = df_ang.copy()[predictor_vars]

if indep_var == "Angenehm":
    dm = dm_ang
elif indep_var == "Laut":
    dm = dm_laut

data = []
dm_new = pd.DataFrame(columns=predictor_vars)
idx_borders = []
idx_border = 0
group_id = []
for sub_idx,sub in enumerate(subjs):
    idx_borders.append([idx_border])
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        for wav_idx,wav in enumerate(wavs):
            data_temp = np.load("{dir}stcs/nc_{a}_{b}_{c}_{f0}-{f1}Hz_{sp}.npy".format(dir=proc_dir,
                                                                                       a=sub,b=cond, c=wav,
                                                                                       sp=spacing,
                                                                                       f0=band_info[band]["freqs"][0],
                                                                                       f1=band_info[band]["freqs"][-1]))
            for epo_idx in range(data_temp.shape[0]):
                sel_inds = (dm["Block"]==cond) & (dm["Wav"]==wav) & (dm["Subj"]==sub)
                dm_new = dm_new.append(dm[sel_inds])
                data.append(data_temp[epo_idx,])
                group_id.append(sub_idx)
                idx_border += 1
    idx_borders[-1].append(idx_border)
data = np.array(data)[...,0]
group_id = np.array(group_id)
col_idx = dm_new.columns.get_loc(indep_var)

formula = "Brain ~ {} + Block + Wav".format(indep_var)
tvals, coeffs = mass_uv_mixedlmm(formula, dm_new, data, group_id, exclude=exclude)
# find clusters
clusters, cluster_stats = _find_clusters(tvals,threshold=threshold,connectivity=connectivity,include=include)

main_result = {"formula":formula, "tvals":tvals, "coeffs":coeffs, "cluster_stats":cluster_stats}
with open("{}dics_{}_{}_main_result".format(proc_dir, indep_var, band), "wb") as f:
    pickle.dump(main_result,f)

# permute
all_perm_cluster_stats = []
for i in range(perm_n):
    print("Permutation {} of {}".format(i, perm_n))
    dm_perm = dm_new.copy()
    for idx_border in idx_borders:
        temp_slice = dm_perm[indep_var][idx_border[0]:idx_border[1]].copy()
        temp_slice = temp_slice.sample(frac=1)
        dm_perm.iloc[idx_border[0]:idx_border[1],col_idx] = temp_slice.values
    perm_tvals, _ = mass_uv_mixedlmm(formula, dm_perm, data, group_id, exclude=exclude)
    _, perm_cluster_stats = _find_clusters(perm_tvals,threshold=threshold,
                                           connectivity=connectivity,
                                           include=include)
    all_perm_cluster_stats.append(perm_cluster_stats)
all_perm_cluster_stats = np.array(all_perm_cluster_stats)
np.save("{}dics_{}_{}_perm_{}.npy".format(proc_dir, indep_var, band, perm_n), all_perm_cluster_stats)
