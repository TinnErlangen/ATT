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

# get command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--band', type=str, required=True)
opt = parser.parse_args()

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33", "ATT_34",
         "ATT_35", "ATT_36", "ATT_37"]

# wavelet and frequency info for each band
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
out_dir = root_dir + "lmm/"
spacing = "ico4"
conds = ["rest","audio","visual","visselten","zaehlen"]
#conds = ["rest","audio","visual","visselten"]
band = opt.band
mat_n = 70
ROIs = []

with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
    aic_comps = pickle.load(f)

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = np.zeros((mat_n,mat_n))
mod_idx = aic_comps["models"].index(mod)
for n_idx in range(node_n):
    if aic_comps["single_winner_ids"][n_idx] == mod_idx:
        cnx_masks[triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1


columns = ("Subj","Block","Region","Hemi")
dm = pd.DataFrame(columns=columns)
group_id = []
for sub_idx,sub in enumerate(subjs):
    for cond_idx,cond in enumerate(conds):
        # we actually only need the dPTE to get the number of trials
        data = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                cond, band))
        for epo_idx in range(data_temp.shape[0]):
            for ROI in ROIs:
                data_temp = data.copy()
                ROI_idx = label_names.index(ROI)
                reg_mask = np.zeros(cnx_params[stat_cond].shape)
                reg_mask[:,ROI_idx] = np.ones(reg_mask.shape[0])
                reg_mask[ROI_idx,:] = np.ones(reg_mask.shape[1])
                data_temp *= reg_mask
                c = cond if cond == "rest" else "task"
                dm = dm.append({"Subj":sub, "Block":c}, ignore_index=True)
                group_id.append(sub_idx)
group_id = np.array(group_id)
