import numpy as np
import mne
from cnx_utils import load_sparse
import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd

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
node_n = 2415
ROIs = ["L3969-lh","L3395-lh","L8143_L7523-lh","L7491_L4557-lh"]
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
triu_inds = np.triu_indices(mat_n, k=1)

with open("{}{}/aic.pickle".format(out_dir,band), "rb") as f:
    aic_comps = pickle.load(f)

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = np.zeros((mat_n,mat_n))
mod_idx = 2
for n_idx in range(node_n):
    if aic_comps["single_winner_ids"][n_idx] == mod_idx:
        cnx_masks[triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1
        cnx_masks[triu_inds[1][n_idx],triu_inds[0][n_idx]] = 1


columns = ("Brain","Subj","Block","InRegion","OutRegion","Hemi")
data_dict = {col:[] for col in columns}
group_id = []
for sub_idx,sub in enumerate(subjs):
    for cond_idx,cond in enumerate(conds):
        # we actually only need the dPTE to get the number of trials
        data = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                cond, band))
        for epo_idx in range(data.shape[0]):
            this_epo = data[epo_idx,].copy()
            this_epo[triu_inds[1],triu_inds[0]] = 1 - this_epo[triu_inds[0],triu_inds[1]]
            #print("Subject: {}, Condition: {}, Epoch: {}".format(sub,cond,epo_idx))
            for ROI in ROIs:
                ROI_idx = label_names.index(ROI)
                cnx_col_inds = list(np.where(cnx_masks[ROI_idx,])[0])
                for col_idx in cnx_col_inds:
                    this_point = this_epo[ROI_idx,col_idx].copy()
                    outname = label_names[col_idx]
                    outhemi = "lh" if "lh" in outname else "rh"
                    data_dict["Brain"].append(this_point)
                    data_dict["Subj"].append(sub)
                    data_dict["Block"].append(cond)
                    data_dict["InRegion"].append(ROI)
                    data_dict["OutRegion"].append(outname)
                    data_dict["Hemi"].append(outhemi)
                    group_id.append(sub_idx)
dm = pd.DataFrame.from_dict(data_dict)
dm_noZ = dm[dm["Block"]!="zaehlen"]
group_id = np.array(group_id)
group_id_noZ = group_id[dm["Block"]!="zaehlen"]

this_dm = dm_noZ
this_group_id = group_id_noZ
this_dm = dm
this_group_id = group_id

formula = "Brain ~ C(Block, Treatment('rest'))"
mod_simple = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
mf_simple = mod_simple.fit(reml=False)

formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('{}'))*C(Block, Treatment('rest'))".format(ROIs[0])
mod_inreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
mf_inreg = mod_inreg.fit(reml=False)

formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('{}'))*C(Block, Treatment('rest')) + Hemi*C(Block, Treatment('rest'))".format(ROIs[-1])
mod_hemi = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
mf_hemi = mod_hemi.fit(reml=False)

# formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('{}'))*C(Block, Treatment('rest')) + OutRegion*C(Block, Treatment('rest'))".format(ROIs[0])
# mod_outreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
# mf_outreg = mod_outreg.fit(reml=False)
