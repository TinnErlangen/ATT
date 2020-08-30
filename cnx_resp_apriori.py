from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
from cnx_utils import load_sparse, plot_directed_cnx, phi
import numpy as np
import mne
import pickle


root_dir = "/home/jeff/ATT_dat/"
proc_dir = root_dir + "proc/"
out_dir = root_dir + "lmm/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
band = "alpha_1"
conds = ["audio", "visual", "visselten"]
ROIs = ["L3969-lh","L3395-lh","L8143_L7523-lh","L7491_L4557-lh"]
#ROIs = ["L3395-lh"]
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = 70
node_n = 2415
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

group_id = []
columns = ("Brain","Subj","Block","ROI","OutRegion","Hemi","RT")
data_dict = {col:[] for col in columns}
for sub_idx,sub in enumerate(subjs):
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        dPTE = load_sparse("{}nc_{}_{}_byresp_dPTE_{}.sps".format(proc_dir, sub,
                                                                  cond, band))
        epo = mne.read_epochs("{}{}_{}_byresp-epo.fif".format(proc_dir, sub,
                                                                     cond))
        for epo_idx in range(len(dPTE)):
            this_epo = dPTE[epo_idx,].copy()
            this_epo[triu_inds[1],triu_inds[0]] = 1 - this_epo[triu_inds[0],triu_inds[1]]
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
                    data_dict["ROI"].append(ROI)
                    data_dict["OutRegion"].append(outname)
                    data_dict["Hemi"].append(outhemi)
                    data_dict["RT"].append(epo.metadata["RT"].iloc[epo_idx])
                    group_id.append(sub_idx)
dm = pd.DataFrame.from_dict(data_dict)
group_id = np.array(group_id)

formula = "RT ~ Brain*Block + Brain*Block*C(ROI, Treatment('L3969-lh'))"
formula = "Brain ~ RT*Block"

mfs = []
for ROI in ROIs:
    this_dm = dm.copy()
    this_dm = this_dm[this_dm["ROI"]==ROI]
    this_group_id = group_id[(dm["ROI"]==ROI)]
    mod = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
    mfs.append(mod.fit(reml=False))

formula = "RT ~ Block"
mod_rt = MixedLM.from_formula(formula, dm, groups=group_id)
mf_rt = mod_rt.fit()
