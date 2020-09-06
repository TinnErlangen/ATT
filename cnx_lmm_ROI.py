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

ROIs = {"left M1 superior":"L3395-lh", "left M1 central":"L3969-lh",
        "left M1 dorsal":"L8143_L7523-lh","left S1 superior0":"L7491_L4557-lh",
        "left S1 superior1":"L8143-lh","left sup-parietal":"L4557-lh"}
inreg_groups = {"left M1 superior":"M1", "left M1 central":"M1",
                "left M1 dorsal":"M1","left S1 superior0":"sup-parietal",
                "left S1 superior1":"sup-parietal",
                "left sup-parietal":"sup-parietal"}

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
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
triu_inds = np.triu_indices(mat_n, k=1)
inreg_consolidate = True

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
            for k,v in ROIs.items():
                ROI_idx = label_names.index(v)
                cnx_col_inds = list(np.where(cnx_masks[ROI_idx,])[0])
                for col_idx in cnx_col_inds:
                    this_point = this_epo[ROI_idx,col_idx].copy()
                    outname = label_names[col_idx]
                    outhemi = "lh" if "lh" in outname else "rh"
                    data_dict["Brain"].append(this_point)
                    data_dict["Subj"].append(sub)
                    data_dict["Block"].append(cond)
                    if inreg_consolidate:
                        this_reg = inreg_groups[k]
                    else:
                        this_reg = k
                    data_dict["InRegion"].append(this_reg)
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

if inreg_consolidate:
    formula = "Brain ~ C(Block, Treatment('rest')) + InRegion*C(Block, Treatment('rest'))"
else:
    formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('left M1 central'))*C(Block, Treatment('rest'))"
mod_inreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
mf_inreg = mod_inreg.fit(reml=False)

if inreg_consolidate:
    formula = "Brain ~ C(Block, Treatment('rest'))*InRegion + C(Block, Treatment('rest'))*OutRegion"
else:
    formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('left M1 central'))*C(Block, Treatment('rest')) + OutRegion*C(Block, Treatment('rest'))"
mod_outreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
mf_outreg = mod_outreg.fit(reml=False)

group_id_m1 = this_group_id[this_dm["InRegion"]=="M1"]
dm_m1 = this_dm[this_dm["InRegion"]=="M1"]
out_regions = list(set(dm_m1["OutRegion"].values))

formula = "Brain ~ C(Block, Treatment('rest'))*OutRegion"
mod_m1 = MixedLM.from_formula(formula, dm_m1, groups=group_id_m1)
mf_m1 = mod_m1.fit(reml=False)
cnx_m1 = {x:[] for x in ["OutRegion","Block","est_dPTE","t"]}
for cond in conds[1:]:
    for o_r in out_regions:
        cnx_m1["OutRegion"].append(o_r)
        cnx_m1["Block"].append(cond)
        predictors = {"Block":cond,"OutRegion":o_r}
        cnx_m1["est_dPTE"].append(mf_m1.predict(predictors)[0])
        if o_r == dm_m1.iloc[0]["OutRegion"]:
            cnx_m1["t"].append(mf_m1.tvalues["C(Block, Treatment('rest'))[T.{}]".format(cond)])
        else:
            cnx_m1["t"].append(mf_m1.tvalues["C(Block, Treatment('rest'))[T.{}]:OutRegion[T.{}]".format(cond,o_r)])
cnx_m1 = pd.DataFrame.from_dict(cnx_m1)
cnx_m1.to_pickle("{}cnx_m1.pickle".format(proc_dir))

group_id_sp = this_group_id[this_dm["InRegion"]=="sup-parietal"]
dm_sp = this_dm[this_dm["InRegion"]=="sup-parietal"]
out_regions = list(set(dm_sp["OutRegion"].values))

mod_sp = MixedLM.from_formula(formula, dm_sp, groups=group_id_sp)
mf_sp = mod_sp.fit(reml=False)
cnx_sp = {x:[] for x in ["OutRegion","Block","est_dPTE","t"]}
for cond in conds[1:]:
    for o_r in out_regions:
        cnx_sp["OutRegion"].append(o_r)
        cnx_sp["Block"].append(cond)
        predictors = {"Block":cond,"OutRegion":o_r}
        cnx_sp["est_dPTE"].append(mf_sp.predict(predictors)[0])
        if o_r == dm_sp.iloc[0]["OutRegion"]:
            cnx_sp["t"].append(mf_sp.tvalues["C(Block, Treatment('rest'))[T.{}]".format(cond)])
        else:
            cnx_sp["t"].append(mf_sp.tvalues["C(Block, Treatment('rest'))[T.{}]:OutRegion[T.{}]".format(cond,o_r)])
cnx_sp = pd.DataFrame.from_dict(cnx_sp)
cnx_sp.to_pickle("{}cnx_sp.pickle".format(proc_dir))
