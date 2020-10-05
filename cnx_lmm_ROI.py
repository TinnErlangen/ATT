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
        "left S1 superior1":"L8143-lh","left sup-parietal":"L4557-lh",
        "right M1 superior":"L3395-rh", "right M1 central":"L3969-rh",
        "right M1 dorsal":"L8143_L7523-rh","right S1 superior0":"L7491_L4557-rh",
        "right S1 superior1":"L8143-rh","right sup-parietal":"L4557-rh"}
ROIs_inv = {v:k for k,v in ROIs.items()}
inreg_groups = {"left M1 superior":"left_M1", "left M1 central":"left_M1",
                "left M1 dorsal":"left_M1","left S1 superior0":"left_sup-parietal",
                "left S1 superior1":"left_sup-parietal",
                "left sup-parietal":"left_sup-parietal",
                "right M1 superior":"right_M1", "right M1 central":"right_M1",
                "right M1 dorsal":"right_M1","right S1 superior0":"right_sup-parietal",
                "right S1 superior1":"right_sup-parietal",
                "right sup-parietal":"right_sup-parietal"}

# parameters and setup
root_dir = "/home/jeff/ATT_dat/"
#root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir + "proc/"
out_dir = root_dir + "lmm/"
spacing = "ico4"

band = opt.band
mat_n = 70
node_n = 2415
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
triu_inds = np.triu_indices(mat_n, k=1)
inreg_consolidate = True
no_Z = False

with open("{}{}/aic.pickle".format(out_dir,band), "rb") as f:
    aic_comps = pickle.load(f)

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = np.zeros((mat_n,mat_n))
mod_idx = 2
for n_idx in range(node_n):
    if aic_comps["single_winner_ids"][n_idx] == mod_idx:
        cnx_masks[triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1
        cnx_masks[triu_inds[1][n_idx],triu_inds[0][n_idx]] = 1

if no_Z:
    conds = ["rest","audio","visual","visselten"]
else:
    conds = ["rest","audio","visual","visselten","zaehlen"]


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
                        if outname in list(ROIs_inv.keys()):
                            if inreg_groups[ROIs_inv[outname]] == this_reg:
                                print("hi {}".format(outname))
                                continue
                    else:
                        this_reg = k
                    data_dict["InRegion"].append(this_reg)
                    data_dict["OutRegion"].append(outname)
                    data_dict["Hemi"].append(outhemi)
                    group_id.append(sub_idx)
dm = pd.DataFrame.from_dict(data_dict)
group_id = np.array(group_id)

if no_Z:
    this_dm = dm[dm["Block"]!="zaehlen"]
    this_group_id = group_id[dm["Block"]!="zaehlen"]
else:
    this_dm = dm
    this_group_id = group_id

# formula = "Brain ~ C(Block, Treatment('rest'))"
# mod_simple = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
# mf_simple = mod_simple.fit(reml=False)
#
# if inreg_consolidate:
#     formula = "Brain ~ C(Block, Treatment('rest')) + InRegion*C(Block, Treatment('rest'))"
# else:
#     formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('left M1 central'))*C(Block, Treatment('rest'))"
# mod_inreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
# mf_inreg = mod_inreg.fit(reml=False)
#
# if inreg_consolidate:
#     formula = "Brain ~ C(Block, Treatment('rest'))*InRegion + C(Block, Treatment('rest'))*OutRegion"
# else:
#     formula = "Brain ~ C(Block, Treatment('rest')) + C(InRegion, Treatment('left M1 central'))*C(Block, Treatment('rest')) + OutRegion*C(Block, Treatment('rest'))"
# mod_outreg = MixedLM.from_formula(formula, this_dm, groups=this_group_id)
# mf_outreg = mod_outreg.fit(reml=False)


regs = ["left_M1", "right_M1", "left_sup-parietal", "right_sup-parietal"]
for reg in regs:
    group_id_reg = this_group_id[this_dm["InRegion"]==reg]
    dm_reg = this_dm[this_dm["InRegion"]==reg]
    out_regions = list(set(dm_reg["OutRegion"].values))
    formula = "Brain ~ C(Block, Treatment('rest'))*OutRegion"

    mod_reg = MixedLM.from_formula(formula, dm_reg, groups=group_id_reg)
    mf_reg = mod_reg.fit(reml=False)
    cnx_reg = {x:[] for x in ["OutRegion","Block","est_dPTE","coef","t","p"]}
    for cond in conds:
        for o_r in out_regions:
            cnx_reg["OutRegion"].append(o_r)
            cnx_reg["Block"].append(cond)
            predictors = {"Block":cond,"OutRegion":o_r}
            cnx_reg["est_dPTE"].append(mf_reg.predict(predictors)[0])
            if o_r == dm_reg.iloc[0]["OutRegion"]:
                if cond == "rest":
                    cnx_reg["coef"].append(mf_reg.params["Intercept"])
                    cnx_reg["t"].append(mf_reg.tvalues["Intercept"])
                    cnx_reg["p"].append(mf_reg.pvalues["Intercept"])
                else:
                    cnx_reg["coef"].append(mf_reg.params["C(Block, Treatment('rest'))[T.{}]".format(cond)])
                    cnx_reg["t"].append(mf_reg.tvalues["C(Block, Treatment('rest'))[T.{}]".format(cond)])
                    cnx_reg["p"].append(mf_reg.pvalues["C(Block, Treatment('rest'))[T.{}]".format(cond)])
            else:
                if cond == "rest":
                    cnx_reg["coef"].append(mf_reg.params["OutRegion[T.{}]".format(o_r)])
                    cnx_reg["t"].append(mf_reg.tvalues["OutRegion[T.{}]".format(o_r)])
                    cnx_reg["p"].append(mf_reg.pvalues["OutRegion[T.{}]".format(o_r)])
                else:
                    cnx_reg["coef"].append(mf_reg.params["C(Block, Treatment('rest'))[T.{}]:OutRegion[T.{}]".format(cond,o_r)])
                    cnx_reg["t"].append(mf_reg.tvalues["C(Block, Treatment('rest'))[T.{}]:OutRegion[T.{}]".format(cond,o_r)])
                    cnx_reg["p"].append(mf_reg.pvalues["C(Block, Treatment('rest'))[T.{}]:OutRegion[T.{}]".format(cond,o_r)])
    cnx_reg = pd.DataFrame.from_dict(cnx_reg)
    if no_Z:
        cnx_reg.to_pickle("{}{}/cnx_{}_no_Z.pickle".format(out_dir, band, reg))
    else:
        cnx_reg.to_pickle("{}{}/cnx_{}.pickle".format(out_dir, band, reg))
