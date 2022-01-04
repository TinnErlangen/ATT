import numpy as np
import mne
from cnx_utils import load_sparse, phi
import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''
this function fits a model for each possible point of observation (e.g. vertex, voxel)
formula is the formula parameter which gets passed on to statsmodels MixedLM,
data is a pandas dataframe which contains the independent variables
uv_data is a 2d numpy array where the first dimension is the number of observations
 and the 2nd is the number of points of observations - the length of the 1st dimension
 must match the number of rows in the data variable
group_id is a 1d numpy array with length number of observations
'''
def mass_uv_mixedlmm(formula, data, uv_data, group_id):
    mods = []
    for d_idx in range(uv_data.shape[1]):
        print("{} of {}".format(d_idx, uv_data.shape[1]), end="\r",
                                flush=True)
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

# get command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--band', type=str, required=True)
opt = parser.parse_args()

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
out_dir = root_dir + "lmm/"
conds = ["rest","audio","visual","visselten","zaehlen"]
z_name = {}
band = opt.band
no_Z = False
if no_Z:
    conds = ["rest","audio","visual","visselten"]
    z_name = "no_Z"

'''
build up the dataframes and group_id which will eventually be passed to
mass_uv_lmm. We will build two models here. "Simple" will make only one contrast:
resting state and task. "Cond" makes distinctions for the different conditions.
'''
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
            data.append(phi(data_temp[epo_idx,], k=1)) # flatten upper diagonal of connectivity matrix, add it to the list
            group_id.append(sub_idx)
data = np.array(data)
group_id = np.array(group_id)

'''
Finally, pass the variables along to mass_uv_lmm, for null, simple, and cond.
The function returns a list of fitted LMM; each list member is a model fitted
at a point of observation (e.g. vertex, voxel). Then save each member of the list.
'''

formula = "Brain ~ 1"
mods_null = mass_uv_mixedlmm(formula, dm_simple, data, group_id)
for mod_idx,mod in enumerate(mods_null):
    if mod == None:
        continue
    mod.save("{}{}/null_reg70_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                      z_name))

formula = "Brain ~ C(Block, Treatment('rest'))"
mods_simple = mass_uv_mixedlmm(formula, dm_simple, data, group_id)
for mod_idx,mod in enumerate(mods_simple):
    if mod == None:
        continue
    mod.save("{}{}/simple_reg70_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                        z_name))

formula = "Brain ~ C(Block, Treatment('rest'))"
mods_cond = mass_uv_mixedlmm(formula, dm_cond, data, group_id)
for mod_idx,mod in enumerate(mods_cond):
    if mod == None:
        continue
    mod.save("{}{}/cond_reg70_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                      z_name))
