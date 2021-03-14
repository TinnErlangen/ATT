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
def mass_uv_mixedlmm(formula, data, uv_data, group_id, re_formula=None,
                     vc_formula=None):
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
root_dir = "/home/jev/ATT_dat/"
#root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir + "proc/"
out_dir = root_dir + "lmm/"
spacing = "ico4"
conds = ["audio","visual","visselten","zaehlen"]
z_name = {}
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
band = opt.band
no_Z = True
if no_Z:
    conds = ["audio","visual","visselten"]
    z_name = "no_Z"

'''
build up the dataframes and group_id which will eventually be passed to
mass_uv_lmm. We will build two models here. "Simple" will make only one contrast:
resting state and task. "Cond" makes distinctions for the different conditions.
'''
data = []
predictor_vars = ("Subj", "Task", "Block", "RT")
df = pd.DataFrame(columns=predictor_vars)
group_id = []
for sub_idx,sub in enumerate(subjs):
    for cond_idx,cond in enumerate(conds):
        dPTE = load_sparse("{}nc_{}_{}_byresp_dPTE_{}.sps".format(proc_dir, sub,
                                                                  cond, band))
        epo = mne.read_epochs("{}{}_{}_byresp-epo.fif".format(proc_dir, sub,
                                                                     cond))
        for epo_idx in range(len(epo)):
            c = cond if cond == "rest" else "task"
            df = df.append({"Subj":sub, "Task":c, "Block":cond,
                            "RT":epo.metadata.iloc[epo_idx]["RT"]},
                             ignore_index=True)
            data.append(phi(dPTE[epo_idx,], k=1)) # flatten upper diagonal of connectivity matrix, add it to the list
            group_id.append(sub_idx)
data = np.array(data)
group_id = np.array(group_id)


'''
Finally, pass the variables along to mass_uv_lmm, for null, simple, and cond.
The function returns a list of fitted LMM; each list member is a model fitted
at a point of observation (e.g. vertex, voxel). Then save each member of the list.
'''


formula = "Brain ~ 1"
re_formula = None
mods_null = mass_uv_mixedlmm(formula, df, data, group_id,
                             re_formula=re_formula)
for mod_idx,mod in enumerate(mods_null):
    if mod == None:
        continue
    mod.save("{}{}/null_reg70_byresp_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                             z_name))

formula = "Brain ~ RT + Block"
re_formula = "1 + RT"
mods_simple = mass_uv_mixedlmm(formula, df, data, group_id,
                             re_formula=re_formula)
for mod_idx,mod in enumerate(mods_simple):
    if mod == None:
        continue
    mod.save("{}{}/simple_reg70_byresp_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                               z_name))

formula = "Brain ~ RT*C(Block, Treatment('audio'))"
re_formula = "1 + RT"
mods_cond = mass_uv_mixedlmm(formula, df, data, group_id,
                             re_formula=re_formula)
for mod_idx,mod in enumerate(mods_cond):
    if mod == None:
        continue
    mod.save("{}{}/cond_reg70_byresp_lmm_{}{}.pickle".format(out_dir,opt.band,mod_idx,
                                                             z_name))
