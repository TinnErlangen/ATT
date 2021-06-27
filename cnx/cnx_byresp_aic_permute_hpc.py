import numpy as np
import mne
import argparse
import pickle
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def load_sparse(filename,convert=True,full=False,nump_type="float32"):
    with open(filename,"rb") as f:
        result = pickle.load(f)
    if convert:
        full_mat = np.zeros(result["mat_sparse"].shape[:-1] + \
          (result["mat_res"],result["mat_res"])).astype(nump_type)
        full_mat[...,result["mat_inds"][0],result["mat_inds"][1]] = \
          result["mat_sparse"]
        result = full_mat
    return result

def phi(mat, k=0):
    if len(mat.shape)>2:
        triu_inds = np.triu_indices(mat.shape[1],k=k)
        return mat[...,triu_inds[0],triu_inds[1]]
    else:
        triu_inds = np.triu_indices(mat.shape[0],k=k)
        return mat[triu_inds[0],triu_inds[1]]

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
parser.add_argument('--perm', type=int, default=500)
parser.add_argument('--iter', type=int, default=0)
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
root_dir = "/scratch/jeffhanna/ATT_dat/"
proc_dir = root_dir + "proc/"
out_dir = root_dir + "lmm/"
spacing = "ico4"
node_n = 2415
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
idx_borders = []
idx_border = 0
for sub_idx,sub in enumerate(subjs):
    idx_borders.append([idx_border])
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
            idx_border += 1
    idx_borders[-1].append(idx_border)

data = np.array(data)
group_id = np.array(group_id)
indep_var = "RT"

if indep_var:
    col_idx = df.columns.get_loc(indep_var)
else:
    col_idx = 1

formula = "Brain ~ RT*C(Block, Treatment('audio'))"
re_formula = "1 + RT"
# permute
perm_n = opt.perm
aics = np.zeros((node_n, perm_n))
for i in range(perm_n):
    print("Permutation {} of {}".format(i, perm_n))
    dm_perm = df.copy()
    for idx_border in idx_borders:
        if indep_var:
            temp_slice = dm_perm[indep_var][idx_border[0]:idx_border[1]].copy()
        else:
            temp_slice = dm_perm["Block"][idx_border[0]:idx_border[1]].copy()
        temp_slice = temp_slice.sample(frac=1)
        dm_perm.iloc[idx_border[0]:idx_border[1],col_idx] = temp_slice.values
    perm_mods = mass_uv_mixedlmm(formula, dm_perm, data, group_id,
                                 re_formula=re_formula)
    for n_idx, pm in enumerate(perm_mods):
        aics[n_idx,i] = pm.aic


np.save("{}cnx_{}_{}_byresp_perm_{}_{}.npy".format(proc_dir, indep_var, band, perm_n, opt.iter),
        aics)
