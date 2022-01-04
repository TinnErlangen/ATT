import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from surfer import Brain
from os import listdir
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM
import re
plt.ion()
import warnings
warnings.filterwarnings("ignore")


mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}

# all subjs
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

band_info = {}
band_info["theta_0"] = {"freqs":list(np.arange(3,7)),"cycles":3}
band_info["alpha_0"] = {"freqs":list(np.arange(7,10)),"cycles":5}
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
band_info["beta_0"] = {"freqs":list(np.arange(13,22)),"cycles":9}
band_info["beta_1"] = {"freqs":list(np.arange(22,31)),"cycles":9}
band_info["gamma_0"] = {"freqs":list(np.arange(31,41)),"cycles":9}
band_info["gamma_1"] = {"freqs":list(np.arange(41,60)),"cycles":9}
band_info["gamma_2"] = {"freqs":list(np.arange(60,90)),"cycles":9}

band = "alpha_1"
freqs = band_info[band]["freqs"]
subjects_dir = "/home/jev/hdd/freesurfer/subjects/"
proc_dir = "../proc/"
lmm_dir = "/home/jev/ATT_dat/lmm_dics/"
spacing = 4
n_jobs = 4
perm_n = 1024
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc=parc,
                                    subjects_dir=subjects_dir)

label_names = [lab.name for lab in labels]
stats_label_names = ["{}_{}".format(l[:-3],l[-2:]) for l in label_names]

conds = ["audio", "visual", "visselten"]
cond_str = [cond+"|" for cond in conds]
cond_str = "".join(cond_str)[:-1]

fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))

data = []
predictor_vars = ["Subj","Block", "RT", "TrialIdx"] + stats_label_names
df = pd.DataFrame(columns=predictor_vars)

morphs = {}
for sub in subjs:
    src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,sub,spacing))
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subject_to="fsaverage",
                                     spacing=spacing,
                                     subjects_dir=subjects_dir,
                                     smooth=None)
    morphs[sub] = morph
    for cond in conds:
        epo = mne.read_epochs("{}{}_{}_byresp-epo.fif".format(proc_dir,sub,cond))
        this_df = epo.metadata.copy()
        this_df["TrialIdx"] = np.arange(len(epo))
        df = df.append(this_df, ignore_index=True)

filelist = listdir(proc_dir+"/stcs/")


for filename in filelist:
    match_str = "nc_ATT_(.*)_(.*)_byresp_{}-{}Hz_([0-9]*)_ico4-lh.stc".format(freqs[0],freqs[-1])
    match = re.search(match_str, filename)
    if not match:
        continue
    subj, block, trial = match.group(1), match.group(2), match.group(3)
    print(filename)
    stc = mne.read_source_estimate("{}/stcs/{}".format(proc_dir,filename))
    stc = morphs["ATT_"+subj].apply(stc)
    ev_str = "Subj=='ATT_{}' and Block=='{}' and TrialIdx=={}".format(subj, block, int(trial))
    row_idx = np.where(np.array(df.eval(ev_str)))[0]
    temp_data = mne.extract_label_time_course(stc,labels,fs_src,mode="mean")
    temp_data = temp_data.mean(axis=1)
    for lab_idx, ln in enumerate(stats_label_names):
        df.at[row_idx,ln] = temp_data[lab_idx]


df = df.astype({ln:np.float64 for ln in stats_label_names})
subjs = list(df["Subj"].unique())
perm_aics = np.zeros((len(stats_label_names), perm_n, 2))
col_idx = df.columns.get_loc("RT")
for perm_idx in range(perm_n):
    print("Perm {} of {}".format(perm_idx, perm_n))
    for subj in subjs:
        row_inds = np.where(df["Subj"]==subj)[0]
        temp_slice = df.iloc[row_inds, col_idx]
        temp_slice = temp_slice.sample(frac=1)
        df.iloc[row_inds, col_idx] = temp_slice
    for ln_idx, ln in enumerate(stats_label_names):
        formula = "{} ~ RT + Block".format(ln)
        re_formula = "1 + RT"
        model = MixedLM.from_formula(formula, df, groups=df["Subj"],
                                     re_formula=re_formula)
        mod_fit = model.fit(reml=False)
        perm_aics[ln_idx, perm_idx, 0] = mod_fit.aic

        formula = "{} ~ RT*C(Block, Treatment('audio'))".format(ln)
        re_formula = "1 + RT"
        model = MixedLM.from_formula(formula, df, groups=df["Subj"],
                                     re_formula=re_formula)
        mod_fit = model.fit(reml=False)
        perm_aics[ln_idx, perm_idx, 1] = mod_fit.aic

np.save("{}dics_byresp_perm_aics.npy".format(proc_dir), perm_aics)
