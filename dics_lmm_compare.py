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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
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
subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
lmm_dir = "/home/jeff/ATT_dat/lmm_dics/"
spacing = 4
n_jobs = 4
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc=parc,subjects_dir=subjects_dir)

region_names = [lab.name for lab in labels]
regions = []
for rn in region_names:
    for l in labels:
        if l.name == rn:
            regions.append(l)

conds = ["rest", "audio", "visual", "visselten", "zaehlen"]
conds = [cond+"|" for cond in conds]
cond_str = "".join(conds)[:-1]

fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))

data = []
predictor_vars = ["Subj","Block"]
dm = pd.DataFrame(columns=predictor_vars)
dm_simple = pd.DataFrame(columns=predictor_vars)

morphs = {}
for sub in subjs:
    src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,sub,spacing))
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subject_to="fsaverage",
                                     spacing=spacing,
                                     subjects_dir=subjects_dir,
                                     smooth=None)
    morphs[sub] = morph

filelist = listdir(proc_dir+"/stcs/")

group_id = []
data = [[] for reg in regions]
for filename in filelist:
    rest_match = re.search("rest_{}-{}Hz_[0-9]*_ico4-lh.stc".format(freqs[0],freqs[-1]),filename)
    other_match = re.search("({})_.*_{}-{}Hz_[0-9]*_ico4-lh.stc".format(cond_str,freqs[0],freqs[-1]),filename)
    if not rest_match and not other_match:
        continue
    print(filename)
    trial_info = re.match("nc_(ATT_[0-9]+)_({})".format(cond_str), filename).groups()
    dm = dm.append({"Subj":trial_info[0],"Block":trial_info[1]},ignore_index=True)
    if trial_info[1] == "rest":
        dm_simple = dm_simple.append({"Subj":trial_info[0],"Block":trial_info[1]},ignore_index=True)
    else:
        dm_simple = dm_simple.append({"Subj":trial_info[0],"Block":"task"},ignore_index=True)
    group_id.append(trial_info[0])
    stc = mne.read_source_estimate("{}/stcs/{}".format(proc_dir,filename))
    stc = morphs[trial_info[0]].apply(stc)
    for reg_idx,reg in enumerate(regions):
        temp_data = mne.extract_label_time_course(stc,reg,fs_src,mode="mean")
        data[reg_idx].append(temp_data.mean())
data = np.array(data) * 1e+26

for reg_idx in range(len(data)):
    dm_temp = dm.copy()
    dm_temp["Brain"] = data[reg_idx,]
    dm_simple_temp = dm_simple.copy()
    dm_simple_temp["Brain"] = data[reg_idx,]

    formula = "Brain ~ 1"
    model = MixedLM.from_formula(formula, dm_temp, groups=group_id)
    mod_fit = model.fit(reml=False)
    mod_fit.save("{}{}/null_reg70_lmm_{}.pickle".format(lmm_dir,band,reg_idx))

    formula = "Brain ~ C(Block, Treatment('rest'))"
    model = MixedLM.from_formula(formula, dm_simple_temp, groups=group_id)
    mod_fit = model.fit(reml=False)
    mod_fit.save("{}{}/simple_reg70_lmm_{}.pickle".format(lmm_dir,band,reg_idx))

    formula = "Brain ~ C(Block, Treatment('rest'))"
    model = MixedLM.from_formula(formula, dm_temp, groups=group_id)
    mod_fit = model.fit(reml=False)
    mod_fit.save("{}{}/cond_reg70_lmm_{}.pickle".format(lmm_dir,band,reg_idx))
