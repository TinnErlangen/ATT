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

band = "alpha_0"
freqs = band_info[band]["freqs"]
subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = 4
n_jobs = 4
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc=parc,subjects_dir=subjects_dir)

region_names = ["L2235-lh", "L2235-rh", "L2340_L1933-lh", "L2340_L1933-rh"]
regions = []
for rn in region_names:
    for l in labels:
        if l.name == rn:
            regions.append(l)

conds = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]

fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))

df = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")
df["Angenehm"]=df_ang["Angenehm"]

data = []
predictor_vars = ["Laut","Angenehm","Subj","Block","Wav"]
dm = pd.DataFrame(columns=predictor_vars)

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
    if not re.search(".*{}-{}Hz_[0-9]*_ico4-lh.stc".format(freqs[0],freqs[-1]),filename):
        continue
    trial_info = re.match("nc_(ATT_[0-9]+)_(audio|visual|visselten)_(4000Hz|7000Hz|4000cheby|4000fftf)", filename).groups()
    sel_inds = ((df["Subj"]==trial_info[0]) & (df["Block"]==trial_info[1]) &
                (df["Wav"]==trial_info[2]))
    dm = dm.append(df[sel_inds])
    group_id.append(trial_info[0])
    stc = mne.read_source_estimate("{}/stcs/{}".format(proc_dir,filename))
    stc = morphs[trial_info[0]].apply(stc)
    for reg_idx,reg in enumerate(regions):
        temp_data = mne.extract_label_time_course(stc,reg,fs_src,mode="pca_flip")
        data[reg_idx].append(temp_data.mean())
data = np.array(data)

formula = "Brain ~ Laut + Angenehm + Block + Wav"
mod_fits = []
for reg_idx in range(len(data)):
    dm_temp = dm.copy()
    dm_temp["Brain"] = data[reg_idx,]
    model = MixedLM.from_formula(formula, dm_temp, groups=group_id)
    mod_fit = model.fit()
    mod_fits.append(mod_fit)

for reg_idx in range(2,4):
    plt.figure()
    intercept = mod_fits[reg_idx].params.get("Intercept")
    intercept_ci = [mod_fits[reg_idx].conf_int()[0]["Intercept"], mod_fits[reg_idx].conf_int()[1]["Intercept"]]
    visselt = mod_fits[reg_idx].params.get("Block[T.visselten]")
    visselt_ci = [mod_fits[reg_idx].conf_int()[0]["Block[T.visselten]"], mod_fits[reg_idx].conf_int()[1]["Block[T.visselten]"]]
    vis = mod_fits[reg_idx].params.get("Block[T.visual]")
    vis_ci = [mod_fits[reg_idx].conf_int()[0]["Block[T.visual]"], mod_fits[reg_idx].conf_int()[1]["Block[T.visual]"]]
    bars = np.array([0,vis,visselt])
    plt.bar(conds,bars)
    plt.vlines(1,ymin=vis_ci[0],ymax=vis_ci[1],color="black")
    plt.vlines(2,ymin=visselt_ci[0],ymax=visselt_ci[1], color="black")
