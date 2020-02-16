import mne
from mayavi import mlab
import pickle
from scipy import stats
import numpy as np
from mne.stats import linear_regression
import pandas as pd
import matplotlib.pyplot as plt
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
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "oct6"
conds = ["audio","visselten","visual"]
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
df_laut = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")

fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",
                                                         spacing))
del fs_src
stcs = []
for sub_idx,sub in enumerate(subjs):
    src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub, spacing))
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subject_to="fsaverage",
                                     subjects_dir=subjects_dir,
                                     spacing=[s["vertno"] for s in src],
                                     smooth=20)
    idx = 0
    for cond_idx,cond in enumerate(conds):
        for wav_idx,wav in enumerate(wavs):
            stc_temp = mne.read_source_estimate(
                    "{dir}stcs/nc_{a}_{b}_{c}_{d}-lh.stc".format(dir=proc_dir,a=sub,
                                                              b=cond,c=wav,d=spacing))
            stcs.append(morph.apply(stc_temp))

df_laut["Intercept"] = 1
temp_df = []
for cond in conds:
    temp_df.append(df_laut[df_laut["Block"]==cond])
df_laut = pd.concat(temp_df)
predictor_vars = ["Laut"] + ["Intercept"]
design_matrix = df_laut.copy()[predictor_vars]
reg_laut = linear_regression(stcs,design_matrix=design_matrix,names=predictor_vars)

df_ang["Intercept"] = 1
temp_df = []
for cond in conds:
    temp_df.append(df_ang[df_ang["Block"]==cond])
df_ang = pd.concat(temp_df)
predictor_vars = ["Angenehm"] + ["Intercept"]
design_matrix = df_ang.copy()[predictor_vars]
reg_ang = linear_regression(stcs,design_matrix=design_matrix,names=predictor_vars)
