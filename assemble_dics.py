import mne
import numpy as np
from os import listdir
import pickle

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
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
runs = ["rest","audio","visselten","visual","zaehlen"]
runs = ["audio","visselten","visual","zaehlen"]
wavs = ["4000Hz","4000cheby","7000Hz","4000fftf"]
filelist = listdir(proc_dir+"stcs/")
spacing = "oct6"

for sub in subjs:
    print("Subject: {a}".format(a=sub))
    src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub, spacing))
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subject_to="fsaverage",
                                     spacing=[s["vertno"] for s in src],
                                     subjects_dir=subjects_dir,
                                     smooth=20)
    for run in runs:
        for wav in wavs:
            X_temp = []
            epo_num = 0
            filename = "nc_{a}_{b}_{c}_{e}_{sp}-lh.stc".format(a=sub,b=run,
                                                          c=wav,e=epo_num,
                                                          sp=spacing)
            while filename in filelist:
                stc = mne.read_source_estimate(proc_dir+"stcs/"+filename)
                stc = morph.apply(stc)
                X_temp.append(stc.data.mean(axis=1,keepdims=True))
                epo_num += 1
                filename = "nc_{a}_{b}_{c}_{e}_{sp}-lh.stc".format(a=sub,b=run,
                                                              c=wav,e=epo_num,
                                                              sp=spacing)
            X_temp = np.array(X_temp,dtype=np.float64)*1e+27
            np.save("{dir}stcs/nc_{a}_{b}_{c}_{sp}_stc.npy".format(dir=proc_dir,
                                                               a=sub,b=run,
                                                               c=wav,
                                                               sp=spacing),
                                                               X_temp)
            # X_mean = np.mean(X_temp,axis=0)
            # X_std = np.std(X_temp,axis=0)
            # X_t = (X_mean*np.sqrt(epo_num-1))/X_std
            # stc.data = X_mean
            # stc.subject = sub
            # stc.save("{dir}stcs/nc_{a}_{b}_{sp}_mean".format(dir=proc_dir,a=sub,
            #                                                  b=run,sp=spacing))
            # stc.data = X_t
            # stc.save("{dir}stcs/nc_{a}_{b}_{sp}_t".format(dir=proc_dir,a=sub,
            #                                               b=run,sp=spacing))
