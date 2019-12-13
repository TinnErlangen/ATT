import mne
import numpy as np
from os import listdir
import pickle

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks
proc_dir = "../proc/"
runs = ["rest","audio","visselten","visual","zaehlen"]
filelist = listdir(proc_dir+"stcs/")

for sub in subjs:
    print("Subject: {a}".format(a=sub))
    for run in runs:
        X_temp = []
        epo_num = 0
        filename = "nc_{a}_{b}_{e}-lh.stc".format(a=sub,b=run,e=epo_num)
        while filename in filelist:
            stc = mne.read_source_estimate(proc_dir+"stcs/"+filename)
            X_temp.append(stc.data)
            epo_num += 1
            filename = "nc_{a}_{b}_{e}-lh.stc".format(a=sub,b=run,e=epo_num)
        X_temp = np.array(X_temp,dtype=np.float64)
        np.save("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=run), X_temp)
        X_mean = np.mean(X_temp,axis=0)
        X_std = np.std(X_temp,axis=0)
        X_t = (X_mean*np.sqrt(epo_num-1))/X_std
        stc.data = X_mean
        stc.subject = sub
        stc.save("{dir}stcs/nc_{a}_{b}_mean".format(dir=proc_dir,a=sub,b=run))
        stc.data = X_t
        stc.save("{dir}stcs/nc_{a}_{b}_t".format(dir=proc_dir,a=sub,b=run))
