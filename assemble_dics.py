import mne
import numpy as np
from os import listdir
import pickle

subjs = ["ATT_10"]
proc_dir = "../proc/stcs/"
runs = ["rest","audio","visselten","visual","zaehlen"]
filelist = listdir(proc_dir)

#runs = ["rest"]
for sub in subjs:
    print("Subject: {a}".format(a=sub))
    for run in runs:
        X_temp = []
        epo_num = 0
        filename = "nc_{a}_{b}_{e}-lh.stc".format(a=sub,b=run,e=epo_num)
        while filename in filelist:
            stc = mne.read_source_estimate(proc_dir+filename)
            X_temp.append(stc.data)
            epo_num += 1
            filename = "nc_{a}_{b}_{e}-lh.stc".format(a=sub,b=run,e=epo_num)
        X_temp = np.array(X_temp,dtype=np.float64)
        np.save("{dir}nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=run), X_temp)
        X_mean = np.mean(X_temp,axis=0)
        X_std = np.std(X_temp,axis=0)
        X_t = (X_mean*np.sqrt(epo_num))/X_std
        stc.data = X_mean
        stc.save("{dir}nc_{a}_{b}_mean".format(dir=proc_dir,a=sub,b=run))
        stc.data = X_t
        stc.save("{dir}nc_{a}_{b}_t".format(dir=proc_dir,a=sub,b=run))
