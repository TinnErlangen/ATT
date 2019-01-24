import mne
import numpy as np

subjs = ["ATT_10"]
runs = ["rest","audio","visselten","visual","zaehlen"]
runs = ["rest"]


base_dir ="../"
proc_dir = base_dir+"proc/"

for sub in subjs:
    for run_idx,run in enumerate(runs):
        epo = mne.read_epochs("{dir}nc_{sub}_{run}-epo.fif".format(
        dir=proc_dir,sub=sub,run=run))
        ica = mne.preprocessing.ICA(n_components=0.95,max_iter=500,
        method="extended-infomax")
        ica.fit(epo)
        ica.save("{dir}nc_{sub}_{run}-ica.fif".format(
        dir=proc_dir,sub=sub,run=run))
