import mne
import numpy as np

# do an ICA decomposition on data, and reference channels.

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
subjs = ["ATT_20", "ATT_21", "ATT_22", "ATT_23",
"ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
runs = [str(x+1) for x in range(5)]
#runs = ["3"]

base_dir ="../"
proc_dir = base_dir+"proc/"

for sub in subjs:
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand-raw.fif".format(
        dir=proc_dir,sub=sub,run=run),preload=True)
        icaref = mne.preprocessing.ICA(n_components=None,max_iter=1000,
                                       method="picard",allow_ref_meg=True)
        picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
        icaref.fit(raw,picks=picks)
        icaref.save("{dir}nc_{sub}_{run}_hand_ref-ica.fif".format(dir=proc_dir,
                                                                  sub=sub,
                                                                  run=run))
        icameg = mne.preprocessing.ICA(n_components=None,max_iter=1000,
                                       method="picard")
        picks = mne.pick_types(raw.info,meg=True,ref_meg=False)
        icameg.fit(raw,picks=picks)
        icameg.save("{dir}nc_{sub}_{run}_hand_meg-ica.fif".format(dir=proc_dir,
                                                                  sub=sub,
                                                                  run=run))
