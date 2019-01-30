import mne
from os import listdir

# takes annotations and bad channels produced by annot_cycler and applies them.
# Useful in case you reconverted, preprocessed the data, and don't want
# to do everything by hand again.

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
subjs = ["ATT_20", "ATT_21", "ATT_22", "ATT_23",
"ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
runs = [str(x+1) for x in range(5)]
#runs = ["3"]
filelist = listdir(proc_dir)

for sub in subjs:
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}-raw.fif".format(dir=proc_dir,
                                                              sub=sub,run=run))
        if "nc_{sub}_{run}-raw.fif".format(sub=sub,run=run) in filelist:
            annot = mne.read_annotations("{dir}nc_{sub}_{run}-annot.fif".format(dir=proc_dir,sub=sub,run=run))
            raw.set_annotations(annot)
        bad_chan_txt = "nc_{sub}_{run}-badchans.txt".format(sub=sub,run=run)
        if bad_chan_txt in filelist:
            with open(proc_dir+bad_chan_txt, "r") as f:
                bad_chans = f.readlines()
            bad_chans = [x.strip() for x in bad_chans]
            raw.info["bads"] = bad_chans
        #raw.info["bads"] += ["A51","A188","A71"] # bad data channels
        raw.info["bads"] += ["MRyA","MRyaA"]  # bad reference channels
        raw.save("{dir}nc_{sub}_{run}_hand-raw.fif".format(
        dir=proc_dir,sub=sub,run=run),overwrite=True)
