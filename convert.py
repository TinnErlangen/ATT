import mne

# Convert from BTI format to MNE-Python

base_dir ="/media/hdd/jeff/ATT_dat/"
raw_dir = base_dir+"raw/"
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28"]
#subjs = ["NEM_16"]
runs = [str(x+1) for x in range(5)]
#runs = ["3"]

for sub in subjs:
    for run_idx,run in enumerate(runs):
        workdir = raw_dir+"nc_"+sub+"/"+run+"/"
        rawmeg = mne.io.read_raw_bti(workdir+"c,rfhp1.0Hz",preload=True,
                                     rename_channels=False)
        rawmeg.save("{dir}nc_{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,
                                                         run=run),
                                                         overwrite=True)
