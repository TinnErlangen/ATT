import mne

# Convert from BTI format to MNE-Python

base_dir ="/home/jeff/ATT_dat/"
base_dir = "/media/hdd/jeff/NEMO_dat/"
raw_dir = base_dir+"raw/"
raw_dir = base_dir
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["NEM_11"]
runs = [str(x+1) for x in range(5)]
runs = [str(x+1) for x in range(4)]

for sub in subjs:
    for run_idx,run in enumerate(runs):
        workfile = "{dir}nc_{s}/{r}/c,rfhp1.0Hz".format(dir=raw_dir,s=sub,r=run)
        print(workfile)
        rawmeg = mne.io.read_raw_bti(workfile,preload=True,
                                     rename_channels=False)
        # rawmeg.save("{dir}nc_{s}_{r}-raw.fif".format(dir=proc_dir,s=sub,
        #                                              r=run),
        #                                              overwrite=True)
