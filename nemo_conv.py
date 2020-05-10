import mne
import numpy as np
# Convert from BTI format to MNE-Python

base_dir ="/home/jeff/ATT_dat/raw/"
raw_dir = base_dir+"raw/"
raw_dir = base_dir
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None
notches = [50, 62, 100, 150, 200]
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])

sub = "NEM_11"
run = 4

workfile = "{dir}nc_{s}/{r}/c,rfhp1.0Hz".format(dir=raw_dir,s=sub,r=run)
print(workfile)
raw = mne.io.read_raw_bti(workfile,preload=True,
                             rename_channels=False)
raw.notch_filter(notches,n_jobs="cuda", notch_widths=breadths)
raw.resample(200,n_jobs="cuda")
