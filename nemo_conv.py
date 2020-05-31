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
raw.drop_channels(["TRIGGER","RESPONSE","UACurrent"])
raw.notch_filter(notches,n_jobs="cuda", notch_widths=breadths)
raw.resample(100,n_jobs="cuda")

drop_every = 1
picks = mne.pick_types(raw.info,meg=True)
drop_list = []
drop_idx = 0
for ch_idx in np.nditer(picks):
    if drop_idx == drop_every:
        drop_list.append(raw.ch_names[ch_idx])
        drop_idx = 0
    else:
        drop_idx += 1
raw.drop_channels(drop_list)

all_picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
ref_picks = mne.pick_types(raw.info,meg=False,ref_meg=True)
ica = mne.preprocessing.ICA(n_components=32,allow_ref_meg=True,method="infomax")
icaref = mne.preprocessing.ICA(n_components=2,allow_ref_meg=True,method="infomax")
