import mne
import numpy as np

subjs = ["NEM_01"]
runs = [str(x) for x in range(3,5)]

base_dir ="../"
proc_dir = base_dir+"proc/"

mini_epochs_num = 4
mini_epochs_len = 2

for sub in subjs:
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw(proc_dir+"nc_"+sub+"_"+run+"_hand_ica-raw.fif")
        events = list(np.load(proc_dir+"nc_"+sub+"_"+run+"_events.npy"))
        new_events = []
        for e in events:
            if e[2] >= 100:
                for me in range(mini_epochs_num):
                    new_events.append(np.array(
                    [e[0]+me*mini_epochs_len*raw.info["sfreq"], 0 ,e[2]+me]))
        new_events = np.array(new_events).astype(int)
        epo = mne.Epochs(raw,new_events,baseline=(None,None),tmin=0,tmax=mini_epochs_len)
        epo.save(proc_dir+"nc_"+sub+"_"+run+"-epo.fif")
