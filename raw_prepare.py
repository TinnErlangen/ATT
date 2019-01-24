import numpy as np
import mne

base_dir ="/media/hdd/jeff/ATT_dat/"
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None
notches = [24,50,62,100,150,200]
breadths = np.array([2,1.5,0.5,0.5,0.5,0.5])
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14","ATT_15","ATT_16","ATT_17", "ATT_18", "ATT_19",]
subjs = ["ATT_10"]
runs = [str(x+1) for x in range(5)]
#runs = ["3"]

for sub in subjs:
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw("{dir}nc_{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run),preload=True)

        raw_events = mne.find_events(raw,stim_channel="TRIGGER",
        consecutive=True, shortest_event=1)
        raw_resps = mne.find_events(raw,stim_channel="RESPONSE",
        consecutive=True, shortest_event=1)
        resp_start = len(raw_events)
        eventsresps = np.concatenate((raw_events,raw_resps))

        # mark parts where no stimulus occurs as bad
        sti = raw.copy().pick_channels(["TRIGGER"]).get_data()
        mark = False
        raw.set_annotations(None)
        for i_idx,i in enumerate(np.nditer(sti)):
            if not i and not mark:
                mark = True
                start = raw.times[i_idx]
                continue
            if i and mark:
                mark = False
                finish = raw.times[i_idx]
                raw.annotations.append(start,finish-start,"bad nostim")
                continue
        if mark == True:
            finish = raw.times[i_idx]
            raw.annotations.append(start,finish-start,"bad nostim")

        picks = mne.pick_types(raw.info,meg=True,ref_meg=True)
        raw.filter(l_freq,h_freq,picks=picks)
        raw.notch_filter(notches,n_jobs="cuda",picks=picks, notch_widths=breadths)
        raw,eventsresps = raw.resample(200,events=eventsresps,n_jobs="cuda")
        meg_events,meg_resps = eventsresps[:resp_start,],eventsresps[resp_start:,]

        raw.save("{dir}nc_{sub}_{run}-raw.fif".format(
        dir=proc_dir,sub=sub,run=run), overwrite=True)
        np.save("{dir}nc_{sub}_{run}_events.npy".format(
        dir=proc_dir,sub=sub,run=run),meg_events)
        np.save("{dir}nc_{sub}_{run}_resps.npy".format(
        dir=proc_dir,sub=sub,run=run),meg_resps)
