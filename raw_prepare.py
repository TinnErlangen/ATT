import numpy as np
import mne

# Get events, mark sections where no stimulus occurs (specific to ATT),
# filter, downsample, save.

base_dir ="/home/jeff/ATT_dat/"
proc_dir = base_dir+"proc/"

l_freq=None
h_freq=None
notches = [16.7, 24, 50, 62, 100, 150, 200]
notches = [50, 62, 100, 150, 200]
breadths = np.array([0.25, 2.0, 1.5, 0.5, 0.5, 0.5, 0.5])
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
#subjs = ["ATT_10"]
runs = [str(x+1) for x in range(5)]
#runs = ["2"]

for sub in subjs:
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw("{dir}nc_{sub}_{run}-raw.fif".format(dir=proc_dir,
                                                              sub=sub,run=run),
                                                              preload=True)
        raw_events = mne.find_events(raw,stim_channel="TRIGGER",
        consecutive=True, shortest_event=1)
        raw_resps = mne.find_events(raw,stim_channel="RESPONSE",
        consecutive=True, shortest_event=1)

        # events and responses need to be temporarily fused together
        resp_start = len(raw_events) # remember where one starts, so we can pull them apart later
        eventsresps = np.concatenate((raw_events,raw_resps))

        # mark parts where no stimulus occurs as bad
        sti = raw.copy().pick_channels(["TRIGGER"]).get_data() # trigger channel data
        mark = False # keeps track of what state we're currently in: stim on or off
        raw.set_annotations(None)
        for i_idx,i in enumerate(np.nditer(sti)): # iterate through every time point in stim
            if not i and not mark: # stim switches to 0
                mark = True
                start = raw.times[i_idx]
                continue
            if i and mark: # stim switches to not-0
                mark = False
                finish = raw.times[i_idx]
                raw.annotations.append(start,finish-start,"bad nostim")
                continue
        if mark == True:
            finish = raw.times[i_idx]
            raw.annotations.append(start,finish-start,"bad nostim")

        picks = mne.pick_types(raw.info,meg=True,ref_meg=True) # get channels we want to filter
        raw.filter(l_freq,h_freq,picks=picks,n_jobs="cuda")
        raw.notch_filter(notches,n_jobs="cuda",picks=picks, notch_widths=breadths)
        raw,eventsresps = raw.resample(200,events=eventsresps,n_jobs="cuda")
        # now that downsampling is done, pull them back apart.
        meg_events,meg_resps = eventsresps[:resp_start,],eventsresps[resp_start:,]

        # save everything
        raw.save("{dir}nc_{sub}_{run}-raw.fif".format(
        dir=proc_dir,sub=sub,run=run), overwrite=True)
        np.save("{dir}nc_{sub}_{run}_events.npy".format(
        dir=proc_dir,sub=sub,run=run),meg_events)
        np.save("{dir}nc_{sub}_{run}_resps.npy".format(
        dir=proc_dir,sub=sub,run=run),meg_resps)
