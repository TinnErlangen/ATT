import mne
import numpy as np

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
subjs = ['ATT_10']
runs = [str(x+1) for x in range(5)]
runs = ["1"]
base_dir ="../"
proc_dir = base_dir+"proc/"

epolen = 2

for sub in subjs:
    epos = []
    for run_idx,run in enumerate(runs):
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
        dir=proc_dir,sub=sub,run=run))
        events = np.load("{dir}nc_{sub}_{run}_events.npy".format(
        dir=proc_dir,sub=sub,run=run))
        #transform trigger values back to normal
        for x in np.nditer(events[:,1:3], op_flags=["readwrite"]):
            if x > 4000:
                x -= 4095
        for x in list(events):
            if x[2] == 100:
                raw.annotations.append(raw.times[x[0]],1.5,"bad aud")
            if x[2] == 200:
                raw.annotations.append(raw.times[x[0]],1.5,"bad vis")

        # identify which block it was
        block_ids = {255:"rest",10:"audio",20:"visselten",30:"visual",40:"zaehlen"}
        try:
            cond = block_ids[events[0,2]]
        except ValueError:
            print("First trigger doesn't correspond to known block code.")
        # identify where which sounds start
        block_id = events[0,2]
        cond_ids = [0]
        cond_idx = []
        if len(events)>1:
            cond_ids = [events[1,2]]
            events[1,1] = 0
            cond_event_idx = list(np.where(events[2:,1]==0)[0]+2)
            cond_ids += list(events[cond_event_idx,2])
            cond_idx = list(events[cond_event_idx,0])
        cond_idx.append(np.Inf)

        # get areas where there were artefacts, or no stimuli
        # label X second (var epolen) stretches not marked bad
        ss = mne.annotations._annotations_starts_stops(raw,["bad blink","bad vis","bad aud",
        "bad nostim"])
        reg_events = []
        focus = 0
        annot_idx = 0
        samp_step = int(epolen*raw.info["sfreq"])
        cond_step = 0
        while focus < len(raw):
            if focus > cond_idx[cond_step]:
                cond_step += 1
            if annot_idx < len(ss[1]) and focus+samp_step > ss[0][annot_idx]:
                focus = ss[1][annot_idx]
                annot_idx += 1
                continue
            reg_events.append([focus,0,block_id+cond_ids[cond_step]])
            raw.annotations.append(raw.times[focus],epolen,"good")
            focus += samp_step

        reg_events = np.array(reg_events)
        epos.append(mne.Epochs(raw,reg_events,baseline=None,tmin=0,tmax=epolen,
        reject_by_annotation=False))
        epos[-1].load_data()

    # find epo with lowest number of epochs
    low_epo_idx = np.argmin([len(x) for x in epos[1:]])+1
    eves = epos[low_epo_idx].events[:,2]%10
    counts = [np.sum(eves==x) for x in range(1,5)]
    # make new shuffled epoch objects with same number as the one with lowest
    new_epos = [epos[0][np.random.permutation(np.arange(
    len(epos[0])))[:np.sum(counts)]]] # resting state first
    for e in epos[1:]: # exclude resting state
        temp_epos = []
        for c in range(4):
            idxs = np.where(e.events[:,2]%10==c+1)[0]
            idxs_perm = np.random.permutation(idxs)[:counts[c]]
            temp_epos.append(e[idxs_perm])
        new_epos.append(mne.concatenate_epochs(temp_epos))
    for ne in new_epos:
        if ne.events[0,2] == 255:
            bl_id = 255
        else:
            bl_id = ne.events[0,2]-ne.events[0,2]%10
        cond = block_ids[bl_id]
        ne.save(proc_dir+"nc_"+sub+"_"+cond+"-epo.fif")
