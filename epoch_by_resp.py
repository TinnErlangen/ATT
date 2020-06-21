import mne
import numpy as np
import pandas as pd

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["ATT_18"]
runs = [str(x+1) for x in range(5)]
runs = ["5"]

proc_dir = "/home/jeff/ATT_dat/proc/"
run_names = ["audio","visselten","visual","zaehlen"]
wav_names = ["4000fftf","4000Hz","7000Hz","4000cheby"]
epolen = 2
for sub in subjs:
    for run_idx,run in enumerate(runs):
        subs, blocks, rts = [], [], []
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_p_hand_ica-raw.fif".format(
          dir=proc_dir,sub=sub,run=run))
        events = np.load("{dir}nc_{sub}_{run}_events.npy".format(
          dir=proc_dir,sub=sub,run=run))
        resps = np.load("{dir}nc_{sub}_{run}_resps.npy".format(
          dir=proc_dir,sub=sub,run=run))
        sfreq = raw.info["sfreq"]
        #transform trigger values back to normal
        for x in np.nditer(events[:,1:3], op_flags=["readwrite"]):
            if x > 4000:
                x -= 4095

        # identify which block it was
        block_ids = {255:"rest",10:"audio",20:"visselten",30:"visual",40:"zaehlen"}
        try:
            cond = block_ids[events[0,2]]
        except ValueError:
            print("First trigger doesn't correspond to known block code.")
        if (cond == "rest") | (cond=="zaehlen"):
            continue

        # get areas where there were artefacts, or no stimuli
        # label X second (var epolen) stretches not marked bad
        ss = mne.annotations._annotations_starts_stops(raw,["bad blink","bad vis","bad aud",
          "bad nostim"])

        block_trig = 100 if cond=="audio" else 200
        temp_events = events[events[:,2]==block_trig]
        new_events = []
        for row_idx in range(len(temp_events)):
            samp_start = temp_events[row_idx,0]
            samp_dist = int(samp_start + 1.5*sfreq)
            if any((resps[:,0]>samp_start) & (resps[:,0]<samp_dist)):
                new_events.append(temp_events[row_idx,])
                subs.append(sub)
                blocks.append(cond)
                resp_samp = resps[(resps[:,0]>samp_start) & (resps[:,0]<samp_dist),0]
                rts.append(((resp_samp - samp_start)/sfreq)[0])
        new_events = np.array(new_events)
        df = pd.DataFrame(columns=["Subj","Block","RT"])
        df["Subj"] = subs
        df["Block"] = blocks
        df["RT"] = rts
        # epo = mne.Epochs(raw,new_events,tmin=-3,tmax=0,baseline=None,metadata=df)
        # epo.save("{}{}_{}_byresp-epo.fif".format(proc_dir,sub,cond),overwrite=True)
