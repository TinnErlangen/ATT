import mne
from mne.beamformer import make_dics,apply_dics_csd
import pickle
from mne.time_frequency import csd_morlet
import numpy as np

proc_dir = "../proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjs = ["ATT_37"]
runs = ["rest","audio","visselten","visual","zaehlen"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
frequencies = [list(np.linspace(7,14,5)) for x in range(5)]
with open("peak_freq_table","rb") as f:
    table = pickle.load(f)

for sub in subjs:

    # if sub not in table:
    #     print("No peak frequencies found for {}. Skipping...".format(sub))
    #     continue
    # freqs = table[sub]

    l_sens = mne.read_label("{dir}nc_{sub}-lh.label".format(dir=proc_dir, sub=sub))
    r_sens = mne.read_label("{dir}nc_{sub}-rh.label".format(dir=proc_dir, sub=sub))
    src = mne.read_source_spaces(proc_dir+sub+"-src.fif")
    for run_idx,run in enumerate(runs):
        # centre_freq = freqs[table["conditions"].index(run)]
        # print("Central Frequency: {}".format(centre_freq))
        freqs = frequencies[run_idx]
        epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(dir=proc_dir, sub=sub, run=run)
        fwd_name = "{dir}nc_{sub}_{run}-fwd.fif".format(dir=proc_dir, sub=sub, run=run)
        epo = mne.read_epochs(epo_name)

        csd = csd_morlet(epo, frequencies=freqs, n_jobs=8, n_cycles=7, decim=3)
        #csd = csd.mean()
        fwd = mne.read_forward_solution(fwd_name)
        filters = make_dics(epo.info, fwd, csd, label=l_sens+r_sens)
        print("\n\n")
        print(len(epo))
        print("\n\n")
        for event in range(len(epo)):
            event_csd = csd_morlet(epo[event], frequencies=freqs,
                                   n_jobs=8, n_cycles=7, decim=3)
            #event_csd = event_csd.mean()
            stc, freqs = apply_dics_csd(event_csd,filters)
            stc.expand([s["vertno"] for s in src])
            stc.subject = sub
            stc.save("{a}stcs/nc_{b}_{c}_{d}".format(
                            a=proc_dir, b=sub, c=run, d=event))
