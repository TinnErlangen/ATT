import mne
from mne.beamformer import make_dics,apply_dics_csd
import pickle
from mne.time_frequency import csd_morlet
import numpy as np

proc_dir = "../proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks
#subjs = ["ATT_24"]
runs = ["audio","visselten","visual","zaehlen"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
n_jobs = 8
spacing = "oct6"
frequencies = [list(np.linspace(7,14,8)) for x in range(5)]
with open("peak_freq_table","rb") as f:
    table = pickle.load(f)

for sub in subjs:
    l_sens = mne.read_label("{dir}nc_{sub}_{sp}-lh.label".format(dir=proc_dir,
                                                                 sub=sub,
                                                                 sp=spacing))
    r_sens = mne.read_label("{dir}nc_{sub}_{sp}-rh.label".format(dir=proc_dir,
                                                                 sub=sub,
                                                                 sp=spacing))
    src = mne.read_source_spaces("{dir}{sub}_{sp}-src.fif".format(dir=proc_dir,
                                                                 sub=sub,
                                                                 sp=spacing))
    all_bads = []
    epos = []
    epo_names = []
    for run_idx,run in enumerate(runs):
        for wav_idx, wav_name in enumerate(wavs):
            freqs = frequencies[run_idx]
            epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
              dir=proc_dir, sub=sub, run=run, wav=wav_name)
            epo = mne.read_epochs(epo_name)
            all_bads += epo.info["bads"]
            all_bads = epo.info["bads"].copy()
            epos.append(epo)
            epo_names.append("{}_{}".format(run,wav_name))
    for x in epos:
        x.info["bads"] = all_bads
        x.info["dev_head_t"] = epos[0].info["dev_head_t"]
    epo = mne.concatenate_epochs(epos)
    csd = csd_morlet(epo, frequencies=freqs, n_jobs=n_jobs, n_cycles=7, decim=3)
    fwd_name = "{dir}nc_{sub}_{sp}-fwd.fif".format(dir=proc_dir, sub=sub, sp=spacing)
    fwd = mne.read_forward_solution(fwd_name)
    filters = make_dics(epo.info, fwd, csd, label=l_sens+r_sens, real_filter=True)
    del epo

    print("\n\n")
    print("\n\n")
    for epo,epo_name in zip(epos,epo_names):
        epo_csd = csd_morlet(epo, frequencies=freqs,
                               n_jobs=n_jobs, n_cycles=7, decim=3)
        stc, freqs = apply_dics_csd(epo_csd,filters)
        stc.expand([s["vertno"] for s in src])
        stc.subject = sub
        stc.save("{a}stcs/nc_{b}_{c}_{sp}".format(
                        a=proc_dir, b=sub, c=epo_name, sp=spacing))
        # for event in range(len(epo)):
        #     event_csd = csd_morlet(epo[event], frequencies=freqs,
        #                            n_jobs=n_jobs, n_cycles=7, decim=3)
        #     stc, freqs = apply_dics_csd(event_csd,filters)
        #     stc.expand([s["vertno"] for s in src])
        #     stc.subject = sub
        #     stc.save("{a}stcs/nc_{b}_{c}_{d}".format(
        #                     a=proc_dir, b=sub, c=runs[epo_idx], d=event))
