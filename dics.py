import mne
from mne.beamformer import make_dics,apply_dics_csd
import pickle
from mne.time_frequency import csd_morlet
import numpy as np

doTones = True
proc_dir = "../proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks



band_info = {}
# band_info["theta_0"] = {"freqs":list(np.arange(3,7)),"cycles":3}
# band_info["alpha_0"] = {"freqs":list(np.arange(7,10)),"cycles":5}
# band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
# band_info["beta_0"] = {"freqs":list(np.arange(13,22)),"cycles":9}
band_info["beta_1"] = {"freqs":list(np.arange(22,31)),"cycles":9}
# band_info["gamma_0"] = {"freqs":list(np.arange(31,41)),"cycles":9}
# band_info["gamma_1"] = {"freqs":list(np.arange(41,60)),"cycles":9}
# band_info["gamma_2"] = {"freqs":list(np.arange(60,90)),"cycles":9}

runs = ["rest","audio","visual","visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
n_jobs = 1
spacing = "ico4"
for k,v in band_info.items():
    f = v["freqs"]
    c = v["cycles"]
    frequencies = [f for x in range(5)]
    print(frequencies)
    for sub in subjs:
        src = mne.read_source_spaces("{dir}{sub}_{sp}-src.fif".format(dir=proc_dir,
                                                                     sub=sub,
                                                                     sp=spacing))
        all_bads = []
        epos = []
        epo_names = []
        epo_conds = []
        epo_cond_names = []
        for run_idx,run in enumerate(runs):
            if run == "rest":
                epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(
                  dir=proc_dir, sub=sub, run=run)
                epo = mne.read_epochs(epo_name)
                all_bads += epo.info["bads"]
                epo_conds.append(epo)
                epo_cond_names.append(run)
            else:
                wav_epos = []
                for wav_idx, wav_name in enumerate(wavs):
                    freqs = frequencies[run_idx]
                    epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
                      dir=proc_dir, sub=sub, run=run, wav=wav_name)
                    epo = mne.read_epochs(epo_name)
                    all_bads += epo.info["bads"]
                    epos.append(epo)
                    wav_epos.append(epo)
                    epo_names.append("{}_{}".format(run,wav_name))
                epo_conds.append(mne.concatenate_epochs(wav_epos))
                epo_cond_names.append(run)

        for x in epos:
            x.info["bads"] = all_bads
            x.info["dev_head_t"] = epos[0].info["dev_head_t"]
        epo = mne.concatenate_epochs(epos)
        csd = csd_morlet(epo, frequencies=freqs, n_jobs=n_jobs, n_cycles=c, decim=3)
        #csd = csd.mean()
        fwd_name = "{dir}nc_{sub}_{sp}-fwd.fif".format(dir=proc_dir, sub=sub, sp=spacing)
        fwd = mne.read_forward_solution(fwd_name)
        filters = make_dics(epo.info, fwd, csd, real_filter=True,
                            weight_norm="nai", reduce_rank=False,
                            pick_ori="max-power")
        del epo, csd, fwd

        print("\n\n")
        print("\n\n")
        if not doTones:
            epos = epo_conds
            epo_names = epo_cond_names
        for epo,epo_name in zip(epos,epo_names):
            epo_csd = csd_morlet(epo, frequencies=freqs,
                                   n_jobs=n_jobs, n_cycles=c, decim=3)
            #epo_csd = epo_csd.mean()
            stc, freqs = apply_dics_csd(epo_csd,filters)
            stc.expand([s["vertno"] for s in src])
            stc.subject = sub
            stc.save("{a}stcs/nc_{b}_{c}_{f0}-{f1}Hz_{sp}".format(
                            a=proc_dir, b=sub, c=epo_name, f0=f[0], f1=f[-1], sp=spacing))
            for event in range(len(epo)):
                print(event)
                event_csd = csd_morlet(epo[event], frequencies=freqs,
                                       n_jobs=n_jobs, n_cycles=7, decim=3)
                stc, freqs = apply_dics_csd(event_csd,filters)
                del event_csd
                stc.expand([s["vertno"] for s in src])
                stc.subject = sub
                stc.save("{a}stcs/nc_{b}_{c}_{f0}-{f1}Hz_{d}_{sp}".format(
                                a=proc_dir, b=sub, c=epo_name, f0=f[0], f1=f[-1], d=event, sp=spacing))
