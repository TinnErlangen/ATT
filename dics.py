import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet
import numpy as np

proc_dir = "../proc/"
subjs = ["ATT_11","ATT_18","ATT_19","ATT_20","ATT_21","ATT_36"]
runs = ["rest","audio","visselten","visual","zaehlen"]
subjects_dir = "/home/jeff/freesurfer/subjects/"

fmin=8
fmax=14
frequencies = np.linspace(fmin, fmax, fmax-fmin+1)

for sub in subjs:
    l_sens = mne.read_label("{dir}nc_{sub}-lh.label".format(dir=proc_dir, sub=sub))
    r_sens = mne.read_label("{dir}nc_{sub}-rh.label".format(dir=proc_dir, sub=sub))
    src = mne.read_source_spaces(proc_dir+sub+"-src.fif")
    for run in runs:
        epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(dir=proc_dir, sub=sub, run=run)
        fwd_name = "{dir}nc_{sub}_{run}-fwd.fif".format(dir=proc_dir, sub=sub, run=run)
        epo = mne.read_epochs(epo_name)
        csd = csd_morlet(epo, frequencies=frequencies, n_jobs=8, n_cycles=7)
        csd = csd.mean()
        fwd = mne.read_forward_solution(fwd_name)
        filters = make_dics(epo.info, fwd, csd, label=l_sens+r_sens)
        print("\n\n")
        print(len(epo))
        print("\n\n")
        for event in range(len(epo)):
            event_csd = csd_morlet(epo[event], frequencies=frequencies,
            n_jobs=8, n_cycles=7)
            event_csd = event_csd.mean()
            stc = apply_dics_csd(event_csd,filters)
            stc[0].expand([s["vertno"] for s in src])
            stc[0].subject = sub
            stc[0].save("{a}stcs/nc_{b}_{c}_{d}".format(
                            a=proc_dir, b=sub, c=run, d=event))
