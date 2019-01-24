import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet
import numpy as np

proc_dir = "../proc/"
subjs = ["ATT_10"]
runs = ["rest","audio","visselten","visual","zaehlen"]
subjects_dir = "/home/jeff/freesurfer/subjects/"

fmin=7
fmax=14
frequencies = np.linspace(fmin, fmax, fmax-fmin+1)

for sub in subjs:
    for run in runs:
        epo_name = "{dir}nc_{sub}_{run}_ica-epo.fif".format(dir=proc_dir, sub=sub, run=run)
        fwd_name = "{dir}nc_{sub}_rest_ica-fwd.fif".format(dir=proc_dir, sub=sub, run=run)
        l_sens = mne.read_label("../proc/nc_"+sub+"_"+run+"_sens_label-lh.label")
        r_sens = mne.read_label("../proc/nc_"+sub+"_"+run+"_sens_label-rh.label")
        epo = mne.read_epochs(epo_name)
        csd = csd_morlet(epo, frequencies=frequencies, n_jobs=8, n_cycles=7)
        csd = csd.mean()
        fwd = mne.read_forward_solution(fwd_name)
        filters = make_dics(epo.info, fwd, csd, label=l_sens+r_sens)
        for event in range(len(epo)):
            event_csd = csd_morlet(epo[event], frequencies=frequencies,
            n_jobs=8, n_cycles=7)
            event_csd = event_csd.mean()
            stc = apply_dics_csd(event_csd,filters)
            stc[0].save("{a}stcs/nc_{b}_{c}_{d}".format(
                            a=proc_dir, b=sub, c=run, d=event))
