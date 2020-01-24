import mne
from mne.beamformer import make_dics,apply_dics_csd
import pickle
from mne.time_frequency import csd_morlet
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

proc_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks
subjs = ["ATT_10"]
runs = ["rest","audio","visselten","visual","zaehlen"]
runs = ["rest"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
spacing="oct4"
frequencies = [list(np.linspace(7,14,8)) for x in range(5)]
cov = mne.read_cov("{dir}empty-cov.fif".format(dir=proc_dir))

for sub in subjs:
    l_sens = mne.read_label("{dir}nc_{sub}_{sp}-lh.label".format(dir=proc_dir, sub=sub, sp=spacing))
    r_sens = mne.read_label("{dir}nc_{sub}_{sp}-rh.label".format(dir=proc_dir, sub=sub, sp=spacing))
    src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub,spacing))
    for run_idx,run in enumerate(runs):
        fwd_name = "{dir}nc_{sub}_{run}_{sp}-fwd.fif".format(dir=proc_dir, sub=sub, run=run, sp=spacing)
        fwd = mne.read_forward_solution(fwd_name)
        epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(dir=proc_dir, sub=sub, run=run)
        epo = mne.read_epochs(epo_name)
        inv_op = make_inverse_operator(epo.info, fwd, cov)
        stcs = apply_inverse_epochs(epo, inv_op, 1, method="sLORETA")
