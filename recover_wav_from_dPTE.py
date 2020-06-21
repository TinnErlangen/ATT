import mne
import numpy as np
from cnx_utils import TriuSparse, load_sparse


proc_dir = "../proc/"

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

runs = ["audio","visselten","visual","zaehlen"]
runs = ["audio","visselten","visual"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
cyc_names = ["theta_0","alpha_0","alpha_1","beta_0","beta_1","gamma_0","gamma_1"]

for sub in subjs:
    for run in runs:
        epos = []
        for wav_idx, wav_name in enumerate(wavs):
            epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
              dir=proc_dir, sub=sub, run=run, wav=wav_name)
            temp_epo = mne.read_epochs(epo_name)
            epos.append(temp_epo)
        for cn in cyc_names:
            dPTE = load_sparse("{dir}nc_{sub}_{run}_dPTE_{cyc}.sps".format(
              dir=proc_dir, sub=sub, run=run, cyc=cn))
            current_idx = 0
            for wav,epo in zip(wavs, epos):
                this_dPTE = dPTE[current_idx:current_idx+len(epo),]
                current_idx += len(epo)
                this_dPTE = TriuSparse(this_dPTE)
                this_dPTE.save("{dir}nc_{sub}_{run}_{wav}_dPTE_{cyc}.sps".format(
                  dir=proc_dir, sub=sub, run=run, wav=wav, cyc=cn))
