import mne
import numpy as np
import argparse
from mne.time_frequency import tfr_array_morlet
# from pyPTE.pyPTE import (get_delay, get_binsize, get_discretized_phase,
#                          compute_dPTE_rawPTE)
from dPTE import epo_dPTE
from joblib import Parallel, delayed
import pickle
from cnx_utils import TriuSparse, load_sparse

def do_PTE(data):
    data = data.T
    delay = get_delay(data)
    phase_inc = data + np.pi
    binsize = get_binsize(phase_inc)
    d_phase = get_discretized_phase(phase_inc, binsize)
    return compute_dPTE_rawPTE(d_phase, delay)

proc_dir = "../proc/"
subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]


# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks
# ATT_29 did not respond
mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}
runs = ["rest","audio","visselten","visual"]
runs = ["audio","visselten","visual"]
runs = ["zaehlen"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]

inv_method="sLORETA"
snr = 1.0
lambda2 = 1.0 / snr ** 2
n_jobs = 4
spacing="ico5"

band_info = {}
band_info["theta_0"] = {"freqs":list(np.arange(3,7)),"cycles":3}
band_info["alpha_0"] = {"freqs":list(np.arange(7,10)),"cycles":5}
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
band_info["beta_0"] = {"freqs":list(np.arange(13,22)),"cycles":9}
band_info["beta_1"] = {"freqs":list(np.arange(22,31)),"cycles":9}
band_info["gamma_0"] = {"freqs":list(np.arange(31,41)),"cycles":9}
band_info["gamma_1"] = {"freqs":list(np.arange(41,60)),"cycles":9}
band_info["gamma_2"] = {"freqs":list(np.arange(60,91)),"cycles":9}
cyc_names = ["theta_0","alpha_0","alpha_1","beta_0","beta_1","gamma_0",
             "gamma_1","gamma_2"]
cyc_names = ["theta_0"]

cov = mne.read_cov("{}empty-cov.fif".format(proc_dir))
fs_labels = mne.read_labels_from_annot("fsaverage", "RegionGrowing_70",
                                       subjects_dir=subjects_dir)
for sub in subjs:
    fwd_name = "{dir}nc_{sub}_{sp}-fwd.fif".format(dir=proc_dir, sub=sub, sp=spacing)
    fwd = mne.read_forward_solution(fwd_name)
    src_name = "{dir}{sub}_{sp}-src.fif".format(dir=proc_dir, sub=sub, sp=spacing)
    src = mne.read_source_spaces(src_name)
    labels = mne.morph_labels(fs_labels,sub_key[sub],subject_from="fsaverage",
                              subjects_dir=subjects_dir)
    for run in runs:
        if run == "rest":
            epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(
              dir=proc_dir, sub=sub, run=run)
            epo = mne.read_epochs(epo_name)
        else:
            epos = []
            for wav_idx, wav_name in enumerate(wavs):
                epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
                  dir=proc_dir, sub=sub, run=run, wav=wav_name)
                temp_epo = mne.read_epochs(epo_name)
                temp_epo.interpolate_bads()
                epos.append(temp_epo)
            epo = mne.concatenate_epochs(epos)
            del epos
            # epo_name = "{dir}{sub}_{run}_byresp-epo.fif".format(dir=proc_dir,
            #                                                      sub=sub,
            #                                                      run=run)
            # epo = mne.read_epochs(epo_name)

        inv_op = mne.minimum_norm.make_inverse_operator(epo.info,fwd,cov)
        stcs = mne.minimum_norm.apply_inverse_epochs(epo,inv_op,lambda2,
                                                    method=inv_method,
                                                    pick_ori="normal")
        l_arr = [s.extract_label_time_course(labels,src,mode="pca_flip").astype("float32") for s in stcs]
        l_arr = np.array(l_arr)

        for cn in cyc_names:
            print(cn)
            f, c = band_info[cn]["freqs"], band_info[cn]["cycles"]

            dPTE = epo_dPTE(l_arr, f, epo.info["sfreq"], n_cycles=c, n_jobs=n_jobs)
            dPTE = TriuSparse(np.array(dPTE))
            dPTE.save("{dir}nc_{sub}_{run}_dPTE_{cn}.sps".format(dir=proc_dir,
                                                               sub=sub,
                                                               run=run,
                                                               cn=cn))
            # dPTE.save("{dir}nc_{sub}_{run}_byresp_dPTE_{cn}.sps".format(dir=proc_dir,
            #                                                             sub=sub,
            #                                                             run=run,
            #                                                             cn=cn))
            del dPTE
