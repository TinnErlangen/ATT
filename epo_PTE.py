import mne
import numpy as np
import argparse
from mne.time_frequency import tfr_array_morlet
from pyPTE.pyPTE import (get_delay, get_binsize, get_discretized_phase,
                         compute_dPTE_rawPTE)
from joblib import Parallel, delayed
import pickle

class TriuSparse():
    def __init__(self,mat,precision=np.float32):
        mat = mat.astype(precision)
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Last two dimensions must be the same.")
        self.mat_res = mat.shape[-1]
        self.mat_inds = np.triu_indices(self.mat_res,k=1,m=self.mat_res)
        self.mat_sparse = mat[...,self.mat_inds[0],self.mat_inds[1]]
    def save(self,filename):
        out_dics = {"mat_res":self.mat_res,"mat_inds":self.mat_inds,
                    "mat_sparse":self.mat_sparse}
        with open(filename,"wb") as f:
            pickle.dump(out_dics,f)

def load_sparse(filename,convert=True):
    with open(filename,"rb") as f:
        result = pickle.load(f)
    if convert:
        full_mat = np.zeros(result["mat_sparse"].shape[:-1] + \
          (result["mat_res"],result["mat_res"]))
        full_mat[...,result["mat_inds"][0],result["mat_inds"][1]] = \
          result["mat_sparse"]
        result = full_mat
    return result

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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks
mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}
runs = ["audio","visselten","visual"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]

inv_method="sLORETA"
snr = 1.0
lambda2 = 1.0 / snr ** 2
n_jobs = 8
spacing="ico5"
freqs = [list(np.arange(4,7)),list(np.arange(8,13)),list(np.arange(13,31)),
         list(np.arange(31,61))]
cycles = [3,5,7,9]
cyc_names = ["theta","alpha","beta","gamma"]
cov = mne.read_cov("{}empty-cov.fif".format(proc_dir))
fs_labels = mne.read_labels_from_annot("fsaverage", "aparc_sub",
                                       subjects_dir=subjects_dir)
for sub in subjs:
    fwd_name = "{dir}nc_{sub}_{sp}-fwd.fif".format(dir=proc_dir, sub=sub, sp=spacing)
    fwd = mne.read_forward_solution(fwd_name)
    src_name = "{dir}{sub}_{sp}-src.fif".format(dir=proc_dir, sub=sub, sp=spacing)
    src = mne.read_source_spaces(src_name)
    labels = mne.morph_labels(fs_labels,sub_key[sub],subject_from="fsaverage",
                              subjects_dir=subjects_dir)
    for run in runs:
        epos = []
        for wav_idx, wav_name in enumerate(wavs):
            epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
              dir=proc_dir, sub=sub, run=run, wav=wav_name)
            temp_epo = mne.read_epochs(epo_name)
            temp_epo.interpolate_bads()
            epos.append(temp_epo)
        epo = mne.concatenate_epochs(epos)
        inv_op = mne.minimum_norm.make_inverse_operator(epo.info,fwd,cov)
        stcs = mne.minimum_norm.apply_inverse_epochs(epo,inv_op,lambda2,
                                                    method=inv_method,
                                                    pick_ori="normal")
        l_arr = [s.extract_label_time_course(labels,src,mode="pca_flip") for s in stcs]
        l_arr = np.array(l_arr)

        for f,c,cn in zip(freqs,cycles,cyc_names):
            print(cn)
            phase = tfr_array_morlet(l_arr,200,f,n_cycles=c,n_jobs=n_jobs,output="phase")
            phase = phase.mean(axis=2)
            results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(do_PTE)(phase[i,]) for i in range(phase.shape[0]))
            dPTE,rPTE = zip(*results)
            del rPTE, results, phase
            dPTE = TriuSparse(np.array(dPTE))
            dPTE.save("{dir}nc_{sub}_{run}_dPTE_{cn}.sps".format(dir=proc_dir,
                                                               sub=sub,
                                                               run=run,
                                                               cn=cn))
            del dPTE
