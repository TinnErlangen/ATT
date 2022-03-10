import mne
from mne.beamformer import make_dics,apply_dics_csd
from mne.time_frequency import csd_morlet
from mayavi import mlab
import pickle
import scipy.sparse
import numpy as np
from os import listdir
import pandas as pd
import re
from dPTE import epo_dPTE

def shuffle_df(df):
    subjs = list(df["Subj"].unique())
    col_idx = df.columns.get_loc("Brain")
    for subj in subjs:
        row_inds = np.where(df["Subj"]==subj)[0]
        these_brains = df.iloc[row_inds, col_idx].copy()
        these_brains = these_brains.sample(frac=1)
        df.iloc[row_inds, col_idx] = these_brains
    return df

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}

# all subjs
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

band_info = {}
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
bi = band_info["alpha_1"]

band = "alpha_1"
freqs = band_info[band]["freqs"]
subjects_dir = "/home/jev/hdd/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = 5
n_jobs = 4
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc=parc,subjects_dir=subjects_dir)
label_names = [lab.name for lab in labels]
snr = 1.0
lambda2 = 1.0 / snr ** 2

df_dict = {"Subj":[], "Cond":[], "DICS":[], "dPTE":[]}

runs = ["rest", "audio", "visual", "visselten"]
wavs = ["4000fftf", "4000Hz", "7000Hz", "4000cheby"]
ROIs = ["L3395-lh", "L8143-lh", "L7491_L4557-lh", "L4557-lh"]
# combine ROIs into one ROI then get rid of the individual ones
ROI_inds = [label_names.index(ROI) for ROI in ROIs]
new_label = labels[ROI_inds[0]]
for ROI_idx in ROI_inds[1:]:
    new_label += labels[ROI_idx]
new_label.name = "Broad_Motor"
labels = [lab for lab in labels if lab.name not in ROIs] + [new_label]
label_names = [lab.name for lab in labels]
ROI_idx = len(label_names)-1  # new label is last in the list

cov = mne.read_cov("{}empty-cov.fif".format(proc_dir))

for sub in subjs:
    src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,sub,spacing))
    morph_labels = mne.morph_labels(labels, sub_key[sub],
                                    subject_from="fsaverage",
                                    subjects_dir=subjects_dir)

    all_bads = []
    epos = []
    for run_idx,run in enumerate(runs):
        if run == "rest":
            epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(
              dir=proc_dir, sub=sub, run=run)
            epo = mne.read_epochs(epo_name)
            all_bads += epo.info["bads"]
            epos.append(epo)
        else:
            for wav_idx, wav_name in enumerate(wavs):
                epo_name = "{dir}nc_{sub}_{run}_{wav}_hand-epo.fif".format(
                  dir=proc_dir, sub=sub, run=run, wav=wav_name)
                epo = mne.read_epochs(epo_name)
                all_bads += epo.info["bads"]
                epos.append(epo)

    for x in epos:
        x.info["bads"] = all_bads
        x.info["dev_head_t"] = epos[0].info["dev_head_t"]
    epo = mne.concatenate_epochs(epos)
    rest_bool = epo.events[:,-1] == 255

    csd = csd_morlet(epo, frequencies=bi["freqs"], n_jobs=n_jobs,
                     n_cycles=bi["cycles"], decim=2)
    fwd_name = "{dir}nc_{sub}_ico{sp}-fwd.fif".format(dir=proc_dir, sub=sub,
                                                      sp=spacing)
    fwd = mne.read_forward_solution(fwd_name)
    filters = make_dics(epo.info, fwd, csd, real_filter=True,
                        weight_norm=None, reduce_rank=True,
                        pick_ori="max-power", label=morph_labels[ROI_idx])

    for event in range(len(epo)):
        print(event)
        event_csd = csd_morlet(epo[event], frequencies=bi["freqs"],
                               n_jobs=n_jobs, n_cycles=bi["cycles"], decim=2)
        stc, freqs = apply_dics_csd(event_csd, filters)
        # dics_datum = mne.extract_label_time_course(stc, morph_labels[ROI_idx],
        #                                            src, mode="mean").mean()
        dics_datum = stc.data.mean() * 1e+26
        df_dict["DICS"].append(dics_datum)
        df_dict["Subj"].append(sub)
        cond = "rest" if epo[event].events[0,-1] == 255 else "task"
        df_dict["Cond"].append(cond)

    # dPTE
    inv_op = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov)
    stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv_op, lambda2,
                                                 method="sLORETA",
                                                 pick_ori="normal")

    l_arr = mne.extract_label_time_course(stcs, morph_labels, src,
                                          mode="pca_flip")
    l_arr = np.array(l_arr)
    dPTE = epo_dPTE(l_arr, bi["freqs"], epo.info["sfreq"],
                    n_cycles=bi["cycles"], n_jobs=n_jobs, roi=ROI_idx)
    dPTE = dPTE[:, ROI_idx,]
    for epo_idx in range(len(dPTE)):
        avg_dpte = (dPTE[epo_idx,][dPTE[epo_idx,].nonzero()]-0.5).mean()
        df_dict["dPTE"].append(avg_dpte)

df = pd.DataFrame.from_dict(df_dict)
df.to_pickle("{}mu_dics_dpte.pickle".format(proc_dir))
