import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import f_mway_rm, f_threshold_mway_rm
import matplotlib.pyplot as plt
from surfer import Brain
plt.ion()


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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

band_info = {}
# band_info["theta_0"] = {"freqs":list(np.arange(3,7)),"cycles":3}
# band_info["alpha_0"] = {"freqs":list(np.arange(7,10)),"cycles":5}
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}
# band_info["beta_0"] = {"freqs":list(np.arange(13,22)),"cycles":9}
# band_info["beta_1"] = {"freqs":list(np.arange(22,31)),"cycles":9}
# band_info["gamma_0"] = {"freqs":list(np.arange(31,41)),"cycles":9}
# band_info["gamma_1"] = {"freqs":list(np.arange(41,60)),"cycles":9}
# band_info["gamma_2"] = {"freqs":list(np.arange(60,90)),"cycles":9}

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = 4
n_jobs = 4
parc = "RegionGrowing_70"

with open("{}{}_apriori".format(proc_dir,parc),"rb") as f:
    labels = pickle.load(f)

conds = ["audio", "visual", "visselten"]
conds = ["audio", "visual"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
effect_idx = 0
factor_levels = [len(conds), len(wavs)]
effects = ["A","B","A:B"]
return_pvals=False
p_thresh = 0.001
threshold = f_threshold_mway_rm(len(subjs), [3,4], effects[effect_idx], p_thresh)
threshold = dict(start=0, step=0.1)
cond_str = conds[0]
for c in conds[1:]:
    cond_str += "_" + c
thresh_str = "tfce" if isinstance(threshold,dict) else p_thresh

# get connectivity
fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))
cnx = mne.spatial_src_connectivity(fs_src)
#del fs_src
exclude = np.load("{}fsaverage_ico{}_exclude.npy".format(proc_dir,spacing))
results = []
figures = []
brains = []
for k,v in band_info.items():
    fr = v["freqs"]
    band = k
    X = [[[] for wav in wavs for cond in conds] for lab in labels]
    for sub_idx,sub in enumerate(subjs):
        src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,sub,spacing))
        vertnos=[s["vertno"] for s in src]
        morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                         subject_to="fsaverage",
                                         spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         smooth=None)

        idx = 0
        for cond_idx,cond in enumerate(conds):
            # comment either this out, or the wav loop
            # X_temp = []
            # stc_temp = mne.read_source_estimate(
            #   "{dir}stcs/nc_{a}_{b}_{f0}-{f1}Hz_ico{d}-lh.stc".format(
            #    dir=proc_dir,a=sub,b=cond,f0=fr[0],f1=fr[-1],
            #    d=spacing))
            # stc_temp = morph.apply(stc_temp)
            # X_temp.append(stc_temp.data.transpose(1,0))
            # X[idx].append(np.vstack(X_temp))
            # idx += 1
            for wav_idx,wav in enumerate(wavs):
                stc_temp = mne.read_source_estimate(
                  "{dir}stcs/nc_{a}_{b}_{c}_{f0}-{f1}Hz_ico{d}-lh.stc".format(
                   dir=proc_dir,a=sub,b=cond,c=wav,f0=fr[0],f1=fr[-1],
                   d=spacing))
                stc_temp = morph.apply(stc_temp)
                for lab_idx, lab in enumerate(labels):
                    X_temp = mne.extract_label_time_course(stc_temp,lab,fs_src,mode="pca_flip")
                    X[lab_idx][idx].append(X_temp.mean(axis=-1).squeeze())
                idx += 1
    X = [[(np.array(x)*1e+26).astype(np.float32) for x in xx] for xx in X]
    del X_temp, morph, src
    X = np.array(X).swapaxes(0, 2)
    result = f_mway_rm(X, factor_levels=factor_levels, effects=effects)[0][0]
    sig_areas = np.where(result[1]<0.05)[0]
    if sig_areas.size==0: continue
    for sa in np.nditer(sig_areas):
        title = "{}_{}".format(k, labels[sa].name)
        figures.append(mlab.figure(title))
        brains.append(Brain('fsaverage', 'both', 'inflated', alpha=0.7,
                      subjects_dir=subjects_dir, figure=figures[-1]))
        brains[-1].add_annotation(parc,color="black")
        brains[-1].add_label(labels[sa])
        plt.figure()
        these_means = X[...,sa]
        these_means = np.array([these_means[...,idx:idx+4].mean(axis=-1) for idx in range(0,len(conds)*4,4)])
        errs = stats.sem(these_means,axis=1)
        these_means = these_means.mean(axis=1)
        plt.bar(np.arange(len(these_means)),these_means,yerr=errs)
        plt.title(title)
    results.append(result)
