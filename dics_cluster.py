import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import f_mway_rm,summarize_clusters_stc,f_threshold_mway_rm
import matplotlib.pyplot as plt
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

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
n_jobs = 4
f_ranges = [[7,14]]
#f_ranges = ["alpha"]
conds = ["visselten","audio"]
#conds = ["audio","visual"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
effect_idx = 0
factor_levels = [len(conds), len(wavs)]
#factor_levels = [len(conds)]
effects = ["A","B","A:B"]
#effects = ["A"]
perm_num = 1024
return_pvals=False
p_thresh = 0.01
threshold = f_threshold_mway_rm(len(subjs), [3,4], effects[effect_idx], p_thresh)
threshold = dict(start=0, step=0.1)
cond_str = conds[0]
for c in conds[1:]:
    cond_str += "_" + c
thresh_str = "tfce" if isinstance(threshold,dict) else p_thresh

if len(effects)==1:
    def stat_fun(*args):
        # get f-values only.
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=return_pvals)[0]
else:
    def stat_fun(*args):
        # get f-values only.
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                         effects=effects, return_pvals=return_pvals)[0][effect_idx]


# get connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)
del fs_src
exclude = np.load("{}fsaverage_exclude.npy".format(proc_dir))

for fr in f_ranges:
    X = [[] for wav in wavs for cond in conds]
    #X = [[] for cond in conds]
    for sub_idx,sub in enumerate(subjs):
        src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub,spacing))
        vertnos=[s["vertno"] for s in src]
        morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                         subject_to="fsaverage",
                                         spacing=5,
                                         subjects_dir=subjects_dir,
                                         smooth=None)

        idx = 0
        for cond_idx,cond in enumerate(conds):
            for wav_idx,wav in enumerate(wavs):
                X_temp = []
                if isinstance(fr,list):
                    stc_temp = mne.read_source_estimate(
                      "{dir}stcs/nc_{a}_{b}_{c}_{f0}-{f1}Hz_{d}-lh.stc".format(
                       dir=proc_dir,a=sub,b=cond,c=wav,f0=fr[0],f1=fr[1],
                       d=spacing))
                elif isinstance(fr,str):
                    stc_temp = mne.read_source_estimate(
                      "{dir}stcs/nc_{a}_{b}_{c}_{d}_{e}-lh.stc".format(
                       dir=proc_dir,a=sub,b=cond,c=wav, d=fr,
                       e=spacing))
                stc_temp = morph.apply(stc_temp)
                X_temp.append(stc_temp.data.transpose(1,0))
                X[idx].append(np.vstack(X_temp))
                idx += 1
    X = [(np.array(x)*1e+26).astype(np.float32) for x in X]
    del X_temp, morph, src
    #X = [x.mean(axis=1,keepdims=True) for x in X]
    f_obs, clusters, cluster_pv, H0 = clu = \
      mne.stats.spatio_temporal_cluster_test(X,connectivity=cnx,n_jobs=n_jobs,
                                             threshold=threshold,
                                             stat_fun=stat_fun,
                                             n_permutations=perm_num,
                                             spatial_exclude=exclude)
    raw_f = stc_temp.copy()
    raw_f.data = f_obs.T
    if isinstance(f_ranges,list):
        raw_f.save("{}stc_f_{}-{}Hz_{}_{}_{}".format(proc_dir, fr[0], fr[1], cond_str,
                                                         thresh_str, effects[effect_idx]))
        with open("{}clu_{}-{}Hz_{}_{}_{}".format(proc_dir, fr[0], fr[1], cond_str,
                                                  thresh_str, effects[effect_idx]),"wb") as f:
            pickle.dump(clu,f)
        try:
            stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",p_thresh=0.05)
            stc_clu.save("{}stc_clu_{}-{}Hz_{}_{}_{}".format(proc_dir, fr[0], fr[1],
                                                             cond_str,
                                                             thresh_str,
                                                             effects[effect_idx]))
        except:
            print("No significant results")
    elif isinstance(fr,str):
        raw_f.save("{}stc_f_{}_{}_{}_{}".format(proc_dir, fr, cond_str, thresh_str, effects[effect_idx]))
        with open("{}clu_{}_{}_{}_{}".format(proc_dir, fr, cond_str, thresh_str, effects[effect_idx]),"wb") as f:
            pickle.dump(clu,f)
        try:
            stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",p_thresh=0.05)
            stc_clu.save("{}stc_clu_{}_{}_{}_{}".format(proc_dir, fr, cond_str, thresh_str, effects[effect_idx]))
        except:
            print("No significant results")











# clu_fig = mlab.figure()
# stc_clu.plot(hemi="lh",clim=dict(kind='value',lims=[0,3.5,7]), surface="white", figure=clu_fig)
# # except:
# #     print("No significant results.")
#
# masks = [stc_clu.data[:,x] for x in range(1,stc_clu.data.shape[1])]
# mask = stc_clu.data[:,0]
# mask[mask>0] = 1
# mask_inds = np.where(mask)[0]
#
# XX = np.array(X).mean(axis=1).mean(axis=1)
# XXX = []
# for idx in range(0,len(conds)*4,4):
#     XXX.append(XX[idx:idx+4,].mean(axis=0))
# XXX = np.array(XXX)*mask
#
# if XXX.shape[0]>2:
#     for cond_idx in range(XXX.shape[0]):
#         stc_ttemp = stc_temp.copy()
#         stc_ttemp.data = np.expand_dims(XXX[cond_idx,],1)
#         mfig = mlab.figure()
#         stc_ttemp.plot(hemi="both",figure=mfig,clim={"kind":"value",
#                        "lims":[1.45e-26,1.55e-26,1.65e-26]})
#         #stc_ttemp.plot(hemi="both",figure=mfig)
#         mlab.title(conds[cond_idx])
# else:
#     first = 0
#     second = 1
#     XXXX = XXX[first,] - XXX[second,]
#     stc_ttemp = stc_temp.copy()
#     stc_ttemp.data = np.expand_dims(XXXX,1)
#     mfig = mlab.figure()
#     stc_ttemp.plot(hemi="lh",figure=mfig,clim={"kind":"value",
#                    "lims":[4.46e-28,1.783e-27,3.12e-27]},surface="white")
#     #stc_ttemp.plot(hemi="both",figure=mfig)
#     mlab.title("{} minus {}".format(conds[first],conds[second]))
#
# avg = XXX[:,mask_inds].mean(axis=1)
# sem = stats.sem(XXX[:,mask_inds],axis=1)
# plt.bar(np.arange(len(conds)),avg,yerr=sem,tick_label=conds)
