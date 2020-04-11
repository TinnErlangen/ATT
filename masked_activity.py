import mne
from mayavi import mlab
import pickle
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
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
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
stat_dir = "broad/"
spacing = "ico5"
conds = ["audio", "visual", "visselten"]
#conds = ["visual","visselten"]
cond_str = conds[0]
for c in conds[1:]:
    cond_str += "_" + c
thresh_str = "tfce"

# get connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir, "fsaverage",
                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)

f_ranges = [[7,14]]
for fr in f_ranges:
    ###### first get our clusters/masks
    clu_stc = mne.read_source_estimate("{}{}stc_clu_{}-{}Hz_{}_{}-lh.stc".format(
                                       proc_dir, stat_dir, fr[0], fr[1], cond_str,
                                       thresh_str))
    clu_stc.subject = "fsaverage"
    with open("{}{}clu_{}-{}Hz_{}_{}".format(proc_dir, stat_dir, fr[0], fr[1], cond_str,
                                             thresh_str),"rb") as f:
        clu = pickle.load(f)
    stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",p_thresh=0.085)
    meta_clusts = _find_clusters(stc_clu.data[:,0],1e-8,connectivity=cnx)[0]
    clust_labels = []
    for mc in meta_clusts:
        temp_stc = stc_clu.copy()
        temp_stc.data[:] = np.zeros((temp_stc.data.shape[0],1))
        temp_stc.data[mc,0] = 1
        lab = [x for x in mne.stc_to_label(temp_stc,src=fs_src) if x][0]
        clust_labels.append(lab)

    X = np.zeros((len(subjs),len(clust_labels),len(conds)))
    for sub_idx,sub in enumerate(subjs):
        src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub,spacing))
        vertnos=[s["vertno"] for s in src]
        morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                         subject_to="fsaverage",
                                         spacing=5,
                                         subjects_dir=subjects_dir,
                                         smooth=None)


        for cond_idx,cond in enumerate(conds):
            X_temp = []
            stc_temp = mne.read_source_estimate(
                    "{dir}stcs/nc_{a}_{b}_{f0}-{f1}Hz_{d}-lh.stc".format(
                      dir=proc_dir,a=sub,b=cond,f0=fr[0],f1=fr[1],
                      d=spacing))
            stc_temp = morph.apply(stc_temp)
            X[sub_idx,:,cond_idx] = mne.extract_label_time_course(stc_temp,clust_labels,fs_src,mode="mean")[:,0]


    for cl_idx,cl in enumerate(clust_labels):
        hemi = cl.hemi
        fig, axes = plt.subplots(3,1)
        mfig = mlab.figure()
        brain = Brain("fsaverage", hemi, "inflated",
                      subjects_dir=subjects_dir, figure=mfig)
        brain.add_label(cl,color="blue",alpha=0.7)
        axes[0].imshow(mlab.screenshot(figure=mfig))
        axes[0].axis("off")
        mlab.close()
        mfig = mlab.figure()
        brain = Brain("fsaverage", hemi, "inflated",
                      subjects_dir=subjects_dir, figure=mfig,
                      views=["med"])
        brain.add_label(cl,color="blue",alpha=0.7)
        axes[1].imshow(mlab.screenshot(figure=mfig))
        axes[1].axis("off")
        mlab.close()
        these_data = X[:,cl_idx,]
        temp_mean = these_data.mean(axis=0)
        temp_sem = sem(these_data,axis=0)
        axes[2].bar(np.arange(len(conds)),temp_mean,yerr=temp_sem,tick_label=conds)
        axes[2].set_ylim((0,temp_mean.max()+temp_sem.max()))
        plt.suptitle("{}-{}Hz, Cluster {}".format(fr[0],fr[1],cl_idx))
