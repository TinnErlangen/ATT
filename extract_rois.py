import mne
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import sem
from mayavi import mlab
from surfer import Brain
plt.ion()

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
clust_thresh = 0.25
fuse_regions=False
normalise=True
custom_regions = [["superiortemporal_1-lh","superiortemporal_2-lh",
                    "superiortemporal_3-lh","superiortemporal_4-lh",
                    "supramarginal_1-lh","supramarginal_2-lh",
                    "supramarginal_3-lh","supramarginal_6-lh",
                    "transversetemporal_1-lh","transversetemporal_2-lh"],
                    ["superiortemporal_1-rh","superiortemporal_2-rh",
                     "superiortemporal_3-rh","superiortemporal_4-rh",
                     "supramarginal_4-rh","supramarginal_5-rh",
                     "supramarginal_6-rh","supramarginal_7-rh",
                     "supramarginal_8-rh","supramarginal_9-rh",
                     "transversetemporal_1-rh"]]
#custom_regions = None

reg_names = ["lateraloccipital-lh","lateraloccipital-rh"]

# draw overall anova with parcellations, lateral and medial views
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'both', subjects_dir=subjects_dir)
names = [label.name for label in labels]

if fuse_regions:
    fused_labels = {}
    for l in labels:
        if "unknown" in l.name:
            continue
        spli = re.split("_",l.name)
        root_name = spli[0]
        hemisphere = re.split("-",spli[1])[1]
        new_name = root_name+"-"+hemisphere
        if new_name in fused_labels:
            fused_labels[new_name] += l
        else:
            fused_labels[new_name] = l
    labels = []
    for k,v in fused_labels.items():
        v.name = k
        labels.append(v)


if custom_regions:
    new_labels = [l for l in labels if l.name == custom_regions[0][0]]
    new_labels.append([l for l in labels if l.name == custom_regions[1][0]][0])
    for cr_idx,cr in enumerate(custom_regions):
        for l in cr:
            new_labels[cr_idx] += [ll for ll in labels if ll.name == l][0]
    these_labels = new_labels
else:
    these_labels = [l for l in labels if l.name in reg_names]

names = [label.name for label in labels]
colors = [label.color for label in labels]

# now get the raw DICS
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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29","ATT_31","ATT_33",
         "ATT_34", "ATT_35", "ATT_36","ATT_37"]
#subjs = ["ATT_10", "ATT_11"]
f_ranges = [[4,7],[8,14],[15,30]]
all_f = [item for sublist in [list(np.arange(f[0],f[1]+1)) for f in f_ranges] for item in sublist]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
conds = ["rest","audio","visual","visselten","zaehlen"]
conds = ["audio","visual","visselten"]
contrasts = [(-1,1,0),(0,1,-1)]
cont_names = ["visual minus audio", "visual minus visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
#conds = ["audio","zaehlen"]
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",
                                                         spacing))

X = [[] for cond in conds]
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
        X_temp = []
        for fr in f_ranges:
            stc_temp = mne.read_source_estimate(
                    "{dir}stcs/nc_{a}_{b}_i{f0}-{f1}Hz_{d}-lh.stc".format(
                      dir=proc_dir,a=sub,b=cond,f0=fr[0],f1=fr[1],
                      d=spacing))
            stc_temp = morph.apply(stc_temp)
            X_temp.append(mne.extract_label_time_course(stc_temp,these_labels,fs_src,mode="mean"))
        X[idx].append(np.hstack(X_temp))
        idx += 1
XX = [np.array(x) for x in X]

subjs_t = subjs + ["mean"]
band_division = {"theta":[1,5],"alpha":[5,12],"beta":[13,28],"low gamma":[28,46],"high gamma":[46,-1]}
band_division = {"alpha":[2,11]}
band_division = {"theta-alpha":[0,12],"beta":[13,28]}

for k,v in band_division.items():
    XXX = np.array(XX)
    if normalise:
        for r_idx in range(XXX.shape[2]):
            for s_idx in range(XXX.shape[1]):
                xm = XXX[:,s_idx,r_idx,v[0]:v[1]].mean()
                xs = XXX[:,s_idx,r_idx,v[0]:v[1]].std()
                XXX[:,s_idx,r_idx,v[0]:v[1]] = (XXX[:,s_idx,r_idx,v[0]:v[1]]-xm)/xs
    vmax = XXX[:,:,:,v[0]:v[1]].max()
    vmax = 3
    vmin = -3
    XXX = np.hstack((XXX,XXX.mean(axis=1,keepdims=True)))
    temp_conds = conds.copy()
    for cont_idx,cont in enumerate(contrasts):
        temp_conds.append(cont_names[cont_idx])
        cont_result = 0
        for c_idx,c in enumerate(cont):
            cont_result += XXX[c_idx,] * c
        cont_result = np.expand_dims(cont_result,0)
        XXX = np.vstack((XXX,cont_result))
    fig, axes = plt.subplots(len(these_labels),len(temp_conds)+1,figsize=(12,8))
    for tl_idx,this_label in enumerate(these_labels):
        mfig = mlab.figure()
        hemi = "lh" if "lh" in this_label.name else "rh"
        brain = Brain('fsaverage', hemi, 'inflated', subjects_dir=subjects_dir,
                      cortex='low_contrast', size=(800, 600),figure=mfig)
        brain.add_label(this_label)
        axes[tl_idx,0].imshow(mlab.screenshot(figure=mfig))
        axes[tl_idx,0].axis("off")
        mlab.close()
        for ax_idx,ax in enumerate(axes[tl_idx,1:]):
            ax.imshow(XXX[ax_idx,:,tl_idx,v[0]:v[1]],vmin=vmin,vmax=vmax,cmap="seismic")
            ax.set_title(temp_conds[ax_idx])
            plt.sca(ax)
            if ax_idx == len(temp_conds)-1:
                plt.yticks(ticks=np.arange(len(subjs_t)),labels=subjs_t)
                ax.yaxis.tick_right()
            else:
                ax.yaxis.set_visible(False)
            plt.xticks(ticks=np.arange(v[1]-v[0]),labels=all_f[v[0]:v[1]])
            plt.xlabel("Hz")
