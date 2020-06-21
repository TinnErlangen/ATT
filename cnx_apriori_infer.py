import pickle
from mne.stats import fdr_correction
from cnx_utils import plot_directed_cnx, load_sparse
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResults
from os import listdir
import mne
from mayavi import mlab
import re
import matplotlib.pyplot as plt
plt.ion()

freq = "alpha_0"
show_means = True
show_coeffs = False
freq_definitions = {"alpha_0":"7-9","alpha_1":"10-12"}
proc_dir = "/home/jeff/ATT_dat/proc/"

side_views = {}
side_views["left_side"] = (-180, 90.0, 630, np.array([ 47, -21, -16]))
side_views["top_side"] = (-119.01520162530294, 7.505784961226141, 762.8694309953917, np.array([  7.37395027, -34.47828734, -18.41562401]))
side_views["right_side"] = (-6, 90.0, 630, np.array([ 47, -21, -16]))
side_views["back_side"] = (-90.73679529657997, 74.68587396406461, 632.2314049586663, np.array([  5.8094426 , -33.97040829, -15.08325186]))

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
mat_n = len(labels)
regions = ["L2235-lh", "L2235-rh", "L2340_L1933-lh", "L2340_L1933-rh"]
sub_labels = []
sub_labels_inds = []
for reg in regions:
    for lab_idx,lab in enumerate(labels):
        if reg == lab.name:
            sub_labels.append(lab)
            sub_labels_inds.append(lab_idx)

fdr_thresh = 0.05
filelist = listdir(proc_dir+freq)
models = [{} for reg in regions]
for filename in filelist:
    if not re.search("[0-9].pickle",filename):
        continue
    source_idx = int(re.search("src_([0-9]*)",filename).groups()[0])
    dest_idx = int(re.search("dest_([0-9]*)",filename).groups()[0])
    models[source_idx][dest_idx] = MixedLMResults.load("{}{}/{}".format(proc_dir,freq,filename))

indep_vars = ["Block[T.visselten]","Block[T.visual]","Wav[T.4000cheby]",
              "Wav[T.4000fftf]", "Wav[T.7000Hz]", "Laut", "Angenehm","visual-visselten"]
iv_disp = {"Block[T.visselten]":"audio-visselten","Block[T.visual]":"audio-visual",
           "Wav[T.4000cheby]":"4000Hz-4000cheby","Wav[T.4000fftf]":"4000Hz-4000fftf",
           "Wav[T.7000Hz]":"4000Hz-7000Hz","Laut":"Laut","Angenehm":"Angenehm",
           "visual-visselten":"visual-visselten"}
# get pvals, correct, coefficients with significance
mats = []
for iv_idx,iv in enumerate(indep_vars):
    pvals = [{} for mod in models]
    for source_idx, mod in enumerate(models):
        mat = np.zeros((mat_n,mat_n))
        for k,v in mod.items():
            if iv == "visual-visselten":
                con = v.t_test(np.array([[0,1,-1,0,0,0,0,0]])).summary_frame()
                pvals[source_idx][k] = con["P>|z|"].values[0]
            else:
                pvals[source_idx][k] = v.pvalues.get(iv)
        hyp,fdr = fdr_correction(list(pvals[source_idx].values()),alpha=fdr_thresh)
        #fdr = list(pvals[source_idx].values()) # if this line not commented out, correction is not done
        for f_idx,(k,v) in enumerate(mod.items()):
            if not hyp[f_idx]:
                continue
            if iv == "visual-visselten":
                con = v.t_test(np.array([[0,1,-1,0,0,0,0,0]])).summary_frame()
                mat[sub_labels_inds[source_idx],k] = con["coef"].values[0]
            else:
                mat[sub_labels_inds[source_idx],k] = v.params.get(iv)
        mats.append(mat)

mat_max = np.array(mats).max()
mat_min = np.array(mats).min()
cond_labels = ["left_A1_auditory_visual", "right_A1_auditory_visual",
               "left_occ_auditory_visual", "right_occ_auditory_visual",
               "left_A1_auditory_visselten", "right_A1_auditory_visselten",
               "left_occ_auditory_visselten", "right_occ_auditory_visselten",]
cond_labels = ["{} {}".format(reg, iv_disp[iv]) for iv in indep_vars for reg in ["left A1", "right A1", "left V1", "right V1"]]

# make displays
if show_coeffs:
    for si, cl in enumerate(cond_labels):
        if (mats[si]==0).all():
            continue
        fig, axes = plt.subplots(2,4,figsize=(38.4,21.6))
        plt.suptitle("{}Hz {}".format(freq_definitions[freq],cl))
        split_mats = [[],[]]
        for mat in mats:
            temp_mat = mat.copy()
            temp_mat[temp_mat<0] = 0
            split_mats[0].append(temp_mat)
            temp_mat = mat.copy()
            temp_mat[temp_mat>0] = 0
            split_mats[1].append(temp_mat)
        for sm, axe, type_title in zip(split_mats, axes, ["Outflow", "Inflow"]):
            total_flow = np.sum(sm[si])
            if (sm[si]==0).all():
                for a in axe:
                    a.axis("off")
                continue
            mfig = mlab.figure(size=(1280,1024))
            brain = plot_directed_cnx(sm[si],labels,parc,fig=mfig,alpha_min=mat_min,
                                      alpha_max=mat_max, uniform_weight=True,
                                      alpha=0.99)
            mlab.view(*side_views["left_side"])
            axe[0].imshow(mlab.screenshot())
            axe[0].axis("off")
            axe[0].set_title("{}\n\nLeft Saggital View".format(type_title))
            mlab.view(*side_views["right_side"])
            axe[1].imshow(mlab.screenshot())
            axe[1].axis("off")
            axe[1].set_title("Right Saggital View")
            mlab.view(*side_views["top_side"])
            axe[2].imshow(mlab.screenshot())
            axe[2].axis("off")
            axe[2].set_title("Horizontal View")
            mlab.view(*side_views["back_side"])
            axe[3].imshow(mlab.screenshot())
            axe[3].axis("off")
            axe[3].set_title("Total estimated change in {}: {:.3f}\n\nRear Coronal View".format(type_title,total_flow))
            mlab.close()
        plt.savefig("{}{}/{}_{}".format(proc_dir,freq,freq,cl))
        plt.close()

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
conds = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
if show_means:
    dPTEs = [np.empty((0,len(labels),len(labels))) for cond in conds]
    for sub_idx,sub in enumerate(subjs):
        for cond_idx,cond in enumerate(conds):
            for wav_idx,wav in enumerate(wavs):
                dPTE = load_sparse("{}nc_{}_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                      cond, wav, freq))
                dPTEs[cond_idx] = np.vstack((dPTEs[cond_idx],dPTE))
dPTEs = [d.mean(axis=0) for d in dPTEs]
dPTEs = np.array(dPTEs)

contrasts = {"audio-visselten":[1,0,-1],"audio-visual":[1,-1,0],"visual-visselten":[0,1,-1]}
for si, cl in enumerate(cond_labels):
    con_match = np.array([con in cl for con in contrasts.keys()])
    if not any(con_match):
        continue
    else:
        contrast = contrasts[list(contrasts.keys())[np.where(con_match)[0][0]]]
        con_inds = list(np.where(contrast)[0]) + [-1]
    contrast_name = [con for con in contrasts.keys() if con in cl][0]
    if (mats[si]==0).all():
        continue
    fig, axes = plt.subplots(3,4,figsize=(38.4,21.6))
    plt.suptitle("{}Hz {}".format(freq_definitions[freq],cl))
    # figure out max and min across masked conditions
    temp_dPTEs = dPTEs[con_inds[:2],].copy()
    temp_dPTEs[:,mats[si]==0] = 0
    alpha_max = temp_dPTEs.max()
    alpha_min = temp_dPTEs[temp_dPTEs!=0].min()
    for ci,axe in zip(con_inds,axes):
        if ci == -1:
            temp_con = np.zeros(dPTEs.shape[1:])
            for con_idx,con in enumerate(contrast):
                temp_con += dPTEs[con_idx,] * con
            temp_con[mats[si]==0] = 0
        else:
            temp_con = dPTEs[ci,].copy()
            temp_con[mats[si]==0] = 0
        mfig = mlab.figure(size=(1280,1024))
        if ci == -1:
            brain = plot_directed_cnx(temp_con,labels,parc,fig=mfig,alpha=0.99)
        else:
            brain = plot_directed_cnx(temp_con,labels,parc,fig=mfig,alpha=0.99,
                                      alpha_min=alpha_min,alpha_max=alpha_max)
        mlab.view(*side_views["left_side"])
        axe[0].imshow(mlab.screenshot())
        axe[0].axis("off")
        if ci == -1:
            axe[0].set_title("{}\n\nLeft Saggital View".format(contrast_name))
        else:
            axe[0].set_title("{}\n\nLeft Saggital View".format(conds[ci]))
        mlab.view(*side_views["right_side"])
        axe[1].imshow(mlab.screenshot())
        axe[1].axis("off")
        axe[1].set_title("Right Saggital View")
        mlab.view(*side_views["top_side"])
        axe[2].imshow(mlab.screenshot())
        axe[2].axis("off")
        axe[2].set_title("Horizontal View")
        mlab.view(*side_views["back_side"])
        axe[3].imshow(mlab.screenshot())
        axe[3].axis("off")
        axe[3].set_title("Rear Coronal View")
        mlab.close()
    plt.savefig("{}{}/{}_raw_{}".format(proc_dir,freq,freq,cl))
    plt.close()
