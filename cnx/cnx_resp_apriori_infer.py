import pickle
from mne.stats import fdr_correction
from cnx_utils import plot_directed_cnx
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResults
from os import listdir
import mne
from mayavi import mlab
import re
import matplotlib.pyplot as plt
plt.ion()

freq = "alpha_0"
freq_definitions = {"theta_0":"3-6","alpha_0":"7-9","alpha_1":"10-12"}
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
    if not re.search("byresp.pickle",filename):
        continue
    source_idx = int(re.search("src_([0-9]*)",filename).groups()[0])
    dest_idx = int(re.search("dest_([0-9]*)",filename).groups()[0])
    models[source_idx][dest_idx] = MixedLMResults.load("{}{}/{}".format(proc_dir,freq,filename))

indep_vars = ["RT","RT:Block[T.visselten]","RT:Block[T.visual]"]
indep_vars = ["RT"]
iv_disp = {"RT":"RT","RT:Block[T.visselten]":"RT.block.visselten","RT:Block[T.visual]":"RT.block.visual"}

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
            if not fdr[f_idx]<fdr_thresh:
                continue
            if iv == "visual-visselten":
                con = v.t_test(np.array([[0,1,-1,0,0,0,0,0]])).summary_frame()
                mat[sub_labels_inds[source_idx],k] = con["coef"].values[0]
            else:
                mat[sub_labels_inds[source_idx],k] = v.params.get(iv)
        mats.append(mat)

mat_max = np.array(mats).max()
mat_min = np.array(mats).min()
cond_labels = ["{} {}".format(reg, iv_disp[iv]) for iv in indep_vars for reg in ["left A1", "right A1", "left V1", "right V1"]]

# make displays
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
    plt.savefig("{}{}/{}_{}.png".format(proc_dir,freq,freq,cl))
    plt.close()
