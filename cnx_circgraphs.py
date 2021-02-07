import mne
from mne.viz import plot_connectivity_circle as pcc
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import re
from mayavi import mlab
from surfer import Brain
plt.ion()

datasets = ["left M1", "left sup-parietal", "right M1", "right sup-parietal"]
dataset = "left_M1"

proc_dir = "/home/jeff/ATT_dat/lmm/alpha_1/"
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
ln_n = len(label_names)
h0_thresh = 0.05
vmin = -0.03
vmax = 0.03
no_Z = False
top_cnx = 8

# order of labels from posterior to anterior
ypos = np.array([l.pos[:,1].mean() for l in labels if "lh" in l.name])
lh_label_names = [ln for ln in label_names if "lh" in ln]
ordered_label_names = [lh_label_names[idx][:-3] for idx in list(np.argsort(ypos))]

if no_Z:
    cnx_pd = pd.read_pickle("{}cnx_{}_no_Z.pickle".format(proc_dir, dataset))
else:
    cnx_pd = pd.read_pickle("{}cnx_{}.pickle".format(proc_dir, dataset))
cnx_pd = cnx_pd[cnx_pd["p"]<h0_thresh]
blocks = list(set(list(cnx_pd["Block"])))

cons = {}
for block in blocks:
    this_cnx_pd = cnx_pd[cnx_pd["Block"]==block]
    these_regs = list(this_cnx_pd["OutRegion"])
    these_reg_inds = np.array([label_names.index(this_reg) for this_reg in these_regs])
    con = np.zeros(len(label_names))
    con[these_reg_inds] = this_cnx_pd["est_dPTE"].values
    con[con!=0] -= 0.5
    cons[block] = con

# subtractions
norest = blocks.copy()
norest.remove("rest")
for block in norest:
    con = cons[block]
    con_inds = np.where(con)
    new_con = np.zeros_like(con)
    for i in zip(*con_inds):
        new_con[i] = con[i] - cons["rest"][i]
    if top_cnx:
        thresh = np.sort(np.abs(new_con).flatten())[-top_cnx]
        new_con[abs(new_con)<thresh] = 0
    cons["{}-rest".format(block)] = new_con

# get index of all connections significant in at least one condition
reg_inds = np.empty(0).astype(int)
for k,v in cons.items():
    if "-rest" not in k:
        continue
    reg_inds = np.concatenate((reg_inds, np.where(v)[0]))
reg_inds = np.unique(reg_inds)
# reduce labels/names
label_names_n = [label_names[idx] for idx in np.nditer(reg_inds)]
labels_n = [labels[idx] for idx in np.nditer(reg_inds)]
# if a region is present in one hemisphere, include it in the other
bh_reg_inds = []
for r_idx, lnn in zip(reg_inds, label_names_n):
    bh_reg_inds.append(label_names.index(lnn[:-3]+"-lh"))
    bh_reg_inds.append(label_names.index(lnn[:-3]+"-rh"))
bh_reg_inds = np.unique(np.array(bh_reg_inds))
bh_label_names = [label_names[idx] for idx in bh_reg_inds]
bh_ordered_names = []
for oln in ordered_label_names:
    if oln+"-lh" in bh_label_names or oln+"-rh" in bh_label_names:
        bh_ordered_names.append(oln)

# angles and colors by hemisphere and posterior-anterior
start_ang = 270
offset_ang = 10
angle_incr = (180-offset_ang*2) / (len(bh_label_names)/2)
angles = np.zeros_like(bh_reg_inds)
node_names = []
colors = []
this_cm = cm.get_cmap("plasma")
gradient = np.linspace(0,1,len(bh_ordered_names))
for ln_idx, ln in enumerate(bh_label_names):
    hemi = "left" if "lh" in ln else "right"
    idx = bh_ordered_names.index(ln[:-3])
    ang = offset_ang + idx * angle_incr
    colors.append(this_cm(gradient[idx]))
    if hemi == "left":
        angles[ln_idx] = start_ang - ang
    else:
        ang = start_ang + ang
        if ang > 360:
            angles[ln_idx] = ang - 360
        else:
            angles[ln_idx] = ang
    node_names.append(ln)

angles = np.hstack((angles, start_ang))
node_names.append(dataset)
colors.append((1,1,1,1))

for block in blocks:
    if block == "rest":
        continue
    this_con = cons["{}-rest".format(block)]
    con = np.hstack((this_con[bh_reg_inds], 0))
    alphas = abs(con)/vmax
    node_n = len(node_names)
    con_mat = np.zeros((node_n, node_n))
    con_mat[-1,] = con
    pcc(con_mat,node_names, node_angles=angles, colormap="seismic", node_colors=colors,
        vmin=vmin, vmax=vmax, title=block, alphas=None, facecolor="white",
        textcolor="black")

fig = mlab.figure()
subjects_dir = "/home/jeff/freesurfer/subjects"
brain = Brain('fsaverage', 'lh', "inflated", alpha=1,
              subjects_dir=subjects_dir, figure=fig)
for bon_idx, bon in enumerate(bh_ordered_names):
    this_label = [l for l in labels if l.name == bon+"-lh"][0]
    brain.add_label(this_label, color=this_cm(gradient[bon_idx]))
