import mne
import bezier
from surfer import Brain
from mayavi import mlab
from cnx_utils import TriuSparse, load_sparse
#mlab.options.offscreen = True
import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.ion()

proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
band = "alpha_1"
figsize = (3840,2160)
parc = "RegionGrowing_70"
src_fname = "fsaverage_ico5-src.fif"
subjects_dir = "/home/jeff/freesurfer/subjects/"
top_cnx = 100
lineres=250
lingrad = np.linspace(0,1,lineres)
azi_incr = 0.1
labels = mne.read_labels_from_annot("fsaverage",parc)
srcs = mne.read_source_spaces(proc_dir+src_fname)
inuse = [src["inuse"].astype(bool) for src in srcs]

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
conds = ["audio", "visual", "visselten"]
conds = ["visual", "audio"]

brains = []
for cond in conds:
    dPTE = []
    for sub in subjs:
        fname = "{}nc_{}_{}_dPTE_{}.sps".format(mat_dir, sub, cond, band)
        temp_dPTE = load_sparse(fname)
        temp_dPTE[np.abs(temp_dPTE)==np.inf] = np.nan
        temp_dPTE = np.mean(temp_dPTE,axis=0,keepdims=True)
        dPTE.append(temp_dPTE)
    dPTE = np.array(dPTE).mean(axis=0)
    mat_inds = np.triu_indices(dPTE.shape[-1],k=1,m=dPTE.shape[1])
    fig = mlab.figure(size=figsize)
    brains.append(Brain('fsaverage', 'both', 'inflated', alpha=0.7,
                  subjects_dir=subjects_dir, figure=fig))
    brain = brains[-1]
    brain.add_text(0, 0, band, "band_title", font_size=40)
    brain.add_text(0, 0.8, cond, "cond_title", font_size=40)
    brain.add_annotation(parc,color="gray")
    if labels:
        rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])
    else:
        rrs = np.vstack([brain.geo["lh"].coords[inuse[0]],
                         brain.geo["rh"].coords[inuse[1]]])
    temp_pte = dPTE[0,].copy()
    temp_pte[np.where(temp_pte)] = temp_pte[np.where(temp_pte)] - 0.5
    alpha_max = np.abs(temp_pte).max()
    thresh = np.sort(np.abs(temp_pte.flatten()))[-top_cnx]
    alpha_min = thresh
    temp_pte[np.abs(temp_pte)<thresh] = 0

    inds_pos = np.where(temp_pte>0)
    origins = rrs[inds_pos[0],]
    dests = rrs[inds_pos[1],]
    inds_neg = np.where(temp_pte<0)
    origins = np.vstack((origins,rrs[inds_neg[1],]))
    dests = np.vstack((dests,rrs[inds_neg[0],]))
    inds = (np.hstack((inds_pos[0],inds_neg[0])),np.hstack((inds_pos[1],inds_neg[1])))

    area_red = np.zeros(len(labels))
    area_blue = np.zeros(len(labels))
    np.add.at(area_red,inds_pos[0],1)
    np.add.at(area_blue,inds_pos[1],1)
    np.add.at(area_red,inds_neg[1],1)
    np.add.at(area_blue,inds_neg[0],1)
    area_weight = area_red + area_blue
    area_red = area_red/np.max((area_red.max(),area_blue.max()))
    area_blue = area_blue/np.max((area_red.max(),area_blue.max()))
    area_weight = area_weight/area_weight.max()

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2
    alphas = ((np.abs(temp_pte[inds[0],inds[1]])-alpha_min)/(alpha_max-alpha_min))
    alphas[alphas<0] = 0

    mlab.points3d(origins[:,0],origins[:,1],origins[:,2],
                  alphas,scale_factor=10,color=(1,0,0),transparent=True)
    mlab.points3d(dests[:,0],dests[:,1],dests[:,2],
                  alphas,scale_factor=10,color=(0,0,1),transparent=True)
    for l_idx, l in enumerate(labels):
        if area_weight[l_idx] == 0:
            continue
        brain.add_label(l,color=(area_red[l_idx],0,area_blue[l_idx]),
                        alpha=area_weight[l_idx])
    spl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                      [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                      [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                      degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)
        mlab.plot3d(spl_pts[idx,0,],spl_pts[idx,1,],spl_pts[idx,2,],
                    lingrad*255,tube_radius=alphas[idx]*2,colormap="RdBu",
                    opacity=alphas[idx])
