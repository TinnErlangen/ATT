import mne
import bezier
from surfer import Brain
from mayavi import mlab
#mlab.options.offscreen = True
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
from cnx_utils import *
from scipy.special import comb
from scipy.stats import chi2


proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
figsize = (3840,2160)
src_fname = "fsaverage_ico5-src.fif"
subjects_dir = "/home/jeff/freesurfer/subjects/"
annotation = "RegionGrowing_70"
colors = [(1.,1.,0.), (0.,1.,1.)]
top_cnx = 50
kappa_quant = 0.99
lineres=250
lingrad = np.linspace(0,1,lineres)
labels = mne.read_labels_from_annot("fsaverage", annotation)
srcs = mne.read_source_spaces(proc_dir+src_fname)
inuse = [src["inuse"].astype(bool) for src in srcs]
sub_avg = True

conds = ["visual","visselten"]
freqs = ["theta"]

if not ((len(conds),len(freqs)) == (1,2) or (len(conds),len(freqs)) == (2,1)):
    raise ValueError("Conds must length 2 and freqs length 1 or vice verse.")
proc_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

if sub_avg:
    sub_lapls = [[],[]]
    for sub in subjs:
        idx = 0
        for cond in conds:
            for freq in freqs:
                dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir,
                                                                   sub, cond,
                                                                   freq))
                sub_lapls[idx].append(dPTE_to_laplace(dPTE).mean(axis=0))
                idx += 1
    sub_lapls = np.array(sub_lapls)
    lapl_a = sub_lapls[0,]
    lapl_b = sub_lapls[1,]
else:
    file_a = "nc_ATT_14_audio_dPTE_theta"
    file_b = "nc_ATT_14_visual_dPTE_theta"
    dPTE_a = load_sparse("{}{}.sps".format(proc_dir,file_a))
    dPTE_b = load_sparse("{}{}.sps".format(proc_dir,file_b))
    lapl_a = dPTE_to_laplace(dPTE_a)
    lapl_b = dPTE_to_laplace(dPTE_b)

df = comb(lapl_a.shape[1],2)
t2, kappa = samp2_chi(lapl_a,lapl_b)
pval = 2*np.min((chi2.cdf(t2,df),1-chi2.cdf(t2,df)))

chis = np.linspace(chi2.ppf(0.00001,df),chi2.ppf(0.99999,df),1000)
ps = chi2.pdf(chis,df)
fig, ax = plt.subplots(1,1)
ax.plot(chis,ps)
ax.axvline(chi2.ppf(0.025,df),color="red")
ax.axvline(chi2.ppf(0.975,df),color="red")
plt.scatter(t2,chi2.pdf(t2,df),color="black")
print("Chi squared: {}\ndf: {}\np: {}".format(t2,df,pval))

# plot
means = [np.triu(lapl_a.mean(axis=0),k=1)*-1, np.triu(lapl_b.mean(axis=0),k=1)*-1]
brains = []
for mean_idx, (mean, color) in enumerate(zip(means, colors)):
    fig = mlab.figure(size=figsize)
    brains.append(Brain('fsaverage', 'both', 'inflated', alpha=1,
                  subjects_dir=subjects_dir, figure=fig))
    brain = brains[-1]
    if len(freqs)>1:
        brain.add_text(0, 0, freqs[mean_idx], "band_title", font_size=40)
    else:
        brain.add_text(0, 0, freqs[0], "band_title", font_size=40)
    if len(conds)>1:
        brain.add_text(0, 0.8, conds[mean_idx], "cond_title", font_size=40)
    else:
        brain.add_text(0, 0.8, conds[0], "cond_title", font_size=40)
    brain.add_annotation(annotation,color="black")
    rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])

    alpha_max = mean.max()
    thresh = np.sort(mean.flatten())[-top_cnx]
    alpha_min = thresh
    mean[mean<thresh] = 0

    inds = np.where(mean>0)
    origins = rrs[inds[0],]
    dests = rrs[inds[1],]

    area_weight = np.zeros(len(labels))
    np.add.at(area_weight,inds[0],mean[inds])
    np.add.at(area_weight,inds[1],mean[inds])
    area_weight = area_weight/area_weight.max()

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2.3
    alphas = ((np.abs(mean[inds[0],inds[1]])-alpha_min)/(alpha_max-alpha_min))*.9+.1

    mlab.points3d(origins[:,0],origins[:,1],origins[:,2],
                  alphas,scale_factor=10,color=color,transparent=True)
    mlab.points3d(dests[:,0],dests[:,1],dests[:,2],
                  alphas,scale_factor=10,color=color,transparent=True)
    for l_idx, l in enumerate(labels):
        if area_weight[l_idx] == 0:
            continue
        brain.add_label(l,color=color, alpha=area_weight[l_idx])
    spl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                      [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                      [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                      degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)
        mlab.plot3d(spl_pts[idx,0,],spl_pts[idx,1,],spl_pts[idx,2,],
                    color=color, tube_radius=alphas[idx]*1.5, opacity=alphas[idx])

# and now the difference
# filter out signficant diagonal elements and put back in matrix form
temp_mat = np.zeros((mean.shape))
triu_inds = np.triu_indices(mean.shape[-1],k=0)
temp_mat[triu_inds[0],triu_inds[1]] = kappa
kappa = np.triu(temp_mat,k=1)

# threshold kappa and use it to mask the difference
triu_inds = np.triu_indices(mean.shape[-1],k=1)
kappa_thresh = np.quantile(kappa[triu_inds],kappa_quant)
kappa[kappa<kappa_thresh] = 0
kappa[kappa>0] = 1
means = [np.triu(lapl_a.mean(axis=0),k=1)*-1, np.triu(lapl_b.mean(axis=0),k=1)*-1]
diff = means[0] - means[1]
#diff_thresh = np.sort(np.abs(diff).flatten())[-top_cnx]
#diff[np.abs(diff)<diff_thresh] = 0
diff *= kappa

# graph it out

fig = mlab.figure(size=figsize)
brains.append(Brain('fsaverage', 'both', 'inflated', alpha=1,
              subjects_dir=subjects_dir, figure=fig))
brain = brains[-1]

if len(freqs)>1:
    brain.add_text(0.7, 0.8, freqs[0], "freq_title1", font_size=40, color=colors[0])
    brain.add_text(0.7, 0., freqs[1], "freq_title2", font_size=40, color=colors[1])
else:
    brain.add_text(0, 0, freqs[0], "band_title", font_size=40)
if len(conds)>1:
    brain.add_text(0.7, 0.8, conds[0], "cond_title1", font_size=40, color=colors[0])
    brain.add_text(0.7, 0., conds[1], "cond_title2", font_size=40, color=colors[1])
else:
    brain.add_text(0, 0.8, conds[0], "cond_title", font_size=40)

brain.add_annotation(annotation,color="black")
rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])

# find the edges for both conditions
inds_1 = np.where(diff>0)
origins_1 = rrs[inds_1[0],]
dests_1 = rrs[inds_1[1],]
inds_2 = np.where(diff<0)
origins_2 = rrs[inds_2[0],]
dests_2 = rrs[inds_2[1],]
all_inds = [inds_1, inds_2]
all_origins = [origins_1, origins_2]
all_dests = [dests_1, dests_2]

# compute color balance and opacity for areas
area_col1 = np.zeros(len(labels))
area_col2 = np.zeros(len(labels))
np.add.at(area_col1, inds_1[0], diff[inds_1])
np.add.at(area_col1, inds_1[1], diff[inds_1])
np.add.at(area_col2, inds_2[0], np.abs(diff[inds_2]))
np.add.at(area_col2, inds_2[1], np.abs(diff[inds_2]))
common_max = np.max((area_col1.max(), area_col2.max()))
area_col1 = area_col1/common_max
area_col2 = area_col2/common_max
area_weight = area_col1 + area_col2
area_weight = area_weight/area_weight.max()
# apply to the areas
for l_idx, l in enumerate(labels):
    if area_weight[l_idx] == 0:
        continue
    color = area_col1[l_idx]*np.array(colors[0]) + \
            area_col2[l_idx]*np.array(colors[1])
    color[color>1] = 1
    color = tuple(color)
    brain.add_label(l,color=color, alpha=area_weight[l_idx])
    mlab.points3d(rrs[l_idx,0],rrs[l_idx,1],rrs[l_idx,2],area_weight[l_idx],
                  scale_factor=10,color=color)

# draw points and edges, loop through the two conditions
for color, inds, origins, dests in zip(colors, all_inds, all_origins, all_dests):
    alpha_max = np.abs(diff[inds]).max()
    alpha_min = np.abs(diff[inds]).min()
    alphas = (np.abs(diff[inds])-alpha_min)/(alpha_max-alpha_min)*0.9+0.1
    # mlab.points3d(origins[:,0],origins[:,1],origins[:,2],
    #               alphas,scale_factor=10,color=color,transparent=True)
    # mlab.points3d(dests[:,0],dests[:,1],dests[:,2],
    #               alphas,scale_factor=10,color=color,transparent=True)

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2.3
    pl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                      [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                      [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                      degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)
        mlab.plot3d(spl_pts[idx,0,],spl_pts[idx,1,],spl_pts[idx,2,],
                    color=color, tube_radius=alphas[idx]*1.5, opacity=alphas[idx])
