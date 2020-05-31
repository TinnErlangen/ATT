import mne
from cnx_utils import TriuSparse, load_sparse, plot_directed_cnx
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
lineres = 250
top_cnx = 150
comp = 0
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
conds = ["audio", "rest"]

try:
    edges = np.load("{}{}-{}_{}_c{}.npy".format(proc_dir,conds[0],conds[1],band,comp))
except:
    edges = np.load("{}{}-{}_{}_c{}.npy".format(proc_dir,conds[1],conds[0],band,comp))

dPTEs = []
for cond in conds:
    dPTE = []
    for sub in subjs:
        fname = "{}nc_{}_{}_dPTE_{}.sps".format(mat_dir, sub, cond, band)
        temp_dPTE = load_sparse(fname)
        temp_dPTE[np.abs(temp_dPTE)==np.inf] = np.nan
        temp_dPTE = np.nanmean(temp_dPTE,axis=0,keepdims=True)
        dPTE.append(temp_dPTE)
    temp_pte = np.array(dPTE).mean(axis=0)[0,]
    temp_pte[np.where(temp_pte)] = temp_pte[np.where(temp_pte)] - 0.5
    temp_pte /= np.abs(temp_pte).max()
    thresh_pte = np.zeros(temp_pte.shape)
    # threshold by statistically significant edges
    for edge_idx in range(len(edges)):
        thresh_pte[edges[edge_idx,0],edges[edge_idx,1]] = \
          temp_pte[edges[edge_idx,0],edges[edge_idx,1]]
    # threshold by top x connections
    top_thresh = np.sort(np.abs(thresh_pte.flatten()))[-top_cnx]
    thresh_pte[np.abs(thresh_pte)<top_thresh] = 0
    dPTEs.append(thresh_pte)

con_pte = dPTEs[0] - dPTEs[1]
alpha_max = max([np.abs(d).max() for d in dPTEs])
alpha_min = min([np.abs(d[d!=0]).min() for d in dPTEs])
brain0 = plot_directed_cnx(dPTEs[0],labels,parc,figsize=figsize,
                           ldown_title=band,rup_title=conds[0],
                           alpha_min=alpha_min, alpha_max=alpha_max)
brain1 = plot_directed_cnx(dPTEs[1],labels,parc,figsize=figsize,
                           ldown_title=band,rup_title=conds[1],
                           alpha_min=alpha_min, alpha_max=alpha_max)
brain2 = plot_directed_cnx(con_pte,labels,parc,figsize=figsize,
                           ldown_title=band,
                           lup_title="{} - {}".format(conds[0],conds[1]),
                           alpha_min=alpha_min, alpha_max=alpha_max)
