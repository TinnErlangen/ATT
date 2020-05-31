from mne.stats import f_mway_rm, f_threshold_mway_rm, fdr_correction
from scipy.stats import sem
from cnx_utils import load_sparse, plot_directed_cnx
import numpy as np
import mne
import matplotlib.pyplot as plt
from mayavi import mlab
import pickle
plt.ion()


proc_dir = "/home/jeff/ATT_dat/proc/"
mat_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
freq = "alpha_1"
conds = ["audio", "visual"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
factor_levels = [len(conds), len(wavs)]
effects = ["A","B","A:B"]
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
effect_idx = 0
p_thresh = 0.05
cluster_idx = 0
f_thresh = f_threshold_mway_rm(len(subjs),factor_levels,effects=effects,
                               pvalue=p_thresh)
regions = ["L2340_L1933-lh", "L10017-lh", "L2340-lh", "L7755-lh", "L5106_L2688-lh",
           "L2340_L1933-rh", "L10017-rh", "L2340-rh", "L5106_L2688-rh"]
sub_labels = []
sub_labels_inds = []
for reg in regions:
    for lab_idx,lab in enumerate(labels):
        if reg == lab.name:
            sub_labels.append(lab)
            sub_labels_inds.append(lab_idx)

# get constraint info
net_names = ["rest-{}_{}_c{}".format(cond,freq,cluster_idx) for cond in ["audio","visual","visselten"]]
net_nets = []
for net_n in net_names:
    net_nets += [tuple(x) for x in list(np.load("{}{}.npy".format(proc_dir,net_n)))]
net_nets = list(set(net_nets))
constrain_inds = tuple(zip(*net_nets))
constrain_inds = [np.array(x) for x in constrain_inds]

# look at every region in the a priori network, instead of hand-defined...
sub_labels_inds = list(set(constrain_inds[0]))
sub_labels = [labels[l] for l in sub_labels_inds]

dPTEs = [[] for c in conds for w in wavs]
for sub in subjs:
    idx = 0
    for cond in conds:
        for wav in wavs:
            dPTE = load_sparse("{}nc_{}_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                  cond, wav,
                                                                  freq))
            dPTEs[idx].append(dPTE)
            idx += 1

dPTEs = [[d.mean(axis=0) for d in dpte] for dpte in dPTEs]
dPTEs = np.array(dPTEs)
dPTEs = np.swapaxes(dPTEs,0,1)

figs = []
brains = []
sig_labels = []
for sl_ind, sl in zip(sub_labels_inds, sub_labels):
    print(sl.name)
    dests = constrain_inds[1][np.where(constrain_inds[0]==sl_ind)]
    origins = np.ones(len(dests),dtype=int)*sl_ind
    # get only data under constraints
    this_dPTE = np.zeros(dPTEs.shape)
    this_dPTE[...,origins,dests] = dPTEs[...,origins,dests]
    # move centre from 0.5 to 0
    this_dPTE[this_dPTE!=0] -= 0.5
    # vectorise
    vec_dPTE = this_dPTE[...,origins,dests]
    f_vals, p_vals = f_mway_rm(vec_dPTE, factor_levels, effects=effects)
    print("Individual cnx: {}".format(p_vals[0,]))
    f_val, p_val = f_mway_rm(vec_dPTE.mean(axis=2,keepdims=True),
                               factor_levels, effects=effects)
    print("All outgoing cnx: {}".format(p_val[0,]))
    if np.any(p_vals[0,]<0.05):
        sig_inds = p_vals[0,]<0.05
        origins = origins[sig_inds]
        dests = dests[sig_inds]
        disp_dPTE = np.zeros(this_dPTE.shape[-2:])
        disp_dPTE[origins,dests] = this_dPTE[:,:,origins,dests].mean(axis=0).mean(axis=0)
        figs.append(mlab.figure(sl.name))
        brains.append(plot_directed_cnx(disp_dPTE,labels,parc,fig=figs[-1],
                      uniform_weight=True))
        for p_idx in np.nditer(np.where(p_vals[0]<0.05)[0]):
            cond_avgs = [vec_dPTE[:,idx:idx+4,p_idx] for idx in range(0,len(conds)*4,4)]
            cond_avgs = np.array(cond_avgs).mean(axis=-1)
            plt.figure()
            plt.bar(np.arange(len(conds)), np.abs(cond_avgs.mean(axis=1)), yerr=sem(cond_avgs, axis=1))
            plt.title(sl.name)
        sig_labels.append(sl)
with open("{}{}_apriori".format(proc_dir,parc), "wb") as f:
    pickle.dump(sig_labels,f)
