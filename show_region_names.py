import mne
import numpy as np
from mayavi import mlab
from surfer import Brain

proc_dir = "/home/jeff/ATT_dat/proc/"
conds = ["audio", "visual", "visselten"]
parc = "RegionGrowing_70"
freq = "alpha_1"
cluster_idx = 0
subjects_dir = "/home/jeff/freesurfer/subjects"

# get constraint info
net_names = ["rest-{}_{}_c{}".format(cond,freq,cluster_idx) for cond in conds]
net_nets = []
for net_n in net_names:
    net_nets += [tuple(x) for x in list(np.load("{}{}.npy".format(proc_dir,net_n)))]
net_nets = list(set(net_nets))
constrain_inds = tuple(zip(*net_nets))
constrain_regs = [list(x) for x in constrain_inds]

# leaving and arriving
constrain_regs = list(set(constrain_regs[0]+constrain_regs[1]))

labels = mne.read_labels_from_annot("fsaverage",parc)
figs = []
brains = []
constrain_regs = range(len(labels))
for cr in constrain_regs:
    figs.append(mlab.figure(labels[cr].name))
    brains.append(Brain('fsaverage', 'both', 'white',
                  subjects_dir=subjects_dir, figure=figs[-1], alpha=0.99))
    brains[-1].add_label(labels[cr])
