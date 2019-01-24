import mne
import pickle
import numpy as np

sub = "ATT_10"
proc_dir = "../proc/stcs/"
subjects_dir="/home/jeff/freesurfer/subjects"

filename = proc_dir+sub+"_clust"
with open(filename,"rb") as file:
    clu = pickle.load(file)
    vertices = [np.arange(4098),np.arange(4098)]
    res = mne.stats.summarize_clusters_stc(
           clu, p_thresh=0.001, tstep=1, tmin=0, subject='nc_'+sub, vertices=vertices)
    res.plot(subjects_dir=subjects_dir,hemi="both",time_viewer=True)
