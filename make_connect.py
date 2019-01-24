import mne
import scipy.sparse

proc_dir = "../proc/"
subjects_dir = "/home/jeff/freesurfer/subjects/"
subjs = ["ATT_10"]

for sub in subjs:

    filename = proc_dir+"nc_"+sub+"-src.fif"

    src = mne.read_source_spaces(filename)
    cnx = mne.spatial_src_connectivity(src)
    scipy.sparse.save_npz(proc_dir+"nc_"+sub+"_cnx.npz",scipy.sparse.coo_matrix(cnx))
