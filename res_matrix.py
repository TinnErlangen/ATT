import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.beamformer import read_beamformer, apply_dics_csd
from mne.io.pick import pick_channels_forward, pick_info, pick_channels
from mne.time_frequency import CrossSpectralDensity as CSD, csd_morlet
from mne.time_frequency.csd import _sym_mat_to_vector

plt.ion()

proc_dir = "/home/jeff/hdd/cora/leakage/"
band = "alpha"

filters = read_beamformer("{}nc_NEM_36_limb_mix_{}-dics.h5".format(proc_dir,band))
fwd = mne.read_forward_solution("{}nc_NEM_36_limb_mix_exp-fwd.fif".format(proc_dir))
epo = mne.read_epochs("{}nc_NEM_36_exp-epo.fif".format(proc_dir))
info = epo.info

# get leadfield from forward, using only good channels
ch_names = [c for c in filters["ch_names"] if (c not in info["bads"])]
fwd = pick_channels_forward(fwd, ch_names, ordered=True)
leadfield = fwd['sol']['data']

# get inverse matrix from DICS filter
invmat = np.abs(filters["weights"][0])

# resolution matrix
resmat = invmat.dot(leadfield)
