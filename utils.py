import mne
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def get_pos(info):
    locs = []
    chs = info["chs"]
    for ch in chs:
        locs.append(ch["loc"][:3])
    locs = np.array(locs)
    return locs

def get_rot(info):
    rots = []
    chs = info["chs"]
    for ch in chs:
        rots.append(np.array(ch["loc"][3:]).reshape(3,3).T)
    return rots

def rots_vec(vec,rots):
    vecs = []
    for rot in rots:
        vecs.append(np.matmul(rot,vec))
    vecs = np.array(vecs)
    return vecs

plt.ion()
filename = "../proc/nc_ATT_19_5_hand-raw.fif"
raw = mne.io.Raw(filename,preload=True)
raw_ref = raw.copy().pick_types(meg=False,ref_meg=True)
raw_meg = raw.copy().pick_types(meg=True,ref_meg=False)
ref_pos = get_pos(raw_ref.info)
meg_pos = get_pos(raw_meg.info)
ref_rots = get_rot(raw_ref.info)
meg_rots = get_rot(raw_meg.info)
ref_vecs = rots_vec(np.ones(3)*0.03,ref_rots)
meg_vecs = rots_vec(np.ones(3)*0.008,meg_rots)

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.quiver(ref_pos[:,0],ref_pos[:,1],ref_pos[:,2],ref_vecs[:,0],ref_vecs[:,1],ref_vecs[:,2])
ax.quiver(meg_pos[:,0],meg_pos[:,1],meg_pos[:,2],meg_vecs[:,0],meg_vecs[:,1],meg_vecs[:,2])
plt.axis("off")
plt.show()
