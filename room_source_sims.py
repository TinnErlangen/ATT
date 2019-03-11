import mne
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
import re

proc_dir = "../proc/sims/"
filelist = listdir(proc_dir)
subjects_dir = "/home/jeff/freesurfer/subjects/"

# # units in m
# room_dims = (2.94,3.95,2.42) # length, depth, height
# sensor_radius = 0.6
#
# supine = 1
#
# sens_cent = (1.47,2.22,0.96) # l,d,h when supine
# resolution = 4 # dipoles per m
#
# # build up room source space
# spacing = [np.linspace(-sens_cent[x],room_dims[x]-sens_cent[x],np.round(room_dims[x]*resolution).astype(int)) for x in range(3)]
# rr = []
# nn = []
# for l in np.nditer(spacing[0]):
#     for d in np.nditer(spacing[1]):
#         for h in np.nditer(spacing[2]):
#             if np.sqrt(l**2 + d**2 + h**2) < sensor_radius:
#                 continue
#             rr.append(np.array([l,d,h]))
#             nn.append(np.random.normal(0,0.2,3))
#
# rr, nn = np.array(rr), np.array(nn)
# if supine:
#     rr_old = rr.copy()
#     rr[:,1] = rr_old[:,2]
#     rr[:,2] = rr_old[:,1]
#     del rr_old

subjs = ["10","15","20","22","25"]
for sub in subjs:
    for filename in filelist:
        m = re.search("{sub}.*hand.*raw".format(sub=sub),filename)
        if m:
            infile = m.string
    raw = mne.io.Raw(proc_dir+infile)
    rr = mne.surface._get_ico_surface(2)['rr']
    rr *= 2.83
    nn = np.random.normal(0,0.2,rr.shape)
    src = mne.setup_volume_source_space(pos=dict(rr=rr,nn=nn))
    bem = mne.make_sphere_model("auto",head_radius=None,info=raw.info)
    fwd = mne.make_forward_solution(raw.info,None,src,bem,n_jobs=4)

    fig = plt.figure()
    ax = fig.gca(projection="3d")


    # ax.scatter(rr[sel_idx,0], rr[sel_idx,1], rr[sel_idx,2], c=dataz[sel_idx], lw=0, s=20)
    # ax.scatter(sens_cent[0], sens_cent[1], sens_cent[2], s=80)
    # ax.set_xlim(-room_dims[0]/2,room_dims[0]/2)
    # ax.set_ylim(-room_dims[1]/2,room_dims[1]/2)
    # ax.set_zlim(-room_dims[2]/2,room_dims[2]/2)
    # plt.show()
