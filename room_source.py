import mne
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# units in m
room_dims = (2.9,3.88,2.42) # half length, depth, height, if sensors in seated position
sensor_radius = 0.6

supine = 1
sens_cent = (0,0.14,-0.61) # l,d,h when supine
resolution = 10 # dipoles per m

# build up room source space
spacing = [np.linspace(-room_dims[x]/2-sens_cent[x],room_dims[x]/2-sens_cent[x],np.round(room_dims[x]*resolution).astype(int)) for x in range(3)]
rr = []
nn = []
shape = np.array([0,0,0])
for l in np.nditer(spacing[0]):
    for d in np.nditer(spacing[1]):
        for h in np.nditer(spacing[2]):
            if np.sqrt(l**2 + d**2 + h**2) < sensor_radius:
                continue
            rr.append(np.array([l,d,h]))
            nn.append(np.random.normal(0,0.2,3))

rr, nn = np.array(rr), np.array(nn)
if supine:
    rr_old = rr.copy()
    rr[:,1] = rr_old[:,2]
    rr[:,2] = rr_old[:,1]
    del rr_old

room = mne.setup_volume_source_space(pos=dict(rr=rr,nn=nn))
room[0]["shape"] = (len(spacing[0]),len(spacing[1]),len(spacing[2]))
room[0]["kind"] = "volume"
raw = mne.io.Raw("../proc/eegamp_noi2_1-raw.fif")
events = []
idx = 0
while idx < len(raw):
    events.append(np.array([idx,0,1]))
    idx += np.round(2*raw.info["sfreq"]).astype(int)
events = np.array(events)
epo = mne.Epochs(raw, events, tmin=0, tmax=2, preload=True)
sph_mod = mne.make_sphere_model(r0=(0,0,0),head_radius=None,relative_radii=(1),sigmas=(0))
fwd = mne.make_forward_solution(epo.info,None,room,sph_mod,n_jobs=4)
csd = mne.time_frequency.csd_morlet(epo,tmin=0,tmax=2,frequencies=[6.2,17,24,50],n_jobs=4)
filters = mne.beamformer.make_dics(epo.info, fwd, csd, reg=0.5, inversion="matrix")
stc, freqs = mne.beamformer.apply_dics_csd(csd, filters)
data = stc.data[:,3]
dataz = (data - np.mean(data))/np.std(data)
sort_idx = np.argsort(dataz)
sel_idx = sort_idx[-500:-1]
#sel_idx = sort_idx
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.scatter(rr[sel_idx,0], rr[sel_idx,1], rr[sel_idx,2], c=dataz[sel_idx], lw=0, s=20)
ax.scatter(sens_cent[0], sens_cent[1], sens_cent[2], s=80)
ax.set_xlim(-room_dims[0]/2,room_dims[0]/2)
ax.set_ylim(-room_dims[1]/2,room_dims[1]/2)
ax.set_zlim(-room_dims[2]/2,room_dims[2]/2)
plt.show()
