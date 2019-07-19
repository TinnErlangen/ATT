import mne
import numpy as np
from mayavi import mlab

subject = "ATT_21"
proc_dir = "../proc/stcs/"
subjects_dir="/home/jeff/freesurfer/subjects"
runs = ["rest","audio","visselten","visual","zaehlen"]
clim = {"kind":"value","lims":[1e-26,3e-26,5.5e-26]}
t_mask = 25
fwd_thresh = 0.35
smoothing_steps = 3

#runs = ["rest"]
fig_idx = 0
for run in runs:
    stc_m = mne.read_source_estimate("{dir}nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=subject,b=run))
    stc_m.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=subject,
    hemi="both", clim=clim, smoothing_steps = smoothing_steps)
    mlab.title(run + " mean")
    fig_idx += 1

    stc_t = mne.read_source_estimate("{dir}nc_{a}_{b}_t-lh.stc".format(dir=proc_dir,a=subject,b=run))
    stc_t.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=subject,
    hemi="both", clim = {"kind":"value","lims":[0,45,50]}, smoothing_steps = smoothing_steps)
    mlab.title(run + " t")
    fig_idx += 1

    # make and display a t-mask of the man
    stc_tm = stc_m.copy()
    stc_tm.data = stc_m.data*(stc_t.data>t_mask).astype(np.float)
    stc_tm.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=subject,
    hemi="both", clim=clim, smoothing_steps = smoothing_steps)
    mlab.title(run + " mean, t-masked")
    fig_idx += 1

fwd = mne.read_forward_solution("../proc/nc_"+subject+"_"+run+"-fwd.fif")
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
mag_map.data[mag_map.data < fwd_thresh] = 0
mag_map.data[mag_map.data > fwd_thresh] = 1
mag_map.plot(subjects_dir=subjects_dir, clim=dict(kind="values",
lims=[0, 0, 1]), smoothing_steps = smoothing_steps)
mlab.title("forward sensitivity")
fig_idx += 1
