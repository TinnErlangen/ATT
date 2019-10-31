import mne
import numpy as np
from mayavi import mlab

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "KER27":"ATT_30","ATT_27_fsaverage":"ATT_27","DEN59":"ATT_26",
           "WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31","Mun79":"ATT_35",
           "ATT_37_fsaverage":"ATT_37","EAM67":"ATT_32","ATT_24_fsaverage":"ATT_24",
           "TGH11":"ATT_14","FIN23":"ATT_17","GIZ04":"ATT_13","BAI97":"ATT_22",
           "WAL70":"ATT_33","ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}

subject = "ATT_17"
proc_dir = "../proc/stcs/"
subjects_dir="/home/jeff/freesurfer/subjects"
runs = ["rest","audio","visselten","visual","zaehlen"]
lower = 3e-27
upper = 3e-26
clim = {"kind":"value","lims":[lower,(upper-lower)/2,upper]}
t_mask = 25
fwd_thresh = 0.35
smoothing_steps = 3

#runs = ["rest"]
fig_idx = 0
for run in runs:
    stc_m = mne.read_source_estimate("{dir}nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=subject,b=run))
    stc_m.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=sub_key[subject],
               hemi="both", clim=clim, smoothing_steps = smoothing_steps)
    mlab.title(run + " mean")
    fig_idx += 1

    # stc_t = mne.read_source_estimate("{dir}nc_{a}_{b}_t-lh.stc".format(dir=proc_dir,a=subject,b=run))
    # stc_t.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=sub_key[subject],
    # hemi="both", clim = {"kind":"value","lims":[0,45,50]}, smoothing_steps = smoothing_steps)
    # mlab.title(run + " t")
    # fig_idx += 1

    # make and display a t-mask of the man
    # stc_tm = stc_m.copy()
    # stc_tm.data = stc_m.data*(stc_t.data>t_mask).astype(np.float)
    # stc_tm.plot(figure=fig_idx,subjects_dir=subjects_dir,subject=subject,
    # hemi="both", clim=clim, smoothing_steps = smoothing_steps)
    # mlab.title(run + " mean, t-masked")
    # fig_idx += 1

# fwd = mne.read_forward_solution("../proc/nc_"+subject+"_"+run+"-fwd.fif")
# mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
# mag_map.data[mag_map.data < fwd_thresh] = 0
# mag_map.data[mag_map.data > fwd_thresh] = 1
# mag_map.plot(subjects_dir=subjects_dir, clim=dict(kind="values",
# lims=[0, 0, 1]), smoothing_steps = smoothing_steps)
# mlab.title("forward sensitivity")
# fig_idx += 1
