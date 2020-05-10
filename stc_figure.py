import mne
from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jeff/ATT_dat/proc/"
subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
subject="fsaverage"
stc_file = "stc_f_7-9Hz_rest_audio_tfce_A-lh.stc"
initial_time = 0

stc = mne.read_source_estimate("{proc}{file}".format(proc=proc_dir,file=stc_file))
if stc.data.min()<0:
    s_max = np.abs(stc.data).max()
    #s_max = 45
    s_mid = s_max/2
    clim = {"kind":"value","pos_lims":[0,s_mid,s_max]}
    colorbar = True
elif stc.data.max() == 1:
    clim = "auto"
    colorbar = False
else:
    s_max = stc.data.max()
    s_min = stc.data.min()
    s_mid = (s_max-s_min)/2 + s_min
    clim = {"kind":"value","lims":[s_min,s_mid,s_max]}
    colorbar = True

fig, axes = plt.subplots(2,2)

mfig = mlab.figure()
brain = stc.plot(hemi="lh",clim=clim, views=["lat"],figure=mfig,
                 subjects_dir=subjects_dir, subject=subject,
                 time_unit="s", initial_time=initial_time,
                 colorbar=colorbar)
axes[0][0].imshow(mlab.screenshot(mfig))
axes[0][0].axis("off")
axes[0][0].set_title("Left")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="lh",clim=clim, views=["med"],figure=mfig,
                 subjects_dir=subjects_dir, subject=subject,
                 time_unit="s", initial_time=initial_time,
                 colorbar=colorbar)
axes[1][0].imshow(mlab.screenshot(mfig))
axes[1][0].axis("off")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="rh",clim=clim, views=["lat"],figure=mfig,
                 subjects_dir=subjects_dir, subject=subject,
                 time_unit="s", initial_time=initial_time,
                 colorbar=colorbar)
axes[0][1].imshow(mlab.screenshot(mfig))
axes[0][1].axis("off")
axes[0][1].set_title("Right")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="rh",clim=clim, views=["med"], figure=mfig,
                 subjects_dir=subjects_dir, subject=subject,
                 time_unit="s", initial_time=initial_time,
                 colorbar=colorbar)
axes[1][1].imshow(mlab.screenshot(mfig))
axes[1][1].axis("off")
mlab.close()
