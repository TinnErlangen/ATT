import mne
from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jeff/ATT_dat/proc/"
subjects_dir = "/home/jeff/freesurfer/subjects/"
subject="fsaverage"
stc_file = "stc_f_7-12Hz_visual_visselten_tfce-lh.stc"

stc = mne.read_source_estimate("{proc}{file}".format(proc=proc_dir,file=stc_file))
s_max = stc.data.max()
s_mid = s_max/2

fig, axes = plt.subplots(2,2)

mfig = mlab.figure()
brain = stc.plot(hemi="lh",clim={"kind":"value","lims":[0,s_mid,s_max]},
                 views=["lat"],figure=mfig,subjects_dir=subjects_dir,
                 subject=subject)
axes[0][0].imshow(mlab.screenshot(mfig))
axes[0][0].axis("off")
axes[0][0].set_title("Left")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="lh",clim={"kind":"value","lims":[0,s_mid,s_max]},
                 views=["med"],figure=mfig,subjects_dir=subjects_dir,
                 subject=subject)
axes[1][0].imshow(mlab.screenshot(mfig))
axes[1][0].axis("off")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="rh",clim={"kind":"value","lims":[0,s_mid,s_max]},
                 views=["lat"],figure=mfig,subjects_dir=subjects_dir,
                 subject=subject)
axes[0][1].imshow(mlab.screenshot(mfig))
axes[0][1].axis("off")
axes[0][1].set_title("Right")
mlab.close()

mfig = mlab.figure()
brain = stc.plot(hemi="rh",clim={"kind":"value","lims":[0,s_mid,s_max]},
                 views=["med"],figure=mfig,subjects_dir=subjects_dir,
                 subject=subject)
axes[1][1].imshow(mlab.screenshot(mfig))
axes[1][1].axis("off")
mlab.close()
