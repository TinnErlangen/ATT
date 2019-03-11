import mne
import numpy as np
from os import listdir
import re

proc_dir = "../proc/sims/"
subjects_dir = "/home/jeff/freesurfer/subjects/"
filelist = listdir(proc_dir)
fwd_thresh = 0.35

subjs = ["10","15","20","22","25"]

for sub in subjs:
    subject = "nc_ATT_"+sub
    trans = "{dir}nc_ATT_{sub}-trans.fif".format(dir=proc_dir,sub=sub)
    src = mne.setup_source_space(subject,surface="white",
                                 subjects_dir=subjects_dir,n_jobs=4,
                                 spacing="oct5")
    src.save("{dir}{sub}-src.fif".format(dir=proc_dir,sub=subject), overwrite=True)
    bem_model = mne.make_bem_model(subject, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{dir}{sub}-bem.fif".format(dir=proc_dir,sub=subject),bem)
    # src.plot()
    # mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')
    # src = mne.read_source_spaces("{dir}{sub}-src.fif".format(dir=proc_dir,sub=subjects))
    # bem = mne.read_bem_solution("{dir}{sub}-bem.fif".format(dir=proc_dir,sub=subjects))

    for filename in filelist:
        m = re.search("{sub}.*hand".format(sub=subject),filename)
        if m:
            infile = m.string
    raw = mne.io.Raw(proc_dir+infile)
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
    meg=True, mindist=5.0, n_jobs=4)
    mne.write_forward_solution(proc_dir+infile[:-8]+"-fwd.fif",fwd, overwrite=True)
