import mne
import numpy as np

subjs = ["ATT_10"]
runs = ["rest","audio","visselten","visual","zaehlen"]
runs = ["1"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/ATT_dat/proc/"

for sub in subjs:
    subject = "nc_"+sub
    trans = "../proc/"+subject+"-trans.fif"
    src = mne.setup_source_space(subject,surface="white",
                                 subjects_dir=subjects_dir,n_jobs=4)
    src.save("../proc/"+subject+"-src.fif", overwrite=True)
    bem_model = mne.make_bem_model(subject, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("../proc/"+subject+"-bem.fif",bem)
    src.plot()
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')
    src = mne.read_source_spaces("../proc/"+subject+"-src.fif")
    bem = mne.read_bem_solution("../proc/"+subject+"-bem.fif")
    for run in runs:
        raw = mne.io.Raw("{dir}nc_{sub}_{run}_hand_ica-raw.fif".format(
                         dir=proc_dir,sub=sub,run=run))
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=4)
        mne.write_forward_solution("{dir}nc_{sub}_{run}-fwd.fif".format(
                                   dir=proc_dir,sub=sub,run=run), fwd,
                                   overwrite=True)
