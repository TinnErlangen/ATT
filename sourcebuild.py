import mne
import numpy as np

subjs = ["ATT_10"]
runs = ["rest","audio","visselten","visual","zaehlen"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
fwd_thresh = 0.35

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
        epo = mne.read_epochs("../proc/"+subject+"_"+run+"_ica-epo.fif")
        fwd = mne.make_forward_solution(epo.info, trans=trans, src=src, bem=bem,
        meg=True, mindist=5.0, n_jobs=2)
        mne.write_forward_solution("../proc/"+subject+"_"+run+"_ica-fwd.fif",
        fwd, overwrite=True)
        mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
        mag_map.data[mag_map.data < fwd_thresh] = 0
        l,r = mne.stc_to_label(mag_map, src=src, subjects_dir=subjects_dir)
        l.name = "l_sens_label"
        r.name = "r_sens_label"
        mne.write_label("../proc/"+subject+"_"+run+"_sens_label-lh.label",l)
        mne.write_label("../proc/"+subject+"_"+run+"_sens_label-rh.label",r)
