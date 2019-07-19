import mne
import numpy as np

subjs = ["ATT_21"]
runs = ["rest","audio","visselten","visual","zaehlen"]
#runs = ["rest"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/ATT_dat/proc/"
thresh = 0.5

for sub in subjs:
    subject = sub
    trans = "../proc/"+subject+"-trans.fif"

    # src = mne.setup_source_space(subject,surface="white",
    #                              subjects_dir=subjects_dir,n_jobs=4)
    # src.save("../proc/"+subject+"-src.fif", overwrite=True)
    # bem_model = mne.make_bem_model(subject, subjects_dir=subjects_dir)
    # bem = mne.make_bem_solution(bem_model)
    # mne.write_bem_solution("../proc/"+subject+"-bem.fif",bem)
    # src.plot()
    # mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')

    src = mne.read_source_spaces("../proc/"+subject+"-src.fif")
    bem = mne.read_bem_solution("../proc/"+subject+"-bem.fif")
    sens = []
    for run in runs:
        epo = mne.read_epochs("{dir}nc_{sub}_{run}_hand-epo.fif".format(
                              dir=proc_dir,sub=sub,run=run))
        fwd = mne.make_forward_solution(epo.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=8)
        mne.write_forward_solution("{dir}nc_{sub}_{run}-fwd.fif".format(
                                   dir=proc_dir,sub=sub,run=run), fwd,
                                   overwrite=True)
        sens.append(mne.sensitivity_map(fwd,ch_type="mag",mode="fixed"))

    avg_sens = sum(sens)/len(sens)
    avg_sens.data[avg_sens.data<thresh] = 0
    sens_lh, sens_rh = mne.stc_to_label(avg_sens,src=src,subjects_dir=subjects_dir,smooth=True)
    sens_lh.save("{dir}nc_{sub}".format(dir=proc_dir,sub=sub,run=run))
    sens_rh.save("{dir}nc_{sub}".format(dir=proc_dir,sub=sub,run=run))
