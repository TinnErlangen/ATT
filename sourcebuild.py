import mne
import numpy as np
#from surfer import Brain
#from mayavi import mlab

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}
#FAO18, WKI71, BRA52 had a defective (?) MRI and fsaverage was used instead
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks

runs = ["audio","visselten","visual","zaehlen"]
#runs = ["audio","visselten","visual"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/ATT_dat/proc/"
thresh = 0.25
n_jobs = 8

spacing = "ico5"
sens =[]
for k,v in mri_key.items():
    trans = "../proc/"+k+"-trans.fif"
    src = mne.setup_source_space(k,surface="white", spacing=spacing,subjects_dir=subjects_dir,n_jobs=n_jobs)
    src.save("{}{}_{}-src.fif".format(proc_dir, v, spacing), overwrite=True)
    bem_model = mne.make_bem_model(k, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{}{}_{}-bem.fif".format(proc_dir, v, spacing),bem)
    src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir, v, spacing))
    bem = mne.read_bem_solution("{}{}_{}-bem.fif".format(proc_dir, v, spacing))
    for run in runs:
        epo = mne.read_epochs("{dir}nc_{sub}_{run}_4000Hz_hand-epo.fif".format(
                              dir=proc_dir,sub=v,run=run))
        fwd = mne.make_forward_solution(epo.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=n_jobs)
        mne.write_forward_solution("{dir}nc_{sub}_{run}_{spacing}-fwd.fif".format(
                                   dir=proc_dir,sub=v,run=run,spacing=spacing),
                                   fwd,overwrite=True)
    del src, bem, fwd
    fwds = []
    for run in runs:
        fwds.append(mne.read_forward_solution(
          "{dir}nc_{sub}_{run}_{spacing}-fwd.fif".format(dir=proc_dir, sub=v,
          run=run,spacing=spacing)))
    avg_fwd = mne.average_forward_solutions(fwds)
    del fwds
    mne.write_forward_solution("{dir}nc_{sub}_{spacing}-fwd.fif".format(
                               dir=proc_dir,sub=v,spacing=spacing), avg_fwd,
                               overwrite=True)
    avg_fwd = mne.read_forward_solution("{dir}nc_{sub}_{spacing}-fwd.fif".format(dir=proc_dir, sub=v, spacing=spacing))
    sen = mne.sensitivity_map(avg_fwd,ch_type="mag",mode="fixed")
    m_to_fs = mne.compute_source_morph(sen,subject_from=k,
                                       subject_to="fsaverage",
                                       spacing=int(spacing[-1]),
                                       subjects_dir=subjects_dir,
                                       smooth=None)
    sen = m_to_fs.apply(sen)
    sens.append(sen)
    sen.save("{dir}nc_{sub}_{sp}_sens".format(dir=proc_dir,sub=v,sp=spacing))

sen = sens[0].copy()
for s in sens[1:]:
    sen.data += s.data
sen.data /= len(sens)
sen.data[sen.data<thresh] = 0
sen.data[sen.data>=thresh] = 1
exclude = np.where(sen.data==0)[0]
np.save("{}fsaverage_{}_exclude.npy".format(proc_dir, spacing),exclude)
