from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from cnx.cnx_utils import plot_rgba, make_brain_image
import mne

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

proc_dir = "../proc/"
df = pd.read_pickle("{}mu_dics_dpte_all.pickle".format(proc_dir))
df = df[df.Reg != "Broad_Motor"]
dics = df["DICS"].values
dics_log = np.log(dics)
df["DICS_log"] = dics_log
vmin, vmax = -1.4, 0.4
colmap = "Reds"
mplcol = MplColorHelper(colmap, vmin, vmax)

views = {"left":{"view":"lateral", "distance":625, "hemi":"lh"},
         "right":{"view":"lateral", "distance":625, "hemi":"rh"},
         "upper":{"view":"dorsal", "distance":650,
                  "focalpoint":(-.77, 3.88, -21.53)},
         "caudal":{"view":"caudal", "distance":650},
         "rostral":{"view":"rostral", "distance":650}
        }

formula = "DICS_log ~ C(Reg, Treatment('L8143-rh'))*Task"
model = MixedLM.from_formula(formula, df, groups=df['Subj'],
                             re_formula=None)
DICS_fit = model.fit(method=["powell", "lbfgs"])
print(DICS_fit.summary())
print(DICS_fit.aic)

labels = mne.read_labels_from_annot("fsaverage", "RegionGrowing_70")
vals = np.zeros(len(labels))
# overall
intercept = DICS_fit.params["Intercept"]
for idx, label in enumerate(labels):
    lab = label.name
    vals[idx] = intercept
    if lab != "L8143-rh":
        if DICS_fit.pvalues[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]"] < .05:
            reg_fx = DICS_fit.params[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]"]
            vals[idx] += reg_fx

cols = np.zeros((len(labels), 4), dtype=float)
for val_idx, val in enumerate(vals):
    col = mplcol.get_rgb(val)[:3]
    alpha = abs(val) / vmax
    cols[val_idx] = np.hstack((col, alpha))
brain = plot_rgba(cols, labels, "RegionGrowing_70", background=(1,1,1))
img = make_brain_image(views, brain)

# task
fx_task = DICS_fit.params["Task[T.task]"]
vals = np.zeros(len(labels))
for idx, label in enumerate(labels):
    lab = label.name
    vals[idx] = intercept + fx_task
    if lab != "L8143-rh":
        if DICS_fit.pvalues[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]:Task[T.task]"] < .05:
            reg_fx = DICS_fit.params[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]:Task[T.task]"]
            vals[idx] += reg_fx

cols = np.zeros((len(labels), 4), dtype=float)
for val_idx, val in enumerate(vals):
    col = mplcol.get_rgb(val)[:3]
    alpha = abs(val) / vmax
    cols[val_idx] = np.hstack((col, alpha))
brain_task = plot_rgba(cols, labels, "RegionGrowing_70", background=(1,1,1))
img_task = make_brain_image(views, brain_task)

fig, axes = plt.subplots(6, 1, figsize=(21.6, 21.6))
fig.suptitle("Log power, high alpha band", fontsize=70)
fig.text(0.05, 0.86, "A", fontsize=70)
fig.text(0.05, 0.7, "B", fontsize=70)
fig.text(0.05, 0.54, "C", fontsize=70)
fig.text(0.05, 0.38, "D", fontsize=70)
fig.text(0.05, 0.22, "E", fontsize=70)
axes[0].imshow(img)
axes[0].axis("off")
axes[1].imshow(img_task)
axes[1].axis("off")


# condition
formula = "DICS_log ~ C(Reg, Treatment('L8143-rh'))*C(Cond, Treatment('rest'))"
model = MixedLM.from_formula(formula, df, groups=df['Subj'],
                             re_formula=None)
DICS_fit = model.fit(method=["powell", "lbfgs"])
print(DICS_fit.summary())
print(DICS_fit.aic)

intercept = DICS_fit.params["Intercept"]
for cond_idx, cond in enumerate(["audio", "visual", "visselten"]):
    if DICS_fit.pvalues[f"C(Cond, Treatment('rest'))[T.{cond}]"] < .05:
        fx_task = DICS_fit.params[f"C(Cond, Treatment('rest'))[T.{cond}]"]
    else:
        fx_task = 0
    vals = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        lab = label.name
        vals[idx] = intercept + fx_task
        if lab != "L8143-rh":
            if DICS_fit.pvalues[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]:C(Cond, Treatment('rest'))[T.{cond}]"] < .05:
                reg_fx = DICS_fit.params[f"C(Reg, Treatment('L8143-rh'))[T.{lab}]:C(Cond, Treatment('rest'))[T.{cond}]"]
                vals[idx] += reg_fx

    cols = np.zeros((len(labels), 4), dtype=float)
    for val_idx, val in enumerate(vals):
        col = mplcol.get_rgb(val)[:3]
        alpha = abs(val) / vmax
        cols[val_idx] = np.hstack((col, alpha))
    brain_cond = plot_rgba(cols, labels, "RegionGrowing_70", background=(1,1,1))
    img_cond = make_brain_image(views, brain_cond)
    axes[2+cond_idx].imshow(img_cond)
    axes[2+cond_idx].axis("off")

norm = Normalize(vmin, vmax)
scalmap = cm.ScalarMappable(norm, colmap)
colbar_size = (img_cond.shape[1], img_cond.shape[0]/3)
colbar_size = np.array(colbar_size) / 100
col_fig, ax = plt.subplots(1,1, figsize=colbar_size)
colbar = plt.colorbar(scalmap, cax=ax, orientation="horizontal")
ax.tick_params(labelsize=70)
ax.set_xlabel("Estimated power", fontsize=70)
col_fig.tight_layout()
col_fig.canvas.draw()
mat = np.frombuffer(col_fig.canvas.tostring_rgb(), dtype=np.uint8)
mat = mat.reshape(col_fig.canvas.get_width_height()[::-1] + (3,))
axes[-1].imshow(mat)
axes[-1].axis("off")
fig.tight_layout()
