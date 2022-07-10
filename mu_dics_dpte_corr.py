from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

proc_dir = "../proc/"
df = pd.read_pickle("{}mu_dics_dpte.pickle".format(proc_dir))
dics = df["DICS"].values
dics_log = np.log(dics)
df["DICS_log"] = dics_log

formula = "dPTE ~ C(Cond, Treatment('rest'))"
model = MixedLM.from_formula(formula, df, groups=df["Subj"])
dPTE_cond_fit = model.fit(reml=True)
print(dPTE_cond_fit.summary())

formula = "DICS_log ~ C(Cond, Treatment('rest'))"
model = MixedLM.from_formula(formula, df, groups=df["Subj"])
DICS_cond_fit = model.fit(reml=True)
print(DICS_cond_fit.summary())

formula = "dPTE ~ DICS_log*C(Cond, Treatment('rest'))"
re_formula = "~DICS*C(Cond, Treatment('rest'))"
#re_formula = None
model = MixedLM.from_formula(formula, df, groups=df["Subj"],
                             re_formula=re_formula)
dPTE_DICS_fit = model.fit(reml=False)
print(dPTE_DICS_fit.summary())
print(dPTE_DICS_fit.aic)
model = MixedLM.from_formula(formula, df, groups=df["Subj"],
                             re_formula=None)
dPTE_DICS_fit = model.fit(reml=False)
print(dPTE_DICS_fit.summary())
print(dPTE_DICS_fit.aic)


formula = "DICS_log ~ dPTE*C(Cond, Treatment('rest'))"
re_formula = "~dPTE*C(Cond, Treatment('rest'))"
model = MixedLM.from_formula(formula, df, groups=df["Subj"],
                             re_formula=re_formula)
DICS_dPTE_fit = model.fit(reml=True)
print(DICS_dPTE_fit.summary())

# get data with subject variance factored out
formula = "dPTE ~ 1"
model = MixedLM.from_formula(formula, df, groups=df["Subj"])
dPTE_null_fit = model.fit(reml=True)

resids = dPTE_null_fit.resid
df["dPTE_resid"] = resids

fig, axes = plt.subplots(4, 6, figsize=(38.4, 28.6))
axes = [ax for axe in axes for ax in axe]
subjs = list(df["Subj"].unique())
for idx, (subj, ax) in enumerate(zip(subjs, axes)):
    sub_inds = df["Subj"] == subj
    sns.regplot(data=df[sub_inds], x="DICS_log", y="dPTE",
                color="blue", scatter_kws={"alpha":0.1}, ax=ax)
    ax.tick_params(axis="both", labelsize=12)
    if idx > 17:
        ax.set_xlabel("Log DICS", fontsize=20)
    else:
        ax.set_xlabel("")
    if idx % 6:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("dPTE", fontsize=20)
    ax.set_title("Participant {}".format(idx+1), fontsize=16)
plt.suptitle("High alpha motor/parietal DICS-dPTE relationship by participant", fontsize=48)
plt.savefig("../images/fig_s4.png")
plt.savefig("../images/fig_s4.tif")

# overall
fig, ax = plt.subplots(1, figsize=(19.2, 12))
sns.regplot(data=df, x="DICS_log", y="dPTE", color="blue",
            scatter_kws={"alpha":0.1})
ax.set_xlabel("Log DICS", fontsize=20)
ax.set_ylabel("dPTE", fontsize=20)
ax.set_ylim(-0.12, 0.12)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.set_title("High alpha motor/parietal DICS-dPTE relationship", fontsize=40)
plt.savefig("../images/fig_5.png")
plt.savefig("../images/fig_5.tif")
