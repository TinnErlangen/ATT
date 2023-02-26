from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

proc_dir = "../proc/"
df = pd.read_pickle("{}mu_dics_dpte.pickle".format(proc_dir))

df["DICS_MP_log"] = np.log(df["DICS_MP"].values)
df["DICS_M_log"] = np.log(df["DICS_M"].values)
df["DICS_P_log"] = np.log(df["DICS_P"].values)
df["DICS_A1_log"] = np.log(df["DICS_A1"].values)
df["DICS_V1_log"] = np.log(df["DICS_V1"].values)

# MP to A1
formula = "DICS_A1_log ~ dPTE_MPtoA1*C(Cond, Treatment('rest'))"
A1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
A1_model_fit = A1_model.fit()
print(A1_model_fit.summary())

# MP to V1
formula = "DICS_V1_log ~ dPTE_MPtoV1*C(Cond, Treatment('rest'))"
V1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
V1_model_fit = V1_model.fit()
print(V1_model_fit.summary())

title_key = {"rest":"Rest", "audio":"Audio", "visual":"Visual",
             "visselten":"Vis. w/distr."}
colors = ["gray", "tab:pink", "tab:purple", "tab:green",]

fig, axes = plt.subplots(2, 4, figsize=(30., 19.2))
for reg_idx, reg in enumerate(["A1", "V1"]):
    this_fit = A1_model_fit if reg == "A1" else V1_model_fit
    rest_intercept = this_fit.params["Intercept"]
    if this_fit.pvalues[f"dPTE_MPto{reg}"] < 0.05:
        rest_coef = this_fit.params[f"dPTE_MPto{reg}"]
    else:
        rest_coef = 0
    for cond_idx, cond in enumerate(["rest", "audio", "visual", "visselten"]):
        this_df = df.query(f"Cond=='{cond}'")
        sns.scatterplot(data=this_df, y=f"DICS_{reg}_log",
                    x=f"dPTE_MPto{reg}", color=colors[cond_idx], alpha=0.35,
                    ax=axes[reg_idx, cond_idx])

        coef = 0
        intercept = rest_intercept
        if cond != "rest":
            if this_fit.pvalues[f"C(Cond, Treatment('rest'))[T.{cond}]"] < 0.05:
                intercept += this_fit.params[f"C(Cond, Treatment('rest'))[T.{cond}]"]
            if this_fit.pvalues[f"dPTE_MPto{reg}:C(Cond, Treatment('rest'))[T.{cond}]"] < 0.05:
                coef = this_fit.params[f"dPTE_MPto{reg}:C(Cond, Treatment('rest'))[T.{cond}]"]
            axes[reg_idx, cond_idx].set_ylabel(None)
            axes[reg_idx, cond_idx].set_yticklabels([])
        else:
            axes[reg_idx, cond_idx].set_ylabel(f"Log Power {reg}", fontsize=34)

        axes[reg_idx, cond_idx].axline((0, intercept), slope=rest_coef+coef,
                                       color=colors[cond_idx], linewidth=5)
        axes[reg_idx, cond_idx].text(.99, 0.92,
                                     f"Est. intercept: {intercept:.2f}\n"
                                     f"Est. slope: {rest_coef+coef:.2f}",
                                     fontsize=18, horizontalalignment='right',
                                     transform=axes[reg_idx, cond_idx].transAxes)

        if reg_idx == 0:
            axes[reg_idx, cond_idx].set_xlabel(f"LH M/P\u2192LH {reg}",
                                               fontsize=34)
            axes[reg_idx, cond_idx].set_xticklabels([])
        else:
            axes[reg_idx, cond_idx].set_xlabel(f"dPTE-0.5\nLH M/P\u2192LH {reg}",
                                               fontsize=34)
        axes[reg_idx, cond_idx].set_ylim(-2., 2.)
        axes[reg_idx, cond_idx].set_xlim(-.125, .125)
        axes[reg_idx, cond_idx].tick_params(axis="both", which="major",
                                            labelsize=34)
        if reg_idx == 0:
            axes[reg_idx, cond_idx].set_title(f"{title_key[cond]}", fontsize=40,
                                              pad=30)

        plt.tight_layout()

plt.savefig("../images/fig_5b.png")
plt.savefig("../images/fig_5b.tif")


# # M to A1
# print("\n\nM to A1")
# formula = "DICS_M_log ~ dPTE_MtoA1*C(Cond, Treatment('rest'))"
# M_a_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# M_a_model_fit = M_a_model.fit()
# print(M_a_model_fit.summary())
# formula = "DICS_A1_log ~ dPTE_MtoA1*C(Cond, Treatment('rest'))"
# A1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# A1_model_fit = A1_model.fit()
# print(A1_model_fit.summary())
#
# # M to V1
# print("\n\nM to V1")
# formula = "DICS_M_log ~ dPTE_MtoV1*C(Cond, Treatment('rest'))"
# M_v_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# M_v_model_fit = M_v_model.fit()
# print(M_v_model_fit.summary())
# formula = "DICS_V1_log ~ dPTE_MtoV1*C(Cond, Treatment('rest'))"
# V1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# V1_model_fit = V1_model.fit()
# print(V1_model_fit.summary())
#
# # P to A1
# print("\n\nP to A1")
# formula = "DICS_P_log ~ dPTE_PtoA1*C(Cond, Treatment('rest'))"
# P_a_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# P_a_model_fit = P_a_model.fit()
# print(P_a_model_fit.summary())
# formula = "DICS_A1_log ~ dPTE_PtoA1*C(Cond, Treatment('rest'))"
# A1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# A1_model_fit = A1_model.fit()
# print(A1_model_fit.summary())
#
# # P to V1
# print("\n\nP to V1")
# formula = "DICS_P_log ~ dPTE_PtoV1*C(Cond, Treatment('rest'))"
# P_v_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# P_v_model_fit = P_v_model.fit()
# print(P_v_model_fit.summary())
# formula = "DICS_V1_log ~ dPTE_PtoV1*C(Cond, Treatment('rest'))"
# V1_model = MixedLM.from_formula(formula, df, groups=df["Subj"])
# V1_model_fit = V1_model.fit()
# print(V1_model_fit.summary())
