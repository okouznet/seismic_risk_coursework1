import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

def mag_max_table(reg, format='tex', file_prefix=""):
    mag_max = pd.DataFrame(np.arange(5.5, 9, 0.1))
    mag_max = sm.add_constant(mag_max)
    res_pred = reg.get_prediction(mag_max)
    mag_max["return_period"] = 1 / (np.exp(res_pred.summary_frame()['mean']))

    if format == 'csv':
        mag_max.to_csv(f"output/tables/{file_prefix}_mag_max.csv")
    if format == 'tex':
        mag_max.to_latex(f"output/tables/{file_prefix}_mag_max.tex")
    return mag_max

def richter_recurrence_rel(df):
    df["YEAR"] = df["DATE"].dt.year

    recurr_rel = df[["DATE", "DEPTH", "MAG"]].groupby("MAG")["DATE"].count().reset_index().sort_values("MAG",
                                                                                                       ascending=False)
    recurr_rel.columns = ["MAG", "n"]

    # recurr_rel = recurr_rel[recurr_rel.MAGNITUDE >= 3]
    recurr_rel["N"] = (recurr_rel["n"].cumsum()) / df.YEAR.nunique()
    recurr_rel["log_N"] = np.log(recurr_rel["N"])
    recurr_rel

    X = recurr_rel['MAG'].astype(float)
    X = sm.add_constant(X)
    Y = recurr_rel["log_N"]
    mod = sm.OLS(endog=Y, exog=X)
    res = mod.fit()

    return recurr_rel, res


def visualize_RN_rel(df, reg, _save=False, file_prefix=''):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(x='MAG', y='log_N', data=df, ax=ax)

    a = reg.params[0]
    b = reg.params[1]
    ax.text(s=f"log(N) = {round(a, 2)} - {round(b, 2) * -1}M", x=4.5, y=1.5)
    if _save:
        plt.savefig(f"output/figs/{file_prefix}_plot_attenuation.png", bbox_inches='tight')
    else:
        plt.show()