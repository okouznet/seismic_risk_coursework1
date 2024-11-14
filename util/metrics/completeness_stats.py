import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import util.metrics.metrics_util as metrics_util
import util.recurrence_relationships as recurrence_relationships

# catalogue_clean = catalogue[catalogue.MAGNITUDE >= MAG_MIN]
def plot_magbin_counts(df, bins, _save=False, file_prefix=""):
    # df = df[df.MAG >= mag_min]
    # bins = list(range(int(mag_min), 9)) + [np.inf]

    labels = bins[:-1]
    df["M_group"] = pd.cut(df['MAG'], bins=bins, labels=labels)
    df["YEAR"] = df["DATE"].dt.year

    X = df.groupby(["YEAR", "M_group"])["MAG"].count().reset_index()
    X = X[X.YEAR <= 2020]
    X.pivot_table(index="YEAR", columns="M_group", values="MAG")

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = X.pivot_table(index="YEAR", columns="M_group", values="MAG")

    plot_data.plot(ax=ax)
    ax.set_title("Counts of seismic events by magnitude range over time")

    if _save:
        plt.savefig(f"output/figs/{file_prefix}_plot_completeness.png", bbox_inches='tight')

    else:
        plt.show()


def plot_init_completeness_assessment(df, _save=False, file_prefix=""):
    # plot overall
    recurr_rel, res = recurrence_relationships.richter_recurrence_rel(df)
    print(res.summary())

    fig, ax = plt.subplots(figsize=(10, 6))
    recurr_rel.plot.scatter(x="MAG", y="log_N", ax=ax, label="Entire catalogue")
    tmp = recurr_rel[recurr_rel.MAG >= 3]
    coef = np.polyfit(tmp.MAG, tmp.log_N, 1)
    poly1d_fn = np.poly1d(coef)
    tmp['Y'] = poly1d_fn(tmp.MAG)

    tmp.plot(x='MAG', y='Y', ax=ax, color="k",
             label="Gutenberg-Richter Law")  # '--k'=black dashed line, 'yo' = yellow circle marker

    sources = df.data_source.unique()
    colors = ['red', 'orange', 'green', 'purple']
    for i in range(len(sources)):
        recurr_rel, res = recurrence_relationships.richter_recurrence_rel(df[df.data_source == sources[i]])
        recurr_rel.plot.scatter(x="MAG", y="log_N", ax=ax, label=sources[i], color=colors[i])

    ax.axvline(x=3, color="black", alpha=0.5)
    ax.set_title("Estimate of Mc by catalogue source")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Log(N)")
    if _save is True:
        plt.savefig(f"output/figs/{file_prefix}_plot_mag_complete.png", bbox_inches='tight')
    else:
        plt.show()

    plt.close()
