import matplotlib.pyplot as plt

def plot_magnitude_kde(df, _save=False, file_prefix=""):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "green", "purple", "black"]
    sources = df.data_source.unique()
    for i in range(len(sources)):

        df[df.data_source == sources[i]]['MAG'].plot.kde(ax=ax, label=sources[i], color=colors[i])

    df['MAG'].plot.kde(ax=ax, color="red", label="Entire Catalogue")
    plt.legend()
    ax.set_xlim(0, 9)
    ax.set_xlabel("Magnitude")
    ax.set_title("Distribution of magnitudes (by source)")
    if _save:
        plt.savefig(f"output/figs/{file_prefix}_plot_kde.png", bbox_inches='tight')
    else:
        plt.show()


def plot_earthquake_counts_over_time(df, _save=False, file_prefix=""):
    print("TEST")
    fig, ax = plt.subplots(figsize=(10, 6))
    df["YEAR"] = df.DATE.dt.year
    counts = df.groupby(["YEAR", "data_source"])["ID"].count().reset_index()
    plot_data = counts.pivot_table(index="YEAR", columns="data_source", values="ID")
    plot_data["Full Catalogue"] = plot_data.sum(axis=1)
    plot_data.plot(ax=ax, kind="line", lw=2, alpha=0.5)

    #ax2 = ax.twinx()
    #plot_data.plot(ax=ax2, kind='bar', width=3, alpha=0.3, zorder=0)
    # atalogue["YEAR"] = catalogue.DATE.dt.year

    plt.legend()
    # ax.set_xlim(0, 9)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Count of seismic events")
    plt.locator_params(axis='x', nbins=20)

    if _save:
        plt.savefig(f"output/figs/{file_prefix}_plot_time_dist.png", bbox_inches='tight')
    else:
        plt.show()

    # plt.close()


def summary_stats(df, file_prefix, format="csv"):

    if format == "csv":
        df["MAG", "DEPTH"].describe().to_csv(f"output/tables/{file_prefix}-summary_stats.tex")

    elif format == "tex":
        df["MAG", "DEPTH"].describe().to_latex(f"output/tables/{file_prefix}-summary_stats.tex")


def stats_by_column(df, column, file_prefix, format="csv"):
    if format == "csv":
        df.groupby(column)["MAG"].describe().to_csv(f"output/tables/{file_prefix}-{column}-stats.tex")
        df.groupby(column)["DATE"].describe().to_csv(f"output/tables/{file_prefix}-{column}-date-ranges.tex")

    elif format == "tex":
        df.groupby(column)["MAG"].describe().to_latex(f"output/tables/{file_prefix}-{column}-stats.tex")
        df.groupby(column)["DATE"].describe().to_latex(f"output/tables/{file_prefix}-{column}-date-ranges.tex")