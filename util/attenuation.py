import statsmodels.api as sm
import util.catalogue_processing as catalogue_processing
import util.catalogue_declustering as catalogue_declustering
import util.catalogue_completeness as catalogue_completeness
import matplotlib.pyplot as plt

import util.metrics.summary_stats as summary_stats
import util.metrics.completeness_stats as completeness_stats
import util.metrics.decluster_stats as decluster_stats
import util.recurrence_relationships as recurrence_relationships
import util.metrics.metrics_util as metrics_util

from dotmap import DotMap
import config as config
import pandas as pd
import numpy as np
import geopandas as gpd



def get_return_period(df, res, pga_thresh, A, B):
    a = res.params['M']
    c1 = res.params['_I_soil']
    b = res.params['r']
    min_mag = (((np.log(pga_thresh)+np.log(df['r']) - b * df['r'] - c1*df['_I_soil']) )/a).max()
    N = np.power(10, A + B * min_mag)

    return N


def calculate_return_period(df, atten_data, recurr_data, pga_thresh):
    assert len(atten_data) == len(recurr_data)
    Ns = []
    for z in recurr_data:
        A = recurr_data[z].params['const']
        B = recurr_data[z].params['MAG']

        N = get_return_period(df, atten_data[z], pga_thresh, A, B)

        Ns.append(N)

    return N

def calculate_exceedance_probabilty(L, N):
    q = 1 - np.exp(-1 * L * N)
    return q


def get_pga_data():

    pga = pd.read_csv("data/noa/noa-pga.csv")
    pga.columns = ["ID", "DATE", "LAT", "LON", "MAG", "DEPTH", "PGA", "PGV", "IO"]
    pga["DATE"] = pd.Series(pd.to_datetime(pga['DATE'], utc=True)).dt.round("min")

    pga = gpd.GeoDataFrame(
        pga, geometry=gpd.points_from_xy(pga.LON, pga.LAT), crs="EPSG:4326"
    )
    stations = pd.read_csv("data/noa/noa-stations.csv")
    stations["Longitude"] = stations["Longitude"].str[:-3].astype(float)
    stations["Latitude"] = stations["Latitude"].str[:-3].astype(float)
    stations = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude), crs="EPSG:4326"
    )
    out = gpd.sjoin_nearest(pga.to_crs("EPSG:32634"), stations[["Station", "geometry"]].to_crs("EPSG:32634"),
                            distance_col="repi_hat", how="left")

    # re-convert into km
    out["repi_hat"] = out.repi_hat / 1000
    out = catalogue_processing.add_vs30(out)
    from shapely import wkt

    zones = pd.read_csv("data/seismic-zones.csv")
    zones['geometry'] = zones['WKT'].apply(wkt.loads)
    zones = gpd.GeoDataFrame(zones, crs='epsg:4326')[["name", "geometry"]]
    zones.columns = ["ZONE", "geometry"]
    out = gpd.sjoin_nearest(out.drop("index_right", axis=1).to_crs("EPSG:32634"), zones.to_crs("EPSG:32634"),
                            how="left")

    zones = out.ZONE
    zones[zones  == "Zone 3"] = "Zone 2"
    out["ZONE"] = zones
    return out

def plot_hazard_curve(df, _save=False, file_prefix="", pga_lim=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    if pga_lim is not None:
        df = df[df.pga < pga_lim]
    df.set_index("pga").plot(ax=ax)
    ax.set_ylabel("q")
    plt.legend()
    if _save is False:
        plt.show()
    else:
        plt.savefig(f"output/figs/{file_prefix}_hazard_curve.png", bbox_inches='tight')

    plt.close()

def plot_pga_vs_dist(reg_df, res, _save=False, file_prefix=""):
    a = res.params['M']
    c1 = res.params['_I_soil']
    b = res.params['r']
    fig, ax = plt.subplots(figsize=(10, 10))
    pga_hat = reg_df["M"] * a + b * reg_df['r'] - np.log(reg_df['r']) + c1 * reg_df['_I_soil']
    reg_df["PGA_HAT"] = pga_hat  # pga_hatnp.log((np.exp(res.fittedvalues ) ))
    reg_df["PGA_HAT1"] = pga_hat + 1
    reg_df["PGA_HAT2"] = pga_hat + 2
    reg_df["PGA_HAT-1"] = pga_hat - 1
    reg_df["PGA_HAT-2"] = pga_hat - 2

    plot_data = reg_df.sort_values("d")
    plot_data["D"] = pd.cut(plot_data['d'], bins=list(range(0, 80, 5)), labels=list(range(0, 80, 5))[:-1])
    # ,
    plot_data = plot_data.groupby("D")[["log_pga", "PGA_HAT", "PGA_HAT1", "PGA_HAT2", "PGA_HAT-1", "PGA_HAT-2"]].mean()
    plot_data.plot(ax=ax, ls="-", marker='.' )

    pga_hat = reg_df["M"] * a + b * reg_df['r'] - np.log(reg_df['r']) #+ c1 * reg_df['_I_soil']
    reg_df["PGA_HAT (Rock)"] = pga_hat  # pga_hatnp.log((np.exp(res.fittedvalues ) ))
    reg_df["PGA_HAT1 (Rock)"] = pga_hat + 1
    reg_df["PGA_HAT2 (Rock)"] = pga_hat + 2
    reg_df["PGA_HAT-1 (Rock)"] = pga_hat - 1
    reg_df["PGA_HAT-2 (Rock)"] = pga_hat - 2

    plot_data = reg_df.sort_values("d")
    plot_data["D"] = pd.cut(plot_data['d'], bins=list(range(0, 80, 5)), labels=list(range(0, 80, 5))[:-1])
    # ,


    plot_data.plot(ax=ax, ls="--", alpha = 0.7, marker="*")

    ax.set_ylabel("PGA")
    plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    if _save is False:
        plt.show()

    else:

        plt.savefig(f"output/figs/{file_prefix}_attenuation.png", bbox_inches='tight')
    plt.close()


def prep_reg_df(df):
    df = df.dropna(subset=['PGA', 'MAG', 'repi_hat', "VS30"])

    reg_df = pd.DataFrame()

    reg_df['log_pga'] = np.log(df["PGA"])
    reg_df['ZONE'] = df['ZONE']
    reg_df['M'] = df['MAG']

    reg_df["_I_rock"] = (df.VS30 >= 650).astype(int)
    reg_df["_I_soil"] = (df.VS30 < 650).astype(int)

    dist = df.repi_hat
    dist[dist < 4.] = 4.
    reg_df["d"] = dist
    reg_df["r"] = (reg_df.d ** 2 + df.DEPTH ** 2)

    return reg_df

def fit_pga(reg_df):
    #X = sm.add_constant(X)
    X = reg_df[["M", "r", "_I_soil"]]
    Y = reg_df['log_pga'] + np.log(reg_df["r"])
    mod = sm.OLS(endog=Y, exog=X)
    res = mod.fit()

    return res


