import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import contextily as cx
from dotmap import DotMap
import config as config
import statsmodels.api as sm
import glob
import abc

from econtools import load_or_build


BUFFER_KM = 200
WORK_DIR = "/Users/skouznet/Documents/School/UCL/CEGE0033 Seismic Risk Assessment/Coursework1/CourseWork1/data"

import pygmt as pygmt


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
    out = add_vs30(out)
    return out


def add_vs30(df):
    region = [config.BUFFER_RANGE.MIN_LON, config.BUFFER_RANGE.MAX_LON, config.BUFFER_RANGE.MIN_LAT,
              config.BUFFER_RANGE.MAX_LAT]
    vs30 = pygmt.grd2xyz("data/global_vs30.grd", output_type='pandas', region=region)

    vs30.columns = ["LON", "LAT", "VS30"]
    vs30['LAT'] = vs30['LAT'].apply(lambda x: round(x, 2))
    vs30['LON'] = vs30['LON'].apply(lambda x: round(x, 2))

    vs30 = vs30.drop_duplicates().groupby(["LAT", "LON"])['VS30'].max().reset_index()

    print(len(vs30))

    df["LAT_OLD"] = df.LAT
    df["LON_OLD"] = df.LON
    df['LAT'] = df['LAT'].apply(lambda x: round(x, 2))
    df['LON'] = df['LON'].apply(lambda x: round(x, 2))

    out = pd.merge(df, vs30, on=["LAT", "LON"], how="inner")

    out["LAT"] = out.LAT_OLD
    out["LON"] = out.LON_OLD

    out = out.drop(["LAT_OLD", "LON_OLD"], axis=1)
    return out


def calculate_source_site_distance(df):
    stations = pd.read_csv("data/noa/noa-stations.csv")
    # [dms2dd(x) for x in stations.Latitude]

    stations["Longitude"] = stations["Longitude"].str[:-3].astype(float)
    stations["Latitude"] = stations["Latitude"].str[:-3].astype(float)
    stations = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude), crs="EPSG:4326"
    )

    out = gpd.sjoin_nearest(df.to_crs("EPSG:32634"), stations[["Station", "geometry"]].to_crs("EPSG:32634"),
                            distance_col="repi_hat", how="left")

    # re-convert into km
    out["repi_hat"] = out.repi_hat / 1000

    return out


def remove_duplicate_lat_long(df):
    def remove_dup_lat(grouped_df):
        last_row = grouped_df.iloc[-1]['LAT']
        grouped_df_filtered = grouped_df[(grouped_df['LAT'] - last_row).abs() < 0.5]
        return grouped_df_filtered

    def remove_dup_lon(grouped_df):
        last_row = grouped_df.iloc[-1]['LON']
        grouped_df_filtered = grouped_df[(grouped_df['LON'] - last_row).abs() < 0.5]
        return grouped_df_filtered

    r_df = df.groupby('cluster').apply(remove_dup_lat).reset_index(drop=True)
    r_df = r_df.groupby('cluster').apply(remove_dup_lon).reset_index(drop=True)
    return r_df


def clean_catalogue(df, _save_dups=False):
    # drop incomplete events
    df = df.dropna(subset=["DATE", "LAT", "LON", "MAG"])
    print("(deduplicate_catalogue)", "incomplete events", len(df), df.DATE.nunique())

    df["MAG"] = pd.to_numeric(df["MAG"])
    df['LAT'] = df['LAT'].apply(lambda x: round(x, 3))
    df['LON'] = df['LON'].apply(lambda x: round(x, 3))
    df['MAG'] = df['MAG'].apply(lambda x: round(x, 1))
    df = df[df.MAG >= 0]

    df["DATE"] = pd.Series(pd.to_datetime(df['DATE'], utc=True)).dt.round("min")

    df = df.drop_duplicates()

    return df


def format_catalogue_usgs():
    csvs = glob.glob(f'{WORK_DIR}/usgs/usgs_*.csv')
    dfs = []
    for csv in csvs:
        df = pd.read_csv(csv)

        df_clean = df[['id', 'time', 'latitude', 'longitude', 'depth', 'mag', 'magType']]
        df_clean.columns = ["ID", "DATE", "LAT", "LON", "DEPTH", "MAG", "MAGTYPE"]
        dfs.append(df_clean)

    catalogue = pd.concat(dfs)

    catalogue['MAG'] = pd.to_numeric(catalogue['MAG'], errors='coerce')
    catalogue['DATE'] = pd.to_datetime(catalogue['DATE'], utc=True, format="%Y-%m-%dT%H:%M:%S.%fZ")

    # within-catalogue deduplications
    print("usgs", "init", len(catalogue), catalogue.DATE.nunique())
    catalogue = clean_catalogue(catalogue, _save_dups=False)
    print("usgs", "post", len(catalogue), catalogue.DATE.nunique())
    return catalogue


def format_catalogue_noa():
    noa_post2000 = pd.read_csv(f"{WORK_DIR}/noa/EarthquakeCatalogueNOAFull.txt", sep='\s+', header=None, engine='python',
                               encoding='utf-8')

    noa_pre2000 = pd.read_csv(f"{WORK_DIR}/noa/EarthquakeCatalogueNOA_pre2000.txt", sep='\s+', header=None, engine='python',
                              encoding='utf-8')

    noa = pd.concat([noa_post2000, noa_pre2000])

    noa["DATE"] = pd.to_datetime(dict(year=pd.to_datetime(noa[0], format='%Y').dt.year,
                                      month=pd.to_datetime(noa[1], format='%b').dt.month,
                                      day=pd.to_datetime(noa[2], format='%d').dt.day,  # ,
                                      hour=noa[3],
                                      minute=noa[4]
                                      # second=noa[5]
                                      ),
                                 )

    # noa = noa.drop([0,1,2,3,4,5], axis=1)
    noa.columns = ["YEAR", "MONTH", "DAY", "HR", "MIN", "SEC", "LAT", "LON", "DEPTH", "MAG", "DATE"]
    noa["ID"] = list(range(len(noa)))

    noa = noa.fillna(-999)
    # noa = pd.read_csv("EarthquakeCatalogueNOAFull.csv")

    catalogue = noa[["ID", "DATE", "LAT", "LON", "DEPTH", "MAG"]].drop_duplicates()
    catalogue["MAGTYPE"] = "ml"
    catalogue['MAG'] = catalogue['MAG'].astype(str).str.replace('*', '').str.replace('ms', '')

    print("noa", "init", len(catalogue), catalogue.DATE.nunique())
    catalogue = clean_catalogue(catalogue, _save_dups=False)
    print("noa", "post", len(catalogue), catalogue.DATE.nunique())

    return catalogue


def format_catalogue_ics():
    ics_supp = pd.read_csv(f"{WORK_DIR}/isc-gem/isc-gem-suppl_clean.csv")
    ics_supp.columns = [x.strip(' ') for x in ics_supp.columns]
    ics_supp['date'] = pd.to_datetime([x.strip(' ') for x in ics_supp['date']], utc=True)

    ics_main = pd.read_csv(f"{WORK_DIR}/isc-gem/isc-gem-cat_clean.csv")
    ics_main.columns = [x.strip(' ') for x in ics_main.columns]
    ics_main['date'] = pd.to_datetime([x.strip(' ') for x in ics_main['date']], utc=True)

    catalogue = pd.concat([ics_main, ics_supp])
    catalogue['mw'] = pd.to_numeric(catalogue['mw'], errors='coerce')

    catalogue = catalogue[["eventid", "date", "lat", "lon", "depth", "mw"]]
    catalogue.columns = ["ID", "DATE", "LAT", "LON", "DEPTH", "MAG"]
    catalogue['DATE'] = pd.to_datetime(catalogue['DATE'], utc=True, format="%Y-%m-%d %H:%M:%S.%f")
    catalogue["MAGTYPE"] = "mw"

    print("ics", "init", len(catalogue), catalogue.DATE.nunique())
    catalogue = clean_catalogue(catalogue, _save_dups=False)
    print("ics", "post", len(catalogue), catalogue.DATE.nunique())
    return catalogue


def format_catalogue_emdat():
    emdat = pd.read_csv(f"{WORK_DIR}/emdat/emdat.csv")

    emdat['DATE'] = pd.to_datetime(dict(year=pd.to_datetime(emdat['Start Year'], format='%Y').dt.year,
                                        month=pd.to_datetime(emdat['Start Month'], format='%m').dt.month,
                                        day=pd.to_datetime(emdat['Start Day'], format='%d').dt.day
                                        ), utc=True)

    emdat["DEPTH"] = None
    catalogue = emdat[['DisNo.', "DATE", "Latitude", "Longitude", 'DEPTH', "Magnitude"]]
    catalogue.columns = ["ID", "DATE", "LAT", "LON", "DEPTH", "MAG"]
    catalogue["MAGTYPE"] = "mw"
    print("emdat", "init", len(catalogue), catalogue.DATE.nunique())

    catalogue = clean_catalogue(catalogue, _save_dups=False)
    print("emdat", "post", len(catalogue), catalogue.DATE.nunique())

    return catalogue


def deduplicate_combined_catalogue(df):
    # Deduplicate across whole catalog
    df["MAGTYPE"] = pd.Categorical(df.MAGTYPE,
                                   categories=['m', 'mb', 'md', 'ml', 'ms', 'mwb', 'mwc', 'mwr', 'mww', 'mw'],
                                   ordered=True)
    df["data_source"] = pd.Categorical(df.data_source, categories=['emdat', 'usgs', 'noa', 'ics'], ordered=True)
    test = df.groupby(["DATE", "LAT", "LON"])[["MAGTYPE", "data_source"]].max().reset_index()
    df = pd.merge(df, test, on=["DATE", "LAT", "LON", "MAGTYPE", "data_source"], how="inner")

    test = df.groupby("DATE")["ID"].unique().reset_index()
    test.columns = ["DATE", "ids"]

    test['n'] = test['ids'].apply(lambda x: len(x))

    dups = test[test.n > 1]

    dups["cluster"] = list(range(len(dups)))

    dups = pd.merge(df, dups, on=["DATE"], how="inner")

    dups = remove_duplicate_lat_long(dups)

    non_dups = df[~df.ID.isin(dups.ID.unique())]

    dups = pd.merge(dups, dups.groupby("cluster")["MAGTYPE"].max().reset_index(), on=["cluster", "MAGTYPE"],
                    how="inner")
    dups = pd.merge(dups, dups.groupby("cluster")["data_source"].max().reset_index(), on=["cluster", "data_source"],
                    how="inner")
    dups = dups[non_dups.columns.values]
    df = pd.concat([dups, non_dups])

    return df


@load_or_build('data/catalogue.csv')
def combine_catalogues():
    usgs = format_catalogue_usgs()
    usgs['data_source'] = "usgs"
    noa = format_catalogue_noa()
    noa['data_source'] = "noa"
    ics = format_catalogue_ics()
    ics['data_source'] = "ics"

    emdat = format_catalogue_emdat()
    emdat['data_source'] = 'emdat'

    df = pd.concat([usgs, noa, ics, emdat])  # emdat

    quakes = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LON, df.LAT), crs="EPSG:4326"
    )

    quakes_in_region =  quakes[(quakes.LAT.between(config.BUFFER_RANGE.MIN_LAT, config.BUFFER_RANGE.MAX_LAT)) &
                               (quakes.LON.between(config.BUFFER_RANGE.MIN_LON, config.BUFFER_RANGE.MAX_LON))]

    # add indicator if inside buffer for Olympia
    olympia = pd.DataFrame({'longitude': [config.OLYMPIA_GEO.LON], 'latitude': [config.OLYMPIA_GEO.LAT]})
    olympia = gpd.GeoDataFrame(olympia,
                               geometry=gpd.points_from_xy(olympia.longitude, olympia.latitude, crs="EPSG:4326"))
    olympia.geometry = olympia.to_crs("EPSG:32634").geometry.buffer(config.STUDY_BUFFER * 1000).to_crs("EPSG:4326")

    quakes_in_zone = gpd.sjoin(quakes_in_region, olympia, how="inner")[["DATE", "LAT", "LON"]].drop_duplicates()
    quakes_in_zone["_I_buffer"] = 1

    quakes_in_region = pd.merge(quakes_in_region, quakes_in_zone, on=["DATE", "LAT", "LON"], how="left").fillna({'_I_buffer': 0})
    print("combined", "init", len(quakes_in_region), quakes_in_region.DATE.nunique())
    quakes_in_region = deduplicate_combined_catalogue(quakes_in_region)
    print("combined", "post", len(quakes_in_region), quakes_in_region.DATE.nunique())

    quakes_w_dist = calculate_source_site_distance(quakes_in_region)
    quakes_w_vs30 = add_vs30(quakes_w_dist)

    # add zone
    from shapely import wkt
    zones = pd.read_csv("data/seismic-zones.csv")
    zones['geometry'] = zones['WKT'].apply(wkt.loads)
    zones = gpd.GeoDataFrame(zones, crs='epsg:4326')[["name", "geometry"]]
    zones.columns = ["ZONE", "geometry"]
    out = gpd.sjoin_nearest(quakes_w_vs30.drop("index_right", axis=1).to_crs("EPSG:32634"), zones.to_crs("EPSG:32634"), how="left")

    zones = out.ZONE
    zones[zones == "Zone 3"] = "Zone 2"
    out["ZONE"] = zones
    return out
