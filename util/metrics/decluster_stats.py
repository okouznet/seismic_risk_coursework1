import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as cx

def plot_main_events(df, main_events,_save=False, file_prefix=True):
    def scaler(min_scale_num, max_scale_num, var):
        return (max_scale_num - min_scale_num) * ((var - min(var)) / (max(var) - min(var))) + min_scale_num

    BUFFER_KM = 200
    faults = gpd.read_file("data/geo_data/faults/Faults_line.shp").to_crs("EPSG:4326")
    faults_projected = gpd.read_file("data/geo_data/faults/Fault_projection.shp").to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(25, 10))
    # lvl = cx.Place('Greece')

    faults.plot(color="brown", ax=ax, linewidth=0.5)
    faults_projected.plot(color="brown", ax=ax, linewidth=0.5, alpha=0.2)

    df.plot(color="grey", ax=ax, alpha = 1, markersize=5)
    main_events["MAG"] = main_events["MAG"].astype(float)
    main_events["marker_size"] = scaler(0, 100, main_events['MAG'].astype(float))
    main_events.plot(  # color="blue",
        column="MAG",
        ax=ax,
        alpha=1,
        cmap='OrRd',
        legend=True,
        markersize=10#scaler(0, 100, main_events['MAG']).astype(float)
    )

    olympia = pd.DataFrame({'longitude': [21.630049], 'latitude': [37.637807]})
    olympia = gpd.GeoDataFrame(olympia,
                               geometry=gpd.points_from_xy(olympia.longitude, olympia.latitude, crs="EPSG:4326"))
    olympia.plot(color="black", edgecolor="black", ax=ax, alpha=1, markersize=150)
    olympia.geometry.to_crs("epsg:32633").buffer(BUFFER_KM * 1000).to_crs("EPSG:4326").boundary.plot(color="black",
                                                                                                     ax=ax, alpha=1,
                                                                                                     ls='--')

    cx.add_basemap(ax=ax, crs=df.crs, source=cx.providers.OpenTopoMap)
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    world = gpd.read_file(url)
    world.plot(ax=ax)

    ax.set_xlim([19, 26])
    ax.set_ylim([35.5, 40])
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    # plt.axis('off')
    plt.title("Olympia, Greece: Earthquake events and faults")
    # ax.legend()
    if _save:
        plt.savefig(f"output/figs/{file_prefix}_plot_olympia_events.png", bbox_inches='tight')
    else:
        plt.show()