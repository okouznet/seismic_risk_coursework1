import util.catalogue_processing as catalogue_processing
import util.catalogue_declustering as catalogue_declustering
import util.catalogue_completeness as catalogue_completeness

import util.metrics.summary_stats as summary_stats
import util.metrics.completeness_stats as completeness_stats
import util.metrics.decluster_stats as decluster_stats
import util.recurrence_relationships as recurrence_relationships
import util.metrics.metrics_util as metrics_util
import util.attenuation as attenuation

from dotmap import DotMap
import config as config
import pandas as pd
import numpy as np
import geopandas as gpd

# TODO: EXPAND AREA
# TODO: split by ZONE
# TODO: limit to 2022.
# TODO: attenuation PER ZONE

if __name__ == "__main__":

    _SAVE=True
    _VERBOSE=True

    catalogue = catalogue_processing.combine_catalogues(_rebuild=False)
    print(len(catalogue))
    catalogue["DATE"] = pd.to_datetime(catalogue["DATE"])
    catalogue = gpd.GeoDataFrame(
        catalogue, geometry=gpd.points_from_xy(catalogue.LON, catalogue.LAT), crs="EPSG:4326"
    )
    catalogue = catalogue[catalogue._I_buffer == 1]

    MAX_YEAR = config.MAX_YEAR
    file_prefix = f"M{str(int(config.MAG_MIN))}_{MAX_YEAR}"
    table_format = "tex"
    catalogue = catalogue[~catalogue.data_source.isin(["emdat"])]
    catalogue = catalogue[catalogue.DATE.dt.year <= MAX_YEAR]
    if _VERBOSE:
        # preliminary stats
        summary_stats.plot_earthquake_counts_over_time(catalogue, _save=_SAVE, file_prefix=file_prefix)

        summary_stats.plot_magnitude_kde(catalogue, _save=_SAVE)

        summary_stats.stats_by_column(catalogue, "data_source", file_prefix, format=table_format)
        summary_stats.stats_by_column(catalogue, "MAGTYPE", file_prefix, format=table_format)

        # preliminary completeness investigation
        completeness_stats.plot_magbin_counts(catalogue, [2, 3, 4, 5, 6, np.inf],  _save=_SAVE, file_prefix=file_prefix)  # np.arange(3, 6.5, 0.5).tolist()
        completeness_stats.plot_init_completeness_assessment(catalogue, _save=_SAVE, file_prefix=file_prefix)

    # decluster catalogue
    catalogue_decluster = catalogue[catalogue.MAG >= config.MAG_MIN].sort_values('MAG', ascending=False)
    print(len(catalogue), len(catalogue_decluster))

    catalogue_decluster["YEAR"] = catalogue_decluster["DATE"].dt.year
    catalogue_decluster["MONTH"] = catalogue_decluster["DATE"].dt.month
    catalogue_decluster["DAY"] = catalogue_decluster["DATE"].dt.day
    catalogue_decluster[['YEAR', 'MONTH', 'DAY']] = catalogue_decluster[['YEAR', 'MONTH', 'DAY']].astype(int)

    vcl, flagvector = catalogue_declustering.decluster(catalogue=DotMap({"data": catalogue_decluster}), config=config.DECLUSTER_CONFIGS)

    catalogue_decluster['cluster'] = vcl
    catalogue_decluster['cluster_flag'] = flagvector

    main_events = catalogue_decluster[catalogue_decluster.cluster_flag == 0]
    #decluster_stats.plot_main_events(catalogue_decluster, main_events, _save=_SAVE, file_prefix="")
    summary_stats.stats_by_column(main_events, "data_source", f"main_events_{file_prefix}", format=table_format)
    summary_stats.stats_by_column(main_events, "MAGTYPE", f"main_events_{file_prefix}", format=table_format)

    # # test catalogue for completeness using Stepp Method
    print(main_events.groupby("ZONE")["ID"].count())
    pga = attenuation.get_pga_data()
    print(pga.groupby("ZONE")["PGA"].count())

    MAG_MAX_TABLES = {}
    COMPLETENESS_TABLES = {}
    RECURR_TABLES = {}
    for z in main_events["ZONE"].unique():
        main_events_z = main_events[main_events.ZONE == z]
        stepp = catalogue_completeness.Stepp1971()
        stepp.completeness(DotMap({"data": main_events_z}), config.COMPLETENESS_CONFIGS)
        # output completeness tables
        completeness_table = pd.DataFrame(stepp.completeness_table)
        completeness_table.columns = ["Year", "Mc"]
        #completeness_table.to_latex(f"output/tables/{file_prefix}_Z{z}_stepp_completeness_table.tex")
        #pd.DataFrame(stepp.model_line).to_latex(f"output/tables/{file_prefix}_Z{z}_stepp_model_line.tex")

        # attenuation relationships
        recurr_rel_df, reg_res = recurrence_relationships.richter_recurrence_rel(main_events_z)
        recurrence_relationships.visualize_RN_rel(recurr_rel_df, reg_res, _save=_SAVE, file_prefix=f"{file_prefix}_Z{z}")

        mag_max_table = recurrence_relationships.mag_max_table(reg_res, file_prefix=f"{file_prefix}_Z{z}", format=table_format)
        MAG_MAX_TABLES[z] = mag_max_table
        COMPLETENESS_TABLES[z] = completeness_table
        RECURR_TABLES[z] = reg_res

    # output tables
    x = pd.concat([MAG_MAX_TABLES[x][[0, "return_period"]] for x in MAG_MAX_TABLES], axis=1)
    x.columns = ["M (Zone 2)", "Zone 2", "M (Zone 1)", "Zone 1"]
    x.to_latex("output/max_mag.tex")
    x = pd.concat([COMPLETENESS_TABLES[x] for x in COMPLETENESS_TABLES], axis=1)
    x.columns = ["Year (Zone 2)", "Zone 2", "Year (Zone 1)", "Zone 1"]
    x.to_latex("output/completeness.tex")

    from statsmodels.iolib.summary2 import summary_col

    with open("recurr_rel.tex", "w") as text_file:
        text_file.write( summary_col([RECURR_TABLES["Zone 1"], RECURR_TABLES["Zone 2"]]).as_latex())

    PGA_THRESH = np.arange(0, 10, 0.005)
    ATTEN_TABLES = {}
    Q_BY_ZONE = {}
    reg_df = attenuation.prep_reg_df(pga)
    for z in pga["ZONE"].unique():
        A = RECURR_TABLES[z].params['const']
        B = RECURR_TABLES[z].params['MAG']

        reg_df_z = reg_df[reg_df.ZONE == z]

        res = attenuation.fit_pga(reg_df_z)

        print(res.summary())
        ATTEN_TABLES[z] = res
        attenuation.plot_pga_vs_dist(reg_df_z, res, _save=_SAVE, file_prefix=f"Z{z}")
        Ns = [attenuation.get_return_period(reg_df_z, res,p, A,B) for p in PGA_THRESH]
        qs_return1 = [attenuation.calculate_exceedance_probabilty(185, N)  for N in Ns]
        qs_return2 = [attenuation.calculate_exceedance_probabilty(480, N)  for N in Ns]

        pga_lim = None
        if z == "Zone 2":
            pga_lim = 0.03
        attenuation.plot_hazard_curve(pd.DataFrame({"q_ret1": qs_return1, "q_ret2": qs_return2, "pga": PGA_THRESH}),
                                              _save=_SAVE,
                                              file_prefix=f"{file_prefix}_Z{z}",
                                              pga_lim=pga_lim
                                      )

    with open("atten_rel.tex", "w") as text_file:
        text_file.write( summary_col([ATTEN_TABLES["Zone 1"], ATTEN_TABLES["Zone 2"]]).as_latex())

    Ns_tot = [attenuation.calculate_return_period(df=reg_df, atten_data=ATTEN_TABLES, recurr_data=RECURR_TABLES, pga_thresh=p) for p in PGA_THRESH]
    qs_return1 = [attenuation.calculate_exceedance_probabilty(185, N) for N in Ns_tot]
    qs_return2 = [attenuation.calculate_exceedance_probabilty(480, N) for N in Ns_tot]
    attenuation.plot_hazard_curve(pd.DataFrame({"q_ret1": qs_return1, "q_ret2": qs_return2, "pga": PGA_THRESH}),
                                  _save=_SAVE,
                                  file_prefix=f"{file_prefix}_TOTAL")

    print(RECURR_TABLES["Zone 1"].summary())
    print(ATTEN_TABLES["Zone 1"].summary())
    # plot for overall