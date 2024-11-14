import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import contextily as cx
from dotmap import DotMap

import statsmodels.api as sm
import glob
import abc

# inspired by https://docs.openquake.org/oq-engine/2.6/_modules/openquake/hmtk/seismicity/declusterer/dec_gardner_knopoff.html
# All functionality has beeen independently verified and tested
# github: https://github.com/gem/oq-engine/blob/engine-2.6/openquake/hmtk/seismicity/declusterer/distance_time_windows.py

def GardnerKnopoffWindow(magnitude):
    sw_space = np.power(10.0, 0.1238 * magnitude + 0.983)
    sw_time = np.power(10.0, 0.032 * magnitude + 2.7389) / 364.75
    sw_time[magnitude < 6.5] = np.power(10.0, 0.5409 * magnitude[magnitude < 6.5] - 0.547) / 364.75
    return sw_space, sw_time


def GruenthalWindow(magnitude):
    sw_space = np.exp(1.77 + np.sqrt(0.037 + 1.02 * magnitude))
    sw_time = np.abs(
        (np.exp(-3.95 + np.sqrt(0.62 + 17.32 * magnitude))) / 364.75)
    sw_time[magnitude >= 6.5] = np.power(
        10, 2.8 + 0.024 * magnitude[magnitude >= 6.5]) / 364.75
    return sw_space, sw_time


def UhrhammerWindow(magnitude):
    sw_space = np.exp(-1.024 + 0.804 * magnitude)
    sw_time = np.exp(-2.87 + 1.235 * magnitude) / 364.75
    return sw_space, sw_time


def haversine(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
                                            (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                      np.sqrt(1 - aval))).T
        i += 1
    return distance


def decimal_year(year, month, day):
    """
    Allows to calculate the decimal year for a vector of dates
    (TODO this is legacy code kept to maintain comparability with previous
    declustering algorithms!)

    :param year: year column from catalogue matrix
    :type year: numpy.ndarray
    :param month: month column from catalogue matrix
    :type month: numpy.ndarray
    :param day: day column from catalogue matrix
    :type day: numpy.ndarray
    :returns: decimal year column
    :rtype: numpy.ndarray
    """
    marker = np.array([0., 31., 59., 90., 120., 151., 181.,
                       212., 243., 273., 304., 334.])
    tmonth = (month - 1).astype(int)
    day_count = marker[tmonth] + day - 1.
    dec_year = year + (day_count / 365.)

    return dec_year


def decluster(catalogue, config):
    """
        The configuration of this declustering algorithm requires two
        objects:
        - A time-distance window object (key is 'time_distance_window')
        - A value in the interval [0,1] expressing the fraction of the
        time window used for aftershocks (key is 'fs_time_prop')

        :param catalogue:
            Catalogue of earthquakes
        :type catalogue: Dictionary
        :param config:
            Configuration parameters
        :type config: Dictionary

        :returns:
          **vcl vector** indicating cluster number,
          **flagvector** indicating which eq events belong to a cluster
        :rtype: numpy.ndarray
    """
    # Get relevant parameters
    neq = len(catalogue.data['MAG'])  # Number of earthquakes
    # Get decimal year (needed for time windows)
    year_dec = decimal_year(catalogue.data['YEAR'], catalogue.data['MONTH'], catalogue.data['DAY'])
    # Get space and time windows corresponding to each event
    sw_space, sw_time = (config['time_distance_window'](catalogue.data['MAG']))

    # Initial Position Identifier
    eqid = np.arange(0, neq, 1)
    # Pre-allocate cluster index vectors
    vcl = np.zeros(neq, dtype=int)
    # Sort magnitudes into descending order

    id0 = np.flipud(np.argsort(catalogue.data['MAG'], kind='heapsort'))

    longitude = catalogue.data['LAT'].iloc[id0].values
    latitude = catalogue.data['LON'].iloc[id0].values
    sw_space = sw_space.iloc[id0].values
    sw_time = sw_time.iloc[id0].values
    year_dec = year_dec.iloc[id0].values
    eqid = eqid[id0]
    flagvector = np.zeros(neq, dtype=int)

    # Begin cluster identification
    clust_index = 0
    for i in range(0, neq - 1):
        if vcl[i] == 0:
            # Find Events inside both fore- and aftershock time windows
            dt = year_dec - year_dec[i]
            vsel = np.logical_and(
                vcl == 0,
                np.logical_and(
                    dt >= (-sw_time[i] * config['fs_time_prop']),
                    dt <= sw_time[i]))
            # Of those events inside time window,
            # find those inside distance window
            # print(longitude[vsel], latitude[vsel])
            # print(longitude.values[1], latitude.values[1])
            vsel1 = haversine(longitude[vsel],
                              latitude[vsel],
                              longitude[i],
                              latitude[i]) <= sw_space[i]

            vsel[vsel] = vsel1[0]
            temp_vsel = np.copy(vsel)
            temp_vsel[i] = False
            if any(temp_vsel):
                # Allocate a cluster number
                vcl[vsel] = clust_index + 1
                flagvector[vsel] = 1
                # For those events in the cluster before the main event,
                # flagvector is equal to -1
                temp_vsel[dt >= 0.0] = False
                flagvector[temp_vsel] = -1
                flagvector[i] = 0
                clust_index += 1

    # Re-sort the catalog_matrix into original order
    id1 = np.argsort(eqid, kind='heapsort')
    eqid = eqid[id1]
    vcl = vcl[id1]
    flagvector = flagvector[id1]

    return vcl, flagvector
