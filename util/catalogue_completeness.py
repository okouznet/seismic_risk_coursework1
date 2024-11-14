import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import contextily as cx
from dotmap import DotMap

import statsmodels.api as sm
import glob
import abc

# Stepp Method

# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (C) 2010-2023 GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (openquake.hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (https://www.globalquakemodel.org/tools-products) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (openquake.hmtk) is therefore distributed
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.

"""
Module :mod:`openquake.hmtk.seismicity.completeness.comp_stepp_1972` defines
the openquake.hmtk implementation of the Stepp (1972) algorithm for analysing
the completeness of an earthquake catalogue
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

MARKER_NORMAL = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
)
MARKER_LEAP = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])

SECONDS_PER_DAY = 86400.0

from datetime import datetime, timedelta

def dt_to_dec(dt):
    """Convert a datetime to decimal year."""
    dt = dt.replace(tzinfo=None)
    year_start = datetime(dt.year, 1, 1)
    year_end = year_start.replace(year=dt.year +1)
    return dt.year + ((dt - year_start).total_seconds() /  # seconds so far
                      float((year_end - year_start).total_seconds()))  # seconds in year


def piecewise_linear_scalar(params, xval):
    """Piecewise linear function for a scalar variable xval (float).

    :param params:
        Piecewise linear parameters (numpy.ndarray) in the following form:
        [slope_i,... slope_n, turning_point_i, ..., turning_point_n, intercept]
        Length params === 2 * number_segments, e.g.
        [slope_1, slope_2, slope_3, turning_point1, turning_point_2, intercept]
    :param xval:
        Value for evaluation of function (float)
    :returns:
        Piecewise linear function evaluated at point xval (float)
    """
    n_params = len(params)
    n_seg, remainder = divmod(n_params, 2)
    if remainder:
        raise ValueError(
            "Piecewise Function requires 2 * nsegments parameters"
        )

    if n_seg == 1:
        return params[1] + params[0] * xval

    gradients = params[0:n_seg]
    turning_points = params[n_seg:-1]
    c_val = np.array([params[-1]])

    for iloc in range(1, n_seg):
        c_val = np.hstack(
            [
                c_val,
                (
                        c_val[iloc - 1]
                        + gradients[iloc - 1] * turning_points[iloc - 1]
                )
                - (gradients[iloc] * turning_points[iloc - 1]),
                ]
        )

    if xval <= turning_points[0]:
        return gradients[0] * xval + c_val[0]
    elif xval > turning_points[-1]:
        return gradients[-1] * xval + c_val[-1]
    else:
        select = np.nonzero(turning_points <= xval)[0][-1] + 1
    return gradients[select] * xval + c_val[select]

def leap_check(year):
    """
    Returns logical array indicating if year is a leap year
    """
    return np.logical_and(
        (year % 4) == 0, np.logical_or((year % 100 != 0), (year % 400) == 0)
    )


def decimal_time(year, month, day, hour, minute, second):
    """
    Returns the full time as a decimal value

    :param year:
        Year of events (integer numpy.ndarray)
    :param month:
        Month of events (integer numpy.ndarray)
    :param day:
        Days of event (integer numpy.ndarray)
    :param hour:
        Hour of event (integer numpy.ndarray)
    :param minute:
        Minute of event (integer numpy.ndarray)
    :param second:
        Second of event (float numpy.ndarray)
    :returns decimal_time:
        Decimal representation of the time (as numpy.ndarray)
    """
    tmo = np.ones_like(year, dtype=int)
    tda = np.ones_like(year, dtype=int)
    tho = np.zeros_like(year, dtype=int)
    tmi = np.zeros_like(year, dtype=int)
    tse = np.zeros_like(year, dtype=float)
    #
    # Checking inputs
    if any(month < 1) or any(month > 12):
        raise ValueError("Month must be in [1, 12]")
    if any(day < 1) or any(day > 31):
        raise ValueError("Day must be in [1, 31]")
    if any(hour < 0) or any(hour > 24):
        raise ValueError("Hour must be in [0, 24]")
    if any(minute < 0) or any(minute > 60):
        raise ValueError("Minute must be in [0, 60]")
    if any(second < 0) or any(second > 60):
        raise ValueError("Second must be in [0, 60]")
    #
    # Initialising values
    if any(month):
        tmo = month
    if any(day):
        tda = day
    if any(hour):
        tho = hour
    if any(minute):
        tmi = minute
    if any(second):
        tse = second
    #
    # Computing decimal
    tmonth = tmo - 1
    day_count = MARKER_NORMAL[tmonth] + tda - 1
    id_leap = leap_check(year)
    leap_loc = np.where(id_leap)[0]
    day_count[leap_loc] = MARKER_LEAP[tmonth[leap_loc]] + tda[leap_loc] - 1
    year_secs = (
            (day_count.astype(float) * SECONDS_PER_DAY)
            + tse
            + (60.0 * tmi.astype(float))
            + (3600.0 * tho.astype(float))
    )
    dtime = year.astype(float) + (year_secs / (365.0 * 24.0 * 3600.0))
    dtime[leap_loc] = year[leap_loc].astype(float) + (
            year_secs[leap_loc] / (366.0 * 24.0 * 3600.0)
    )
    return dtime



def get_bilinear_residuals_stepp(input_params, xvals, yvals, slope1_fit):
    """
    Returns the residual sum-of-squares value of a bilinear fit to a data
    set - with a segment - 1 gradient fixed by an input value (slope_1_fit)

    :param list input_params:
        Input parameters for the bilinear model [slope2, crossover_point,
                                                 intercept]
    :param numpy.ndarray xvals:
        x-values of the data to be fit

    :param numpy.ndarray yvals:
        y-values of the data to be fit

    :param float slope1_fit:
        Gradient of the first slope

    :returns:
        Residual sum-of-squares of fit
    """
    params = np.hstack([slope1_fit, input_params])
    num_x = len(xvals)
    y_model = np.zeros(num_x, dtype=float)
    residuals = np.zeros(num_x, dtype=float)
    for iloc in range(0, num_x):
        y_model[iloc] = piecewise_linear_scalar(params, xvals[iloc])
        residuals[iloc] = (yvals[iloc] - y_model[iloc]) ** 2.0
    return np.sum(residuals)



class Stepp1971():
    """
    Implements the completeness analysis methodology of Stepp (1972)
    Stepp, J. C. (1972) Analysis of Completeness of the Earhquake Sample in
    the Puget Sound Area and Its Effect on Statistical Estimates of Earthquake
    Hazard, NOAA Environmental Research Laboratories.

    The original methodology of J. C. Stepp (1972) implements a graphical
    method in which the deviation of the observed rate from the expected
    Poisson rate is assessed by judgement. To implement the selection
    in an automated fashion this implementation uses optimisation of a
    2-segment piecewise linear fit to each magnitude bin, using the
    segment intersection point to identify the completeness period.

    Adaptation implemented by Weatherill, G. A., GEM Model Facility, Pavia

    :attribute numpy.ndarray magnitude_bin:
        Edges of the magnitude bins

    :attribute numpy.ndarray sigma:
        Sigma lambda defined by Equation 4 in Stepp (1972)

    :attribute numpy.ndarray time_values:
        Duration values

    :attribute numpy.ndarray model_line:
        Expected Poisson rate for each magnitude bin

    :attribute numpy.ndarray completeness_table:
        Resulting completeness table
    """

    def __init__(self):
        #BaseCatalogueCompleteness.__init__(self)
        self.magnitude_bin = None
        self.time_values = None
        self.sigma = None
        self.model_line = None
        self.completeness_table = None
        self.end_year = None

    def completeness(self, catalogue, config):
        """
        Gets the completeness table.

        :param catalogue:
            Earthquake catalogue as instance of
            :class:`openquake.hmtk.seismicity.catalogue.Catalogue`

        :param dict config:
            Configuration parameters of the algorithm, containing the
            following information:
            'magnitude_bin' Size of magnitude bin (non-negative float)
            'time_bin' Size (in dec. years) of the time window
            (non-negative float)
            'increment_lock' Boolean to indicate whether to ensure
            completeness magnitudes always decrease with more recent bins

        :returns:
            2-column table indicating year of completeness and corresponding
            magnitude numpy.ndarray
        """
        # If mag_bin is an array then bins are input, otherwise a single
        # parameter is input
        # dyear = decimal_time(
        #    catalogue.data["DATE"].dt.year,
        #    catalogue.data["DATE"].dt.month,
        #    catalogue.data["DATE"].dt.day,
        #    catalogue.data["DATE"].dt.hour,
        #    catalogue.data["DATE"].dt.minute,
        #    catalogue.data["DATE"].dt.second,
        # )
        dyear = catalogue.data['DATE'].apply(lambda x: dt_to_dec(x)).values
        mag = catalogue.data["MAG"]

        # Get magnitude bins
        self.magnitude_bin = self._get_magnitudes_from_spacing(
            catalogue.data["MAG"], config["magnitude_bin"]
        )
        print("magnitude bins" ,self.magnitude_bin)


        # Get time bins
        _s_year, time_bin = self._get_time_limits_from_config(config, dyear)

        print("time bins" ,_s_year, time_bin)

        # Count magnitudes
        (
            self.sigma,
            _counter,
            n_mags,
            n_times,
            self.time_values,
        ) = self._count_magnitudes(mag, dyear, time_bin)


        print(self.sigma ,_counter ,n_mags ,n_times ,self.time_values)
        # Get completeness magnitudes
        comp_time, _gradient_2, self.model_line = self.get_completeness_points(
            self.time_values, self.sigma, n_mags, n_times
        )

        print
        # If the increment lock is selected then ensure completeness time
        # does not decrease
        if config["increment_lock"]:
            for iloc in range(0, len(comp_time)):
                cond = (
                               iloc > 0 and (comp_time[iloc] < comp_time[iloc - 1])
                       ) or np.isnan(comp_time[iloc])
                if cond:
                    comp_time[iloc] = comp_time[iloc - 1]

        self.completeness_table = np.column_stack(
            [np.floor(self.end_year - comp_time), self.magnitude_bin[:-1]]
        )
        return self.completeness_table

    def simplify(self, deduplicate=True, mag_range=None, year_range=None):
        """
        Simplify a completeness table result. Intended to work with
        'increment_lock' enabled.
        """

        if self.completeness_table is None:
            return

        years = self.completeness_table[:, 0]
        mags = self.completeness_table[:, 1]
        keep = np.array([True] * years.shape[0])

        if deduplicate:
            keep[1:] = years[1:] != years[:-1]

        if year_range is not None:
            year_min, year_max = year_range
            if year_min is not None:
                too_early = years < year_min
                keep &= years >= years[too_early].max()
                self.completeness_table[too_early, 0] = year_min
            if year_max is not None:
                keep &= years <= year_max

        if mag_range is not None:
            mag_min, mag_max = mag_range
            if mag_min is not None:
                keep &= mags >= mag_min
            if mag_max is not None:
                keep &= mags <= mag_max

        self.completeness_table = self.completeness_table[keep, :]
        self.model_line = self.model_line[:, keep]
        self.sigma = self.sigma[:, keep]
        self.magnitude_bin = self.magnitude_bin[np.hstack((keep, True))]

    def _get_time_limits_from_config(self, config, dec_year):
        """
        Defines the time bins for consideration based on the config time_bin
        settings - also sets self.end_year (int) the latest year in catalogue

        :param dict config:
            Configuration for the Stepp (1971) algorithm

        :param numpy.ndarray dec_year:
            Time of the earthquake in decimal years

        :returns:
            * start_year: Earliest year found in the catalogue
            * time_bin: Bin edges of the time windows
        """
        cond = isinstance(config["time_bin"], list) or isinstance(
            config["time_bin"], np.ndarray
        )
        if cond:
            # Check to make sure input years are in order from recent to oldest
            for ival in range(1, len(config["time_bin"])):
                diff = config["time_bin"][ival] - config["time_bin"][ival - 1]
                if diff > 0.0:
                    raise ValueError(
                        "Configuration time windows must be "
                        "ordered from recent to oldest!"
                    )

            self.end_year = config["time_bin"][0]
            start_year = config["time_bin"][-1]
            time_bin = np.array(config["time_bin"])
        else:
            self.end_year = np.floor(np.max(dec_year))
            start_year = np.floor(np.min(dec_year))
            if (self.end_year - start_year) < config["time_bin"]:
                raise ValueError(
                    "Catalogue duration smaller than time bin"
                    " width - change time window size!"
                )
            time_bin = np.arange(
                self.end_year - config["time_bin"],
                start_year - config["time_bin"],
                -config["time_bin"],
                )

        return start_year, time_bin

    def _get_magnitudes_from_spacing(self, magnitudes, delta_m):
        """If a single magnitude spacing is input then create the bins

        :param numpy.ndarray magnitudes:
            Vector of earthquake magnitudes

        :param float delta_m:
            Magnitude bin width

        :returns: Vector of magnitude bin edges (numpy.ndarray)
        """
        min_mag = np.min(magnitudes)
        max_mag = np.max(magnitudes)
        if (max_mag - min_mag) < delta_m:
            raise ValueError("Bin width greater than magnitude range!")

        mag_bins = np.arange(
            np.floor(min_mag), np.ceil(max_mag + delta_m), delta_m
        )

        # Check to see if there are magnitudes in lower and upper bins
        is_mag = np.logical_and(
            mag_bins - max_mag < delta_m, min_mag - mag_bins < delta_m
        )
        mag_bins = mag_bins[is_mag]
        return mag_bins

    def _count_magnitudes(self, mags, times, time_bin):
        """
        For each completeness magnitude-year counts the number of events
        inside each magnitude bin.

        :param numpy.ndarray mags:
            Magnitude of earthquakes

        :param numpy.ndarray times:
            Vector of decimal event times

        :param numpy.ndarray time_bin:
            Vector of bin edges of the time windows

        :returns:
            * sigma - Poisson variance (numpy.ndarray)
            * counter - Number of earthquakes in each magnitude-time bin
            * n_mags - number of magnitude bins (Integer)
            * n_times - number of time bins (Integer)
            * n_years - effective duration of each time window (numpy.ndarray)
        """
        n_mags = len(self.magnitude_bin) - 1
        n_times = len(time_bin)
        counter = np.zeros([n_times, n_mags], dtype=int)
        # Count all the magnitudes later than or equal to the reference time
        for iloc in range(0, n_times):
            id0 = times > time_bin[iloc]
            counter[iloc, :] = np.histogram(mags[id0], self.magnitude_bin)[0]
        # Get sigma_lambda  = sqrt(n / nyears) / sqrt(n_years)
        sigma = np.zeros([n_times, n_mags], dtype=float)
        n_years = np.floor(np.max(times)) - time_bin
        for iloc in range(0, n_mags):
            id0 = counter[:, iloc] > 0
            if any(id0):
                nvals = counter[id0, iloc].astype(float)
                sigma[id0, iloc] = np.sqrt((nvals / n_years[id0])) / np.sqrt(
                    n_years[id0]
                )

        return sigma, counter, n_mags, n_times, n_years

    def get_completeness_points(self, n_years, sigma, n_mags, n_time):
        """Fits a bilinear model to each sigma-n_years combination
        in order to get the crossover point. The gradient of the first line
        must always be 1 / sqrt(T), but it is free for the other lines

        :param numpy.ndarray  n_years:
            Duration of each completeness time window

        :param numpy.ndarray sigma:
            Poisson variances of each time-magnitude combination

        :param int n_mags:
            Number of magnitude bins

        :param int n_time:
            Number of time bins

        :returns:
            * comp_time (Completeness duration)
            * gradient_2 (Gradient of second slope of piecewise linear fit)
            * model_line (Expected Poisson rate for data (only used for plot)
        """
        time_vals = np.log10(n_years)
        sigma_vals = np.copy(sigma)
        valid_mapper = np.ones([n_time, n_mags], dtype=bool)
        valid_mapper[sigma_vals < 1e-9] = False
        comp_time = np.zeros(n_mags, dtype=float)
        gradient_2 = np.zeros(n_mags, dtype=float)
        model_line = np.zeros([n_time, n_mags], dtype=float)
        for iloc in range(0, n_mags):
            id0 = valid_mapper[:, iloc]
            if np.sum(id0) < 3:
                # Too few events for fitting a bilinear model!
                comp_time[iloc] = np.nan
                gradient_2[iloc] = np.nan
                model_line[:, iloc] = np.nan
            else:
                (
                    comp_time[iloc],
                    gradient_2[iloc],
                    model_line[id0, iloc],
                ) = self._fit_bilinear_to_stepp(
                    time_vals[id0], np.log10(sigma[id0, iloc])
                )
        return comp_time, gradient_2, model_line

    def _fit_bilinear_to_stepp(self, xdata, ydata, initial_values=None):
        """
        Returns the residuals of a bilinear fit subject to the following
        constraints: 1) gradient of slope 1 = 1 / sqrt(T)
                     2) Crossover (x_c) < 0
                     3) gradient 2 is always < 0

        :param numpy.ndarray xdata:
            x-value of the data set

        :param numpy.ndarray ydata:
            y-value of the data set

        :param list initial_values:
            For unit-testing allows the possibility to specify the initial
            values of the algorithm [slope_2, cross_over, intercept]

        :returns:
            * completeness_time: The duration of completeness of the bin
            * Gradient of the second slope
            * model_line: Expected Poisson model
        """
        fixed_slope = -0.5  # f'(log10(T^-0.5)) === 0.5
        if isinstance(initial_values, list) or isinstance(
                initial_values, np.ndarray
        ):
            x_0 = initial_values
        else:
            x_0 = [-1.0, xdata[int(len(xdata) / 2)], xdata[0]]

        bnds = ((None, fixed_slope), (0.0, None), (None, None))
        result, _, convergence_info = fmin_l_bfgs_b(
            get_bilinear_residuals_stepp,
            x_0,
            args=(xdata, ydata, fixed_slope),
            approx_grad=True,
            bounds=bnds,
            disp=0,
        )

        if convergence_info["warnflag"] != 0:
            # Optimisation has failed to converge - print the reason why
            print(convergence_info["task"])
            return np.nan, np.nan, np.nan * np.ones(len(xdata))

        # Result contains three parameters = m_2, x_c, c_0
        # x_c is the crossover point (i.e. the completeness_time)
        # m_2 is the gradient of the latter slope
        # c_0 is the intercept - which helps locate the line at the data
        model_line = 10.0 ** (fixed_slope * xdata + result[2])
        completeness_time = 10.0 ** result[1]
        return completeness_time, result[0], model_line

