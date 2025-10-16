#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
import datetime
import calendar
import scipy.ndimage.interpolation as sc
from scipy.ndimage import rotate


def sun_position(time, location):
    """
    Calculate solar position (zenith and azimuth) using SPA algorithm.
    
    Implements the Solar Position Algorithm (SPA) as described in:
    Reda, I. and Andreas, A. (2004). Solar position algorithm for solar radiation applications.
    Solar Energy, 76(5), 577-589.
    
    Args:
        time (dict): Time information with keys:
            - 'year' (int): Year
            - 'month' (int): Month (1-12)
            - 'day' (int): Day of month
            - 'hour' (int): Hour (0-23)
            - 'min' (int): Minute (0-59)
            - 'sec' (int): Second (0-59)
            - 'UTC' (float): UTC offset in hours (e.g., -5 for EST)
        
        location (dict): Geographic location with keys:
            - 'latitude' (float): Latitude in degrees (-90 to 90)
            - 'longitude' (float): Longitude in degrees (-180 to 180)
            - 'altitude' (float): Elevation above sea level in meters
    
    Returns:
        dict: Solar position containing:
            - 'zenith' (float): Solar zenith angle in degrees (0=directly overhead, 90=horizon)
            - 'azimuth' (float): Solar azimuth in degrees (0=North, 90=East, 180=South, 270=West)
    
    Notes:
        - Accounts for atmospheric refraction
        - Accuracy: ~0.0003° for years 2000-6000
        - All angles in degrees unless otherwise specified
    
    Example:
        >>> time = {'year': 2020, 'month': 7, 'day': 18, 'hour': 12, 'min': 0, 'sec': 0, 'UTC': -5}
        >>> location = {'latitude': 30.27, 'longitude': -97.74, 'altitude': 0}
        >>> sun = sun_position(time, location)
        >>> print(f"Zenith: {sun['zenith']:.2f}°, Azimuth: {sun['azimuth']:.2f}°")
    """

    # 1. Calculate the Julian Day, and Century. Julian Ephemeris day, century
    # and millenium are calculated using a mean delta_t of 33.184 seconds.
    julian = julian_calculation(time)
    #print(julian)

    # 2. Calculate the Earth heliocentric longitude, latitude, and radius
    # vector (L, B, and R)
    earth_heliocentric_position = earth_heliocentric_position_calculation(julian)

    # 3. Calculate the geocentric longitude and latitude
    sun_geocentric_position = sun_geocentric_position_calculation(earth_heliocentric_position)

    # 4. Calculate the nutation in longitude and obliquity (in degrees).
    nutation = nutation_calculation(julian)

    # 5. Calculate the true obliquity of the ecliptic (in degrees).
    true_obliquity = true_obliquity_calculation(julian, nutation)

    # 6. Calculate the aberration correction (in degrees)
    aberration_correction = abberation_correction_calculation(earth_heliocentric_position)

    # 7. Calculate the apparent sun longitude in degrees)
    apparent_sun_longitude = apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction)

    # 8. Calculate the apparent sideral time at Greenwich (in degrees)
    apparent_stime_at_greenwich = apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity)

    # 9. Calculate the sun rigth ascension (in degrees)
    sun_rigth_ascension = sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position)

    # 10. Calculate the geocentric sun declination (in degrees). Positive or
    # negative if the sun is north or south of the celestial equator.
    sun_geocentric_declination = sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity,
                                                                        sun_geocentric_position)

    # 11. Calculate the observer local hour angle (in degrees, westward from south).
    observer_local_hour = observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension)

    # 12. Calculate the topocentric sun position (rigth ascension, declination and
    # rigth ascension parallax in degrees)
    topocentric_sun_position = topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                                                  observer_local_hour, sun_rigth_ascension,
                                                                  sun_geocentric_declination)

    # 13. Calculate the topocentric local hour angle (in degrees)
    topocentric_local_hour = topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position)

    # 14. Calculate the topocentric zenith and azimuth angle (in degrees)
    sun = sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour)

    return sun


def julian_calculation(t_input):
    """
    Calculate Julian day and related time parameters.
    
    Args:
        t_input (dict): Time dictionary with year, month, day, hour, min, sec, UTC
    
    Returns:
        dict: Julian day, century, ephemeris day/century/millennium
    """

    if not isinstance(t_input, dict):

        time = dict()
        time['UTC'] = 0
        time['year'] = t_input.year
        time['month'] = t_input.month
        time['day'] = t_input.day
        time['hour'] = t_input.hour
        time['min'] = t_input.minute
        time['sec'] = t_input.second
    else:
        time = t_input

    if time['month'] == 1 or time['month'] == 2:
        Y = time['year'] - 1
        M = time['month'] + 12
    else:
        Y = time['year']
        M = time['month']

    ut_time = ((time['hour'] - time['UTC'])/24) + (time['min']/(60*24)) + (time['sec']/(60*60*24))   # time of day in UT time.
    D = time['day'] + ut_time   # Day of month in decimal time, ex. 2sd day of month at 12:30:30UT, D=2.521180556

    # In 1582, the gregorian calendar was adopted
    if time['year'] == 1582:
        if time['month'] == 10:
            if time['day'] <= 4:   # The Julian calendar ended on October 4, 1582
                B = (0)
            elif time['day'] >= 15:   # The Gregorian calendar started on October 15, 1582
                A = np.floor(Y/100)
                B = 2 - A + np.floor(A/4)
            else:
                print('This date never existed!. Date automatically set to October 4, 1582')
                time['month'] = 10
                time['day'] = 4
                B = 0
        elif time['month'] < 10:   # Julian calendar
            B = 0
        else: # Gregorian calendar
            A = np.floor(Y/100)
            B = 2 - A + np.floor(A/4)
    elif time['year'] < 1582:   # Julian calendar
        B = 0
    else:
        A = np.floor(Y/100)    # Gregorian calendar
        B = 2 - A + np.floor(A/4)

    julian = dict()
    julian['day'] = D + B + np.floor(365.25*(Y+4716)) + np.floor(30.6001*(M+1)) - 1524.5

    delta_t = 0   # 33.184;
    julian['ephemeris_day'] = (julian['day']) + (delta_t/86400)
    julian['century'] = (julian['day'] - 2451545) / 36525
    julian['ephemeris_century'] = (julian['ephemeris_day'] - 2451545) / 36525
    julian['ephemeris_millenium'] = (julian['ephemeris_century']) / 10

    return julian


def earth_heliocentric_position_calculation(julian):
    """
    Calculate Earth's heliocentric position (longitude, latitude, radius).
    
    Args:
        julian (dict): Julian day parameters
    
    Returns:
        dict: Earth heliocentric position (longitude, latitude, radius in AU)
    """

    # Tabulated values for the longitude calculation
    # L terms  from the original code.
    L0_terms = np.array([[175347046.0, 0, 0],
                        [3341656.0, 4.6692568, 6283.07585],
                        [34894.0, 4.6261, 12566.1517],
                        [3497.0, 2.7441, 5753.3849],
                        [3418.0, 2.8289, 3.5231],
                        [3136.0, 3.6277, 77713.7715],
                        [2676.0, 4.4181, 7860.4194],
                        [2343.0, 6.1352, 3930.2097],
                        [1324.0, 0.7425, 11506.7698],
                        [1273.0, 2.0371, 529.691],
                        [1199.0, 1.1096, 1577.3435],
                        [990, 5.233, 5884.927],
                        [902, 2.045, 26.298],
                        [857, 3.508, 398.149],
                        [780, 1.179, 5223.694],
                        [753, 2.533, 5507.553],
                        [505, 4.583, 18849.228],
                        [492, 4.205, 775.523],
                        [357, 2.92, 0.067],
                        [317, 5.849, 11790.629],
                        [284, 1.899, 796.298],
                        [271, 0.315, 10977.079],
                        [243, 0.345, 5486.778],
                        [206, 4.806, 2544.314],
                        [205, 1.869, 5573.143],
                        [202, 2.4458, 6069.777],
                        [156, 0.833, 213.299],
                        [132, 3.411, 2942.463],
                        [126, 1.083, 20.775],
                        [115, 0.645, 0.98],
                        [103, 0.636, 4694.003],
                        [102, 0.976, 15720.839],
                        [102, 4.267, 7.114],
                        [99, 6.21, 2146.17],
                        [98, 0.68, 155.42],
                        [86, 5.98, 161000.69],
                        [85, 1.3, 6275.96],
                        [85, 3.67, 71430.7],
                        [80, 1.81, 17260.15],
                        [79, 3.04, 12036.46],
                        [71, 1.76, 5088.63],
                        [74, 3.5, 3154.69],
                        [74, 4.68, 801.82],
                        [70, 0.83, 9437.76],
                        [62, 3.98, 8827.39],
                        [61, 1.82, 7084.9],
                        [57, 2.78, 6286.6],
                        [56, 4.39, 14143.5],
                        [56, 3.47, 6279.55],
                        [52, 0.19, 12139.55],
                        [52, 1.33, 1748.02],
                        [51, 0.28, 5856.48],
                        [49, 0.49, 1194.45],
                        [41, 5.37, 8429.24],
                        [41, 2.4, 19651.05],
                        [39, 6.17, 10447.39],
                        [37, 6.04, 10213.29],
                        [37, 2.57, 1059.38],
                        [36, 1.71, 2352.87],
                        [36, 1.78, 6812.77],
                        [33, 0.59, 17789.85],
                        [30, 0.44, 83996.85],
                        [30, 2.74, 1349.87],
                        [25, 3.16, 4690.48]])

    L1_terms = np.array([[628331966747.0, 0, 0],
                        [206059.0, 2.678235, 6283.07585],
                        [4303.0, 2.6351, 12566.1517],
                        [425.0, 1.59, 3.523],
                        [119.0, 5.796, 26.298],
                        [109.0, 2.966, 1577.344],
                        [93, 2.59, 18849.23],
                        [72, 1.14, 529.69],
                        [68, 1.87, 398.15],
                        [67, 4.41, 5507.55],
                        [59, 2.89, 5223.69],
                        [56, 2.17, 155.42],
                        [45, 0.4, 796.3],
                        [36, 0.47, 775.52],
                        [29, 2.65, 7.11],
                        [21, 5.34, 0.98],
                        [19, 1.85, 5486.78],
                        [19, 4.97, 213.3],
                        [17, 2.99, 6275.96],
                        [16, 0.03, 2544.31],
                        [16, 1.43, 2146.17],
                        [15, 1.21, 10977.08],
                        [12, 2.83, 1748.02],
                        [12, 3.26, 5088.63],
                        [12, 5.27, 1194.45],
                        [12, 2.08, 4694],
                        [11, 0.77, 553.57],
                        [10, 1.3, 3286.6],
                        [10, 4.24, 1349.87],
                        [9, 2.7, 242.73],
                        [9, 5.64, 951.72],
                        [8, 5.3, 2352.87],
                        [6, 2.65, 9437.76],
                        [6, 4.67, 4690.48]])

    L2_terms = np.array([[52919.0, 0, 0],
                        [8720.0, 1.0721, 6283.0758],
                        [309.0, 0.867, 12566.152],
                        [27, 0.05, 3.52],
                        [16, 5.19, 26.3],
                        [16, 3.68, 155.42],
                        [10, 0.76, 18849.23],
                        [9, 2.06, 77713.77],
                        [7, 0.83, 775.52],
                        [5, 4.66, 1577.34],
                        [4, 1.03, 7.11],
                        [4, 3.44, 5573.14],
                        [3, 5.14, 796.3],
                        [3, 6.05, 5507.55],
                        [3, 1.19, 242.73],
                        [3, 6.12, 529.69],
                        [3, 0.31, 398.15],
                        [3, 2.28, 553.57],
                        [2, 4.38, 5223.69],
                        [2, 3.75, 0.98]])

    L3_terms = np.array([[289.0, 5.844, 6283.076],
                        [35, 0, 0],
                        [17, 5.49, 12566.15],
                        [3, 5.2, 155.42],
                        [1, 4.72, 3.52],
                        [1, 5.3, 18849.23],
                        [1, 5.97, 242.73]])
    L4_terms = np.array([[114.0, 3.142, 0],
                        [8, 4.13, 6283.08],
                        [1, 3.84, 12566.15]])

    L5_terms = np.array([1, 3.14, 0])
    L5_terms = np.atleast_2d(L5_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = L0_terms[:, 0]
    B0 = L0_terms[:, 1]
    C0 = L0_terms[:, 2]

    A1 = L1_terms[:, 0]
    B1 = L1_terms[:, 1]
    C1 = L1_terms[:, 2]

    A2 = L2_terms[:, 0]
    B2 = L2_terms[:, 1]
    C2 = L2_terms[:, 2]

    A3 = L3_terms[:, 0]
    B3 = L3_terms[:, 1]
    C3 = L3_terms[:, 2]

    A4 = L4_terms[:, 0]
    B4 = L4_terms[:, 1]
    C4 = L4_terms[:, 2]

    A5 = L5_terms[:, 0]
    B5 = L5_terms[:, 1]
    C5 = L5_terms[:, 2]

    JME = julian['ephemeris_millenium']

    # Compute the Earth Heliochentric longitude from the tabulated values.
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = np.sum(A4 * np.cos(B4 + (C4 * JME)))
    L5 = A5 * np.cos(B5 + (C5 * JME))

    earth_heliocentric_position = dict()
    earth_heliocentric_position['longitude'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                                          (L3 * np.power(JME, 3)) +
                                                          (L4 * np.power(JME, 4)) +
                                                          (L5 * np.power(JME, 5))) / 1e8
    # Convert the longitude to degrees.
    earth_heliocentric_position['longitude'] = earth_heliocentric_position['longitude'] * 180/np.pi

    # Limit the range to [0,360]
    earth_heliocentric_position['longitude'] = set_to_range(earth_heliocentric_position['longitude'], 0, 360)

    # Tabulated values for the earth heliocentric latitude.
    # B terms  from the original code.
    B0_terms = np.array([[280.0, 3.199, 84334.662],
                        [102.0, 5.422, 5507.553],
                        [80, 3.88, 5223.69],
                        [44, 3.7, 2352.87],
                        [32, 4, 1577.34]])

    B1_terms = np.array([[9, 3.9, 5507.55],
                         [6, 1.73, 5223.69]])

    A0 = B0_terms[:, 0]
    B0 = B0_terms[:, 1]
    C0 = B0_terms[:, 2]

    A1 = B1_terms[:, 0]
    B1 = B1_terms[:, 1]
    C1 = B1_terms[:, 2]

    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))

    earth_heliocentric_position['latitude'] = (L0 + (L1 * JME)) / 1e8

    # Convert the latitude to degrees.
    earth_heliocentric_position['latitude'] = earth_heliocentric_position['latitude'] * 180/np.pi

    # Limit the range to [0,360];
    earth_heliocentric_position['latitude'] = set_to_range(earth_heliocentric_position['latitude'], 0, 360)

    # Tabulated values for radius vector.
    # R terms from the original code
    R0_terms = np.array([[100013989.0, 0, 0],
                        [1670700.0, 3.0984635, 6283.07585],
                        [13956.0, 3.05525, 12566.1517],
                        [3084.0, 5.1985, 77713.7715],
                        [1628.0, 1.1739, 5753.3849],
                        [1576.0, 2.8469, 7860.4194],
                        [925.0, 5.453, 11506.77],
                        [542.0, 4.564, 3930.21],
                        [472.0, 3.661, 5884.927],
                        [346.0, 0.964, 5507.553],
                        [329.0, 5.9, 5223.694],
                        [307.0, 0.299, 5573.143],
                        [243.0, 4.273, 11790.629],
                        [212.0, 5.847, 1577.344],
                        [186.0, 5.022, 10977.079],
                        [175.0, 3.012, 18849.228],
                        [110.0, 5.055, 5486.778],
                        [98, 0.89, 6069.78],
                        [86, 5.69, 15720.84],
                        [86, 1.27, 161000.69],
                        [85, 0.27, 17260.15],
                        [63, 0.92, 529.69],
                        [57, 2.01, 83996.85],
                        [56, 5.24, 71430.7],
                        [49, 3.25, 2544.31],
                        [47, 2.58, 775.52],
                        [45, 5.54, 9437.76],
                        [43, 6.01, 6275.96],
                        [39, 5.36, 4694],
                        [38, 2.39, 8827.39],
                        [37, 0.83, 19651.05],
                        [37, 4.9, 12139.55],
                        [36, 1.67, 12036.46],
                        [35, 1.84, 2942.46],
                        [33, 0.24, 7084.9],
                        [32, 0.18, 5088.63],
                        [32, 1.78, 398.15],
                        [28, 1.21, 6286.6],
                        [28, 1.9, 6279.55],
                        [26, 4.59, 10447.39]])

    R1_terms = np.array([[103019.0, 1.10749, 6283.07585],
                        [1721.0, 1.0644, 12566.1517],
                        [702.0, 3.142, 0],
                        [32, 1.02, 18849.23],
                        [31, 2.84, 5507.55],
                        [25, 1.32, 5223.69],
                        [18, 1.42, 1577.34],
                        [10, 5.91, 10977.08],
                        [9, 1.42, 6275.96],
                        [9, 0.27, 5486.78]])

    R2_terms = np.array([[4359.0, 5.7846, 6283.0758],
                        [124.0, 5.579, 12566.152],
                        [12, 3.14, 0],
                        [9, 3.63, 77713.77],
                        [6, 1.87, 5573.14],
                        [3, 5.47, 18849]])

    R3_terms = np.array([[145.0, 4.273, 6283.076],
                        [7, 3.92, 12566.15]])

    R4_terms = [4, 2.56, 6283.08]
    R4_terms = np.atleast_2d(R4_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = R0_terms[:, 0]
    B0 = R0_terms[:, 1]
    C0 = R0_terms[:, 2]

    A1 = R1_terms[:, 0]
    B1 = R1_terms[:, 1]
    C1 = R1_terms[:, 2]

    A2 = R2_terms[:, 0]
    B2 = R2_terms[:, 1]
    C2 = R2_terms[:, 2]

    A3 = R3_terms[:, 0]
    B3 = R3_terms[:, 1]
    C3 = R3_terms[:, 2]

    A4 = R4_terms[:, 0]
    B4 = R4_terms[:, 1]
    C4 = R4_terms[:, 2]

    # Compute the Earth heliocentric radius vector
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = A4 * np.cos(B4 + (C4 * JME))

    # Units are in AU
    earth_heliocentric_position['radius'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                             (L3 * np.power(JME, 3)) +
                                             (L4 * np.power(JME, 4))) / 1e8

    return earth_heliocentric_position


def sun_geocentric_position_calculation(earth_heliocentric_position):
    """Calculate geocentric sun position from Earth heliocentric position. SPA Step 3."""

    sun_geocentric_position = dict()
    sun_geocentric_position['longitude'] = earth_heliocentric_position['longitude'] + 180
    # Limit the range to [0,360];
    sun_geocentric_position['longitude'] = set_to_range(sun_geocentric_position['longitude'], 0, 360)

    sun_geocentric_position['latitude'] = -earth_heliocentric_position['latitude']
    # Limit the range to [0,360]
    sun_geocentric_position['latitude'] = set_to_range(sun_geocentric_position['latitude'], 0, 360)
    return sun_geocentric_position


def nutation_calculation(julian):
    """
    Calculate nutation in longitude and obliquity.
    
    Args:
        julian (dict): Julian parameters
    
    Returns:
        dict: Nutation in longitude and obliquity (degrees)
    """


    # All Xi are in degrees.
    JCE = julian['ephemeris_century']

    # 1. Mean elongation of the moon from the sun
    p = np.atleast_2d([(1/189474), -0.0019142, 445267.11148, 297.85036])

    # X0 = polyval(p, JCE);
    X0 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]   # This is faster than polyval...

    # 2. Mean anomaly of the sun (earth)
    p = np.atleast_2d([-(1/300000), -0.0001603, 35999.05034, 357.52772])

    # X1 = polyval(p, JCE)
    X1 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 3. Mean anomaly of the moon
    p = np.atleast_2d([(1/56250), 0.0086972, 477198.867398, 134.96298])

    # X2 = polyval(p, JCE);
    X2 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 4. Moon argument of latitude
    p = np.atleast_2d([(1/327270), -0.0036825, 483202.017538, 93.27191])

    # X3 = polyval(p, JCE)
    X3 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 5. Longitude of the ascending node of the moon's mean orbit on the
    # ecliptic, measured from the mean equinox of the date
    p = np.atleast_2d([(1/450000), 0.0020708, -1934.136261, 125.04452])

    # X4 = polyval(p, JCE);
    X4 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # Y tabulated terms from the original code
    Y_terms = np.array([[0, 0, 0, 0, 1],
                        [-2, 0, 0, 2, 2],
                        [0, 0, 0, 2, 2],
                        [0, 0, 0, 0, 2],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [-2, 1, 0, 2, 2],
                        [0, 0, 0, 2, 1],
                        [0, 0, 1, 2, 2],
                        [-2, -1, 0, 2, 2],
                        [-2, 0, 1, 0, 0],
                        [-2, 0, 0, 2, 1],
                        [0, 0, -1, 2, 2],
                        [2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [2, 0, -1, 2, 2],
                        [0, 0, -1, 0, 1],
                        [0, 0, 1, 2, 1],
                        [-2, 0, 2, 0, 0],
                        [0, 0, -2, 2, 1],
                        [2, 0, 0, 2, 2],
                        [0, 0, 2, 2, 2],
                        [0, 0, 2, 0, 0],
                        [-2, 0, 1, 2, 2],
                        [0, 0, 0, 2, 0],
                        [-2, 0, 0, 2, 0],
                        [0, 0, -1, 2, 1],
                        [0, 2, 0, 0, 0],
                        [2, 0, -1, 0, 1],
                        [-2, 2, 0, 2, 2],
                        [0, 1, 0, 0, 1],
                        [-2, 0, 1, 0, 1],
                        [0, -1, 0, 0, 1],
                        [0, 0, 2, -2, 0],
                        [2, 0, -1, 2, 1],
                        [2, 0, 1, 2, 2],
                        [0, 1, 0, 2, 2],
                        [-2, 1, 1, 0, 0],
                        [0, -1, 0, 2, 2],
                        [2, 0, 0, 2, 1],
                        [2, 0, 1, 0, 0],
                        [-2, 0, 2, 2, 2],
                        [-2, 0, 1, 2, 1],
                        [2, 0, -2, 0, 1],
                        [2, 0, 0, 0, 1],
                        [0, -1, 1, 0, 0],
                        [-2, -1, 0, 2, 1],
                        [-2, 0, 0, 0, 1],
                        [0, 0, 2, 2, 1],
                        [-2, 0, 2, 0, 1],
                        [-2, 1, 0, 2, 1],
                        [0, 0, 1, -2, 0],
                        [-1, 0, 1, 0, 0],
                        [-2, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, -2, 2, 2],
                        [-1, -1, 1, 0, 0],
                        [0, 1, 1, 0, 0],
                        [0, -1, 1, 2, 2],
                        [2, -1, -1, 2, 2],
                        [0, 0, 3, 2, 2],
                        [2, -1, 0, 2, 2]])

    nutation_terms = np.array([[-171996, -174.2, 92025, 8.9],
                                [-13187, -1.6, 5736, -3.1],
                                [-2274, -0.2, 977, -0.5],
                                [2062, 0.2, -895, 0.5],
                                [1426, -3.4, 54, -0.1],
                                [712, 0.1, -7, 0],
                                [-517, 1.2, 224, -0.6],
                                [-386, -0.4, 200, 0],
                                [-301, 0, 129, -0.1],
                                [217, -0.5, -95, 0.3],
                                [-158, 0, 0, 0],
                                [129, 0.1, -70, 0],
                                [123, 0, -53, 0],
                                [63, 0, 0, 0],
                                [63, 0.1, -33, 0],
                                [-59, 0, 26, 0],
                                [-58, -0.1, 32, 0],
                                [-51, 0, 27, 0],
                                [48, 0, 0, 0],
                                [46, 0, -24, 0],
                                [-38, 0, 16, 0],
                                [-31, 0, 13, 0],
                                [29, 0, 0, 0],
                                [29, 0, -12, 0],
                                [26, 0, 0, 0],
                                [-22, 0, 0, 0],
                                [21, 0, -10, 0],
                                [17, -0.1, 0, 0],
                                [16, 0, -8, 0],
                                [-16, 0.1, 7, 0],
                                [-15, 0, 9, 0],
                                [-13, 0, 7, 0],
                                [-12, 0, 6, 0],
                                [11, 0, 0, 0],
                                [-10, 0, 5, 0],
                                [-8, 0, 3, 0],
                                [7, 0, -3, 0],
                                [-7, 0, 0, 0],
                                [-7, 0, 3, 0],
                                [-7, 0, 3, 0],
                                [6, 0, 0, 0],
                                [6, 0, -3, 0],
                                [6, 0, -3, 0],
                                [-6, 0, 3, 0],
                                [-6, 0, 3, 0],
                                [5, 0, 0, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0]])

    # Using the tabulated values, compute the delta_longitude and
    # delta_obliquity.
    Xi = np.array([X0, X1, X2, X3, X4])    # a col mat in octave

    tabulated_argument = Y_terms.dot(np.transpose(Xi)) * (np.pi/180)

    delta_longitude = (nutation_terms[:, 0] + (nutation_terms[:, 1] * JCE)) * np.sin(tabulated_argument)
    delta_obliquity = (nutation_terms[:, 2] + (nutation_terms[:, 3] * JCE)) * np.cos(tabulated_argument)

    nutation = dict()    # init nutation dictionary
    # Nutation in longitude
    nutation['longitude'] = np.sum(delta_longitude) / 36000000

    # Nutation in obliquity
    nutation['obliquity'] = np.sum(delta_obliquity) / 36000000

    return nutation


def true_obliquity_calculation(julian, nutation):
    """Calculate true obliquity of the ecliptic. SPA Step 5."""


    p = np.atleast_2d([2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448])

    # mean_obliquity = polyval(p, julian.ephemeris_millenium/10);
    U = julian['ephemeris_millenium'] / 10
    mean_obliquity = p[0, 0] * np.power(U, 10) + p[0, 1] * np.power(U, 9) + \
                     p[0, 2] * np.power(U, 8) + p[0, 3] * np.power(U, 7) + \
                     p[0, 4] * np.power(U, 6) + p[0, 5] * np.power(U, 5) + \
                     p[0, 6] * np.power(U, 4) + p[0, 7] * np.power(U, 3) + \
                     p[0, 8] * np.power(U, 2) + p[0, 9] * U + p[0, 10]

    true_obliquity = (mean_obliquity/3600) + nutation['obliquity']

    return true_obliquity


def abberation_correction_calculation(earth_heliocentric_position):
    """Calculate aberration correction. SPA Step 6."""

    aberration_correction = -20.4898/(3600*earth_heliocentric_position['radius'])
    return aberration_correction


def apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction):
    """Calculate apparent sun longitude. SPA Step 7."""

    apparent_sun_longitude = sun_geocentric_position['longitude'] + nutation['longitude'] + aberration_correction
    return apparent_sun_longitude


def apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity):
    """Calculate apparent sidereal time at Greenwich. SPA Step 8."""

    JD = julian['day']
    JC = julian['century']

    # Mean sideral time, in degrees
    mean_stime = 280.46061837 + (360.98564736629*(JD-2451545)) + \
                 (0.000387933*np.power(JC, 2)) - \
                 (np.power(JC, 3)/38710000)

    # Limit the range to [0-360];
    mean_stime = set_to_range(mean_stime, 0, 360)

    apparent_stime_at_greenwich = mean_stime + (nutation['longitude'] * np.cos(true_obliquity * np.pi/180))
    return apparent_stime_at_greenwich


def sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """Calculate sun right ascension. SPA Step 9."""

    argument_numerator = (np.sin(apparent_sun_longitude * np.pi/180) * np.cos(true_obliquity * np.pi/180)) - \
        (np.tan(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180))
    argument_denominator = np.cos(apparent_sun_longitude * np.pi/180);

    sun_rigth_ascension = np.arctan2(argument_numerator, argument_denominator) * 180/np.pi
    # Limit the range to [0,360];
    sun_rigth_ascension = set_to_range(sun_rigth_ascension, 0, 360)
    return sun_rigth_ascension


def sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """Calculate geocentric sun declination. SPA Step 10."""

    argument = (np.sin(sun_geocentric_position['latitude'] * np.pi/180) * np.cos(true_obliquity * np.pi/180)) + \
        (np.cos(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180) *
         np.sin(apparent_sun_longitude * np.pi/180))

    sun_geocentric_declination = np.arcsin(argument) * 180/np.pi
    return sun_geocentric_declination


def observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension):
    """Calculate observer local hour angle. SPA Step 11."""

    observer_local_hour = apparent_stime_at_greenwich + location['longitude'] - sun_rigth_ascension
    # Set the range to [0-360]
    observer_local_hour = set_to_range(observer_local_hour, 0, 360)
    return observer_local_hour


def topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                       observer_local_hour, sun_rigth_ascension, sun_geocentric_declination):
    """Calculate topocentric sun position. SPA Step 12."""

    # Equatorial horizontal parallax of the sun in degrees
    eq_horizontal_parallax = 8.794 / (3600 * earth_heliocentric_position['radius'])

    # Term u, used in the following calculations (in radians)
    u = np.arctan(0.99664719 * np.tan(location['latitude'] * np.pi/180))

    # Term x, used in the following calculations
    x = np.cos(u) + ((location['altitude']/6378140) * np.cos(location['latitude'] * np.pi/180))

    # Term y, used in the following calculations
    y = (0.99664719 * np.sin(u)) + ((location['altitude']/6378140) * np.sin(location['latitude'] * np.pi/180))

    # Parallax in the sun rigth ascension (in radians)
    nominator = -x * np.sin(eq_horizontal_parallax * np.pi/180) * np.sin(observer_local_hour * np.pi/180)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (x * np.sin(eq_horizontal_parallax * np.pi/180) *
                                                                    np.cos(observer_local_hour * np.pi/180))
    sun_rigth_ascension_parallax = np.arctan2(nominator, denominator)
    # Conversion to degrees.
    topocentric_sun_position = dict()
    topocentric_sun_position['rigth_ascension_parallax'] = sun_rigth_ascension_parallax * 180/np.pi

    # Topocentric sun rigth ascension (in degrees)
    topocentric_sun_position['rigth_ascension'] = sun_rigth_ascension + (sun_rigth_ascension_parallax * 180/np.pi)

    # Topocentric sun declination (in degrees)
    nominator = (np.sin(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180))) * \
                np.cos(sun_rigth_ascension_parallax)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180)) * \
                                                                   np.cos(observer_local_hour * np.pi/180)
    topocentric_sun_position['declination'] = np.arctan2(nominator, denominator) * 180/np.pi
    return topocentric_sun_position


def topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position):
    """Calculate topocentric local hour angle. SPA Step 13."""

    topocentric_local_hour = observer_local_hour - topocentric_sun_position['rigth_ascension_parallax']
    return topocentric_local_hour


def sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour):
    """Calculate topocentric zenith and azimuth angles with atmospheric refraction. SPA Step 14."""


    # Topocentric elevation, without atmospheric refraction
    argument = (np.sin(location['latitude'] * np.pi/180) * np.sin(topocentric_sun_position['declination'] * np.pi/180)) + \
    (np.cos(location['latitude'] * np.pi/180) * np.cos(topocentric_sun_position['declination'] * np.pi/180) *
     np.cos(topocentric_local_hour * np.pi/180))
    true_elevation = np.arcsin(argument) * 180/np.pi

    # Atmospheric refraction correction (in degrees)
    argument = true_elevation + (10.3/(true_elevation + 5.11))
    refraction_corr = 1.02 / (60 * np.tan(argument * np.pi/180))

    # For exact pressure and temperature correction, use this,
    # with P the pressure in mbar amd T the temperature in Kelvins:
    # refraction_corr = (P/1010) * (283/T) * 1.02 / (60 * tan(argument * pi/180));

    # Apparent elevation
    apparent_elevation = true_elevation + refraction_corr

    sun = dict()
    sun['zenith'] = 90 - apparent_elevation

    # Topocentric azimuth angle. The +180 conversion is to pass from astronomer
    # notation (westward from south) to navigation notation (eastward from
    # north);
    nominator = np.sin(topocentric_local_hour * np.pi/180)
    denominator = (np.cos(topocentric_local_hour * np.pi/180) * np.sin(location['latitude'] * np.pi/180)) - \
    (np.tan(topocentric_sun_position['declination'] * np.pi/180) * np.cos(location['latitude'] * np.pi/180))
    sun['azimuth'] = (np.arctan2(nominator, denominator) * 180/np.pi) + 180

    # Set the range to [0-360]
    sun['azimuth'] = set_to_range(sun['azimuth'], 0, 360)
    return sun


def set_to_range(var, min_interval, max_interval):
    """
    Normalize angle to specified range.
    
    Args:
        var: Angle value
        min_interval: Minimum value (typically 0)
        max_interval: Maximum value (typically 360)
    
    Returns:
        Normalized angle in [min_interval, max_interval)
    """
    var = var - max_interval * np.floor(var/max_interval)

    if var < min_interval:
        var = var + max_interval
    return var

def Solweig_2015a_metdata_noload(inputdata, location, UTC):
    """
    Process meteorological data and calculate solar geometry for each time step.
    
    Computes solar position (altitude, azimuth) for all hours in the met data
    and organizes the data for SOLWEIG calculations.
    
    Args:
        inputdata (np.ndarray): Meteorological data array
        location (dict): Geographic location (latitude, longitude, altitude)
        UTC (float): UTC offset in hours
    
    Returns:
        tuple: (Met, altitude, azimuth, zen, jday, I0, CI, Twater, TgK, Tstart, 
                TgK_wall, Tstart_wall, firstdaytime, timeadd, timestepdec) containing
                processed meteorological forcing and solar geometry
    
    Notes:
        - Calculates solar position for every time step
        - Prepares data for SOLWEIG radiation calculations
        - Handles multiple time steps efficiently
    """

    met = inputdata
    data_len = len(met[:, 0])
    dectime = met[:, 1]+met[:, 2] / 24 + met[:, 3] / (60*24.)
    dectimemin = met[:, 3] / (60*24.)
    if data_len == 1:
        halftimestepdec = 0
    else:
        halftimestepdec = (dectime[1] - dectime[0]) / 2.
    time = dict()
    time['sec'] = 0
    time['UTC'] = UTC
    sunmaximum = 0.
    leafon1 = 97  #TODO this should change
    leafoff1 = 300  #TODO this should change

    # initialize matrices
    altitude = np.empty(shape=(1, data_len))
    azimuth = np.empty(shape=(1, data_len))
    zen = np.empty(shape=(1, data_len))
    jday = np.empty(shape=(1, data_len))
    YYYY = np.empty(shape=(1, data_len))
    leafon = np.empty(shape=(1, data_len))
    altmax = np.empty(shape=(1, data_len))

    sunmax = dict()

    for i, row in enumerate(met[:, 0]):
        if met[i, 1] == 221:
            test = 4
        YMD = datetime.datetime(int(met[i, 0]), 1, 1) + datetime.timedelta(int(met[i, 1]) - 1)
        # Finding maximum altitude in 15 min intervals (20141027)
        if (i == 0) or (np.mod(dectime[i], np.floor(dectime[i])) == 0):
            fifteen = 0.
            sunmaximum = -90.
            sunmax['zenith'] = 90.
            while sunmaximum <= 90. - sunmax['zenith']:
                sunmaximum = 90. - sunmax['zenith']
                fifteen = fifteen + 15. / 1440.
                HM = datetime.timedelta(days=(60*10)/1440.0 + fifteen)
                YMDHM = YMD + HM
                time['year'] = YMDHM.year
                time['month'] = YMDHM.month
                time['day'] = YMDHM.day
                time['hour'] = YMDHM.hour
                time['min'] = YMDHM.minute
                sunmax = sun_position(time,location)
        altmax[0, i] = sunmaximum

        half = datetime.timedelta(days=halftimestepdec)
        H = datetime.timedelta(hours=met[i, 2])
        M = datetime.timedelta(minutes=met[i, 3])
        YMDHM = YMD + H + M - half
        time['year'] = YMDHM.year
        time['month'] = YMDHM.month
        time['day'] = YMDHM.day
        time['hour'] = YMDHM.hour
        time['min'] = YMDHM.minute
        sun = sun_position(time, location)
        if (sun['zenith'] > 89.0) & (sun['zenith'] <= 90.0):    # Hopefully fixes weird values in Perez et al. when altitude < 1.0, i.e. close to sunrise/sunset
            sun['zenith'] = 89.0
        altitude[0, i] = 90. - sun['zenith']
        zen[0, i] = sun['zenith'] * (np.pi/180.)
        azimuth[0, i] = sun['azimuth']

        # day of year and check for leap year
        if calendar.isleap(time['year']):
            dayspermonth = np.atleast_2d([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        else:
            dayspermonth = np.atleast_2d([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        # jday[0, i] = np.sum(dayspermonth[0, 0:time['month']-1]) + time['day'] # bug when a new day 20191015
        YYYY[0, i] = met[i, 0]
        doy = YMD.timetuple().tm_yday
        jday[0, i] = doy
        if (doy > leafon1) | (doy < leafoff1):
            leafon[0, i] = 1
        else:
            leafon[0, i] = 0

    return YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax

