# Wet bulb temperature formulation from the atmos python package: https://github.com/AusClimateService/atmos/
# Reference: Warren, R.A., 2025. A consistent treatment of mixed‐phase saturation for atmospheric thermodynamics. 
# Quarterly Journal of the Royal Meteorological Society, 151(766), p.e4866.

# Balck globe temperature calculation derived from Shonk et al., 2026.
# Shonk, J.K., Blunn, L.P., Kumar, V., Wurtz, J. and Masson, V., 2026. 
#UCanWBGT: urban street canyon heat stress calculation for weather and climate models. 
#Quarterly Journal of the Royal Meteorological Society, p.e70082.

import numpy as np
from numba import vectorize
import torch

# ---------------------------------------------------------------------
# Constants actually needed for RH -> q and isobaric wet-bulb temperature
# ---------------------------------------------------------------------

# Specific gas constant for dry air (J/kg/K)
Rd = 287.0

# Specific gas constant for water vapour (J/kg/K)
Rv = 461.5

# Ratio of gas constants for dry air and water vapour
eps = Rd / Rv

# Isobaric specific heat of dry air (J/kg/K)
cpd = 1005.0

# Isobaric specific heat of water vapour (J/kg/K)
cpv = 2040.0  # optimised value from Ambaum (2020)

# Isobaric specific heat of liquid water (J/kg/K)
cpl = 4220.0  # triple-point value from Wagner and Pruß (2002)

# Isobaric specific heat of ice (J/kg/K)
cpi = 2097.0  # triple-point value from Feistel and Wagner (2006)

# Triple point temperature (K)
T0 = 273.16

# Saturation vapour pressure at the triple point (Pa)
es0 = 611.657  # Guildner et al. (1976)

# Latent heat of vaporisation at the triple point (J/kg)
Lv0 = 2.501e6  # Wagner and Pruß (2002)

# Latent heat of freezing at the triple point (J/kg)
Lf0 = 0.333e6  # included in original constants block

# Latent heat of sublimation at the triple point (J/kg)
Ls0 = Lv0 + Lf0

# Temperature above which all condensate is assumed to be liquid (K)
T_liq = 273.15

# Temperature below which all condensate is assumed to be ice (K)
T_ice = 253.15

# Precision for iterative temperature calculations (K)
precision = 0.001

# Maximum number of iterations for iterative calculations
max_n_iter = 20


def effective_specific_heat(q, qt=None, omega=0.0):
    """
    Computes effective isobaric specific heat for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction

    Returns:
        cpm (float or ndarray): effective isobaric specific heat (J/kg/K)

    """
    if qt is None:
        # (Eq. 17 from Warren 2025)
        cpm = (1 - q) * cpd + q * cpv
    else:
        # (Eq. 16 from Warren 2025)
        ql = (1 - omega) * (qt - q)
        qi = omega * (qt - q)
        cpm = (1 - qt) * cpd + q * cpv + ql * cpl + qi * cpi

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lv (float or ndarray): latent heat of vaporisation (J/kg)

    """
    # (Eq. 23 from Warren 2025)
    Lv = Lv0 + (cpv - cpl) * (T - T0)

    return Lv


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    # (Eq. 24 from Warren 2025)
    Ls = Ls0 + (cpv - cpi) * (T - T0)

    return Ls


def mixed_phase_latent_heat(T, omega):
    """
    Computes mixed-phase latent heat for a given temperature and ice fraction
    using equations from Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        omega (float or ndarray): ice fraction

    Returns:
        Lx (float or ndarray): mixed-phase latent heat (J/kg)

    """

    # Compute mixed-phase specific heat
    # (Eq. 30 from Warren 2025)
    cpx = (1 - omega) * cpl + omega * cpi

    # Compute mixed-phase latent heat at the triple point
    # (Eq. 31 from Warren 2025)
    Lx0 = (1 - omega) * Lv0 + omega * Ls0

    # Compute mixed-phase latent heat
    # (Eq. 32 from Warren 2025)
    Lx = Lx0 + (cpv - cpx) * (T - T0)

    return Lx


def vapour_pressure(p, q, qt=None):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    # (Eq. 15 from Warren 2025)
    if qt is None:
        e = p * q / (eps * (1 - q) + q)
    else:
        e = p * q / (eps * (1 - qt) + q)

    return e


def saturation_vapour_pressure(T, phase='liquid', omega=0.0):
    """
    Computes saturation vapour pressure (SVP) for a given temperature using
    equations from Ambaum (2020) and Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        es (float or ndarray): saturation vapour pressure (Pa)

    """

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute SVP over liquid water
        # (Eq. 26 from Warren 2025; cf. Eq. 13 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpl - cpv) / Rv)) * \
            np.exp((Lv0 / (Rv * T0)) - (Lv / (Rv * T)))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute SVP over ice
        # (Eq. 27 from Warren 2025; cf. Eq. 17 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpi - cpv) / Rv)) * \
            np.exp((Ls0 / (Rv * T0)) - (Ls / (Rv * T)))

    elif phase == 'mixed':

        # Compute mixed-phase specific heat
        # (Eq. 30 from Warren 2025)
        cpx = (1 - omega) * cpl + omega * cpi

        # Compute mixed-phase latent heat at the triple point
        # (Eq. 31 from Warren 2025)
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Compute mixed-phase latent heat
        # (Eq. 32 from Warren 2025)
        Lx = Lx0 + (cpv - cpx) * (T - T0)

        # Compute mixed-phase SVP
        # (Eq. 29 from Warren 2025)
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))

    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return es


def saturation_specific_humidity(p, T, qt=None, phase='liquid', omega=0.0):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        qs (float or ndarray): saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    if qt is None:
        # (Eq. 14 from Warren 2025, with qv = qt = qs and e = es)
        qs = eps * es / (p - (1 - eps) * es)
    else:
        # (Eq. 14 from Warren 2025, with qv = qs and e = es)
        qs = (1 - qt) * eps * es / (p - es)

    return qs


def relative_humidity(p, T, q, qt=None, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure,
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure(p, q, qt=qt)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


@vectorize(['float32(float32)', 'float64(float64)'], nopython=True)
def _lambertw(x):
    """
    Evaluates the lower branch of the Lambert-W function using PSEM
    approximation from Vazquez-Leal et al. (2019).

    Args:
        x (float): dependent variable

    Returns:
        y (float): Lambert W function (lower branch)

    """

    y = np.nan

    # W_{-1,1}
    if (x >= -0.3678794411714423) and (x < -0.34):
        a1 = -7.874564067684664
        a2 = -63.11879948166995
        a3 = -168.6110850408981
        a4 = -150.1089086912451
        b1 = 15.97679839497612
        b2 = 98.26612857148953
        b3 = 293.9558944644677
        b4 = 430.4471947824411
        b5 = 247.8576700279611
        alpha = x * (a1 + x * (a2 + x * (a3 + x * a4)))
        beta = 1. + x * (b1 + x * (b2 + x * (b3 + x * (b4 + x * b5))))
        y = (alpha / beta) * (x + np.exp(-1.)) - 1.

    # W_{-1,2}
    if (x >= -0.34) and (x < -0.1):
        a1 = -1362.78381643109
        a2 = -1386.04132570149
        a3 = 11892.1649836015
        a4 = 16904.0507511421
        b1 = 251.440197724561
        b2 = -1264.99554712435
        b3 = -5687.63429510978
        b4 = -2639.24130979048
        y = (x * (a1 + x * (a2 + x * (a3 + x * a4)))) / \
            (1. + x * (b1 + x * (b2 + x * (b3 + x * b4))))

    # W_{-1,3}
    if (x >= -0.1) and (x < 0.):
        a1 = 1.01999365162218
        a2 = -12.6917365519443
        a3 = -45.1506015092455
        b1 = -22.9809693297808
        b2 = -104.692066099727
        b3 = -95.2085341727207
        k0 = (x * (a1 + x * (a2 + x * a3))) / \
             (1. + x * (b1 + x * (b2 + x * b3)))
        k1 = np.log(-x)
        k2 = k1 - np.log(-k1) + np.log(-k1) / k1
        y = k0 + k2

    # Iterate once to improve accuracy
    z = np.log(x / y) - y
    t = 2. * (1. + y) * (1. + y + (2. / 3.) * z)
    e = (z / (1. + y)) * (t - z) / (t - 2. * z)
    y = y * (1. + e)

    return y


def ice_fraction(Tstar, phase='mixed'):
    """
    Computes ice fraction given temperature at saturation using nonlinear
    parameterisation of Warren (2025).
    """

    Tstar = np.atleast_1d(Tstar)

    if phase == 'liquid':
        omega = np.zeros_like(Tstar)
    elif phase == 'ice':
        omega = np.ones_like(Tstar)
    elif phase == 'mixed':
        omega = 0.5 * (1 - np.cos(np.pi * ((T_liq - Tstar) / (T_liq - T_ice))))
        omega[Tstar <= T_ice] = 1.0
        omega[Tstar >= T_liq] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if omega.size == 1:
        omega = omega.item()

    return omega


def ice_fraction_derivative(Tstar, phase='mixed'):
    """
    Computes derivative of ice fraction with respect to temperature at
    saturation using nonlinear parameterisation of Warren (2025).
    """

    Tstar = np.atleast_1d(Tstar)

    if phase == 'liquid' or phase == 'ice':
        domega_dTstar = np.zeros_like(Tstar)
    elif phase == 'mixed':
        domega_dTstar = -0.5 * (np.pi / (T_liq - T_ice)) * \
                np.sin(np.pi * ((T_liq - Tstar) / (T_liq - T_ice)))
        domega_dTstar[(Tstar <= T_ice) | (Tstar >= T_liq)] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if domega_dTstar.size == 1:
        domega_dTstar = domega_dTstar.item()

    return domega_dTstar


def dewpoint_temperature(p, T, q, phase='liquid', limit=True):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity.
    """

    if phase == 'liquid':

        RH = relative_humidity(p, T, q, phase='liquid')

        beta = (cpl - cpv) / Rv
        alpha = -(1 / beta) * (Lv0 + (cpl - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        W = _lambertw(fn)
        Td = alpha / W

    elif phase == 'ice':

        RH = relative_humidity(p, T, q, phase='ice')

        beta = (cpi - cpv) / Rv
        alpha = -(1 / beta) * (Ls0 + (cpi - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        W = _lambertw(fn)
        Td = alpha / W

    elif phase == 'mixed':

        Td = T

        converged = False
        count = 0
        while not converged:

            Td_prev = Td
            omega = ice_fraction(Td)
            RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

            cpx = (1 - omega) * cpl + omega * cpi
            Lx0 = (1 - omega) * Lv0 + omega * Ls0

            beta = (cpx - cpv) / Rv
            alpha = -(1 / beta) * (Lx0 + (cpx - cpv) * T0) / Rv
            fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
            W = _lambertw(fn)
            Td = alpha / W

            if np.nanmax(np.abs(Td - Td_prev)) < precision:
                converged = True
            else:
                count += 1
                if count == max_n_iter:
                    print(f"Saturation-point temperature not converged after {max_n_iter} iterations")
                    break

    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if limit:
        Td = np.minimum(Td, T)

    return Td


def specific_humidity_from_relative_humidity(p, T, rh, phase='liquid'):
    """
    Converts relative humidity in percent to specific humidity in kg/kg.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        rh (float or ndarray): relative humidity (%)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        q (float or ndarray): specific humidity (kg/kg)
    """

    RH = np.asarray(rh) / 100.0

    if phase == 'liquid':
        omega = 0.0
    elif phase == 'ice':
        omega = 1.0
    elif phase == 'mixed':
        omega = ice_fraction(T)
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es

    # Inversion of:
    # e = p * q / (eps * (1 - q) + q)
    q = eps * e / (p - (1 - eps) * e)

    return q


def isobaric_wet_bulb_temperature(
    p, T, q, phase='liquid', method='Romps', limit=True
    ):
    """
    Computes isobaric (a.k.a. thermodynamic) wet-bulb temperature using
    equations from Warren (2025) or Romps (2026).
    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q, phase=phase)

    # Initialise Tw using the "one-third rule" (Knox et al. 2017)
    Tw = T - (1 / 3) * (T - Td)

    if method == 'Warren':

        if phase == 'liquid':
            Lv_T = latent_heat_of_vaporisation(T)
        elif phase == 'ice':
            Ls_T = latent_heat_of_sublimation(T)
        elif phase == 'mixed':
            omega_T = ice_fraction(T)
            Lx_T = mixed_phase_latent_heat(T, omega_T)
        else:
            raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

        converged = False
        count = 0
        while not converged:

            Tw_prev = Tw

            if phase == 'liquid':
                omega_Tw = 0.0
            elif phase == 'ice':
                omega_Tw = 1.0
            elif phase == 'mixed':
                omega_Tw = ice_fraction(Tw)

            qs_Tw = saturation_specific_humidity(
                p, Tw, phase=phase, omega=omega_Tw
            )

            cpm_qs_Tw = effective_specific_heat(qs_Tw)

            if phase == 'liquid':

                Lv_Tw = latent_heat_of_vaporisation(Tw)

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                    Lv_Tw / (Rv * Tw**2)

                f = cpm_qs_Tw * (T - Tw) - Lv_T * (qs_Tw - q)
                fprime = ((cpv - cpd) * (T - Tw) - Lv_T) * dqs_dTw - cpm_qs_Tw

            elif phase == 'ice':

                Ls_Tw = latent_heat_of_sublimation(Tw)

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                    Ls_Tw / (Rv * Tw**2)

                f = cpm_qs_Tw * (T - Tw) - Ls_T * (qs_Tw - q)
                fprime = ((cpv - cpd) * (T - Tw) - Ls_T) * dqs_dTw - cpm_qs_Tw

            elif phase == 'mixed':

                domega_dTw = ice_fraction_derivative(Tw)
                Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)

                esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
                esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * (
                    Lx_Tw / (Rv * Tw**2) +
                    np.log(esi_Tw / esl_Tw) * domega_dTw
                )

                f = cpm_qs_Tw * (T - Tw) - Lx_T * (qs_Tw - q)
                fprime = ((cpv - cpd) * (T - Tw) - Lx_T) * dqs_dTw - cpm_qs_Tw

            Tw = Tw - f / fprime

            if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
                converged = True
            else:
                count += 1
                if count == max_n_iter:
                    print(f"Tw not converged after {max_n_iter} iterations")
                    break

    elif method == 'Romps':

        cpm = effective_specific_heat(q)

        converged = False
        count = 0
        while not converged:

            Tw_prev = Tw

            if phase == 'liquid':
                omega_Tw = 0.0
            elif phase == 'ice':
                omega_Tw = 1.0
            elif phase == 'mixed':
                omega_Tw = ice_fraction(Tw)

            qs_Tw = saturation_specific_humidity(
                p, Tw, phase=phase, omega=omega_Tw
            )

            if phase == 'liquid':

                Lv_Tw = latent_heat_of_vaporisation(Tw)

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                    Lv_Tw / (Rv * Tw**2)

                dLv_dTw = (cpv - cpl)

                f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lv_Tw
                fprime = -(cpm * (T - Tw) + Lv_Tw) * dqs_dTw - \
                    cpm * (1 - qs_Tw) - (qs_Tw - q) * dLv_dTw

            elif phase == 'ice':

                Ls_Tw = latent_heat_of_sublimation(Tw)

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                    Ls_Tw / (Rv * Tw**2)

                dLs_dTw = (cpv - cpi)

                f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Ls_Tw
                fprime = -(cpm * (T - Tw) + Ls_Tw) * dqs_dTw - \
                    cpm * (1 - qs_Tw) - (qs_Tw - q) * dLs_dTw

            elif phase == 'mixed':

                domega_dTw = ice_fraction_derivative(Tw)
                Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)
                cpx = (1 - omega_Tw) * cpl + omega_Tw * cpi

                dLx_dTw = (cpv - cpx) + (Tw - T0) * (cpl - cpi) * domega_dTw

                esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
                esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

                dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * (
                    Lx_Tw / (Rv * Tw**2) +
                    np.log(esi_Tw / esl_Tw) * domega_dTw
                )

                f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lx_Tw
                fprime = -(cpm * (T - Tw) + Lx_Tw) * dqs_dTw - \
                    cpm * (1 - qs_Tw) - (qs_Tw - q) * dLx_dTw

            Tw = Tw - f / fprime

            if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
                converged = True
            else:
                count += 1
                if count == max_n_iter:
                    print(f"Tw not converged after {max_n_iter} iterations")
                    break

    else:
        raise ValueError(
            "isobaric_method must be either 'Warren' or 'Romps'"
        )

    if limit:
        Tw = np.minimum(T, Tw)

    return Tw


def isobaric_wet_bulb_temperature_from_rh(
    p, T, rh, phase='liquid', method='Romps', limit=True
    ):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature,
    and relative humidity (%), converting RH internally to specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        rh (float or ndarray): relative humidity (%)
        phase (str, optional): condensed water phase
        method (str, optional): 'Warren' or 'Romps'
        limit (bool, optional): limit Tw <= T

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)
    """

    q = specific_humidity_from_relative_humidity(p, T, rh, phase=phase)
    Tw = isobaric_wet_bulb_temperature(
        p, T, q, phase=phase, method=method, limit=limit
    )

    return Tw - 273.15

def black_globe_temperature(hcg, Tmrt_mat, Ta_mat, emissivity=0.95):
    """
    Compute black globe temperature (Tg) in degC using the closed-form solution
    shown in the attached equations.

    Args:
        hcg (torch.Tensor): 2D convective heat transfer coefficient array
        Tmrt_mat (torch.Tensor): 2D mean radiant temperature array (degC)
        Ta_mat (torch.Tensor): 2D air temperature array (degC)
        emissivity (float, optional): globe emissivity. Default is 0.95.

    Returns:
        torch.Tensor: 2D black globe temperature array (degC)
    """

    sigma = 5.670374419e-8  # Stefan-Boltzmann constant

    dtype = hcg.dtype
    device = hcg.device

    emissivity = torch.as_tensor(emissivity, dtype=dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)

    a = hcg / (emissivity * sigma)
    b = (Tmrt_mat + 273.15)**4 + a * (Ta_mat + 273.15)

    m = 9.0 * a**2
    n = 27.0 * a**4
    p = 256.0 * b**3

    E = (m + 1.73205 * torch.sqrt(n + p))**(1.0 / 3.0)
    Q = 3.4943 * b
    k = 0.381571 * E - Q / E

    # Numerical protection against tiny negative values from floating-point roundoff
    k_safe = torch.clamp(k, min=torch.finfo(dtype).tiny)
    inner = 2.0 * a / torch.sqrt(k_safe) - k
    inner = torch.clamp(inner, min=0.0)

    i = 0.5 * torch.sqrt(inner)
    j = 0.5 * torch.sqrt(torch.clamp(k, min=0.0))

    Tg = i - j - 273.15

    return Tg
