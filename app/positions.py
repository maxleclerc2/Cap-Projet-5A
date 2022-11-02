"""
This file contains the functions processing the coordinates of the satellite
"""

from sgp4.api import Satrec, SGP4_ERRORS
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CIRS, TEME, cartesian_to_spherical
from pymap3d import eci2ecef
from datetime import datetime
import logging

from numpy import cos, sin, rad2deg, deg2rad, sqrt, power, square, floor, pi, cbrt, nan, squeeze


def kep2car(SatInfo):
    """
    CAR = PV
    Returns the TEME positions from the keplerian parameters

    Parameters
    ----------
    SatInfo : SatClass
        The satellite to work with. Contains the keplerian parameters.

    Returns
    -------
    None.

    """
    a = SatInfo.SMA[-1]
    e = SatInfo.ECC[-1]
    
    # E = eccentric argument of latitude
    E = M2E(e, deg2rad(SatInfo.MEAN_ANOMALY[-1]))
    cosE = cos(E)
    sinE = sin(E)
    
    # Other needed quantities
    r = a * (1 - e * cosE)
    n = sqrt(wgs72.mu / power(a, 3))
    eta = sqrt(1 - square(e))
    
    # Position and velocity in "natural" frame
    X = a * (cosE - e)
    Y = a * eta * sinE
    
    VX = -n * square(a) / r * sinE
    VY = n * square(a) / r * eta * cosE
    
    # Position and velocity in cartesian frame
    # Using direct formulas is the fastest way to compute matrix R:
    cp = cos(deg2rad(SatInfo.ARGP[-1]))
    sp = sin(deg2rad(SatInfo.ARGP[-1]))
    cg = cos(deg2rad(SatInfo.RAAN[-1]))
    sg = sin(deg2rad(SatInfo.RAAN[-1]))
    ci = cos(deg2rad(SatInfo.INCLINATION[-1]))
    si = sin(deg2rad(SatInfo.INCLINATION[-1]))
    # P = first column of matrix R
    # Q = second column of matrix R
    P = [(cg * cp-sg * ci * sp), (sg * cp+cg * ci * sp), (si * sp)]
    Q = [(-cg * sp-sg * ci * cp), (-sg * sp+cg * ci * cp), (si * cp)]
    
    teme_pos = [P[0] * X + Q[0] * Y,
           P[1] * X + Q[1] * Y,
           P[2] * X + Q[2] * Y]
    teme_vel = [P[0] * VX + Q[0] * VY,
           P[1] * VX + Q[1] * VY,
           P[2] * VX + Q[2] * VY]
    
    SatInfo.TEME_POSITION.append(teme_pos)
    SatInfo.TEME_SPEED.append(teme_vel)


def M2E(e, M):
    """
    Mean anomaly to eccentric anomaly (Kepler's equation)
                                       
    Solution of kepler's equation (M to E) - Ellipse only
    (EQ)  M = E - e*sin(E)  

    Parameters
    ----------
    e : float
        Eccentricity.
    M : float
        Mean anomaly.

    Returns
    -------
    E : float
        Eccentric anomaly.

    """
    reducedM = rMod(M, -pi, pi)
    E = initialGuess(e, reducedM)
    
    e1 = 1 - e
    noCancellationRisk = (e1 + E * E / 6) >= 0.1
    
    for i in range(2):
        fdd  = e * sin(E)
        fddd = e * cos(E)
        
        if noCancellationRisk:
            f = (E - fdd) - reducedM
            fd = 1 - fddd
        else:
            # NB: CL__E_esinE is an accurate computation of E-e*sin(E) 
            # for e close  to 1 and E close to 0
            f = esinE(e,E) - reducedM
            s = sin(0.5 * E)
            fd = e1 + 2 * e * s * s
        
        dee = f * fd / (0.5 * f * fdd - fd * fd)
        
        # Update eccentric anomaly, using expressions that limit underflow problems
        w = fd + 0.5 * dee * (fdd + dee * fddd / 3)
        fd = fd + dee * (fdd + 0.5 * dee * fddd)
        E = E - (f - dee * (fd - w)) / fd
    
    # Expand the result back to original range
    E = E + (M - reducedM)
    
    return E


def rMod(x, a, b):
    """
    Modulo with result in range

    Parameters
    ----------
    x : float
        Value to modulo.
    a : float
        Below value.
    b : float
        Above value.

    Returns
    -------
    y : float
        Processed x.
    """
    delta = b - a
    nrev = floor((x - a) / delta)
    y = x - nrev * delta
    
    return y


def initialGuess(e, reducedM):
    """
    Compute initial guess according to A. W. Odell and R. H. Gooding S12 starter
    
    A,B =  Coefficients to compute Kepler equation solver starter

    Parameters
    ----------
    e : float
        Eccentricity.
    reducedM : float
        Mean anomaly after modulo.

    Raises
    ------
    RuntimeError
        Invalid reducedM. Please check the value.

    Returns
    -------
    E : float
        Initial solution of the equation.

    """
    k1 = 3 * pi + 2
    k2 = pi - 1
    k3 = 6 * pi - 1
    A  = 3 * k2 * k2 / k1
    B  = k3 * k3 / (6 * k1)

    E = 0
    if abs(reducedM) < 1.0 / 6.0:
        E = reducedM + e * (cbrt(6 * reducedM) - reducedM)
    elif abs(reducedM) >= 1.0 / 6.0 and reducedM < 0:
        w = pi*e + reducedM
        E = reducedM + e * (A*w / (B*w - w) - pi*e - reducedM)
    elif abs(reducedM) >= 1.0 / 6.0 and reducedM >= 0:
        w = pi*e - reducedM
        E = reducedM + e * (pi*e - A*w / (B*w - w) - reducedM)
    else:
        raise RuntimeError("Invalid reducedM")
    
    return E


def esinE(e, E):
    """
    Accurate computation of E - e*sin(E).
    (useful when E is close to 0 and e close to 1,
     i.e. near the perigee of almost parabolic orbits)

    Parameters
    ----------
    e : float
        Eccentricity.
    E : float
        Solution of the equation.

    Returns
    -------
    x : float
        DESCRIPTION.

    """
    x = (1 - e) * sin(E)
    mE2 = -E * E
    term = E
    d = 0
    x0 = nan * [1] * x
    
    K = range(len(x))  # indices to be dealt with
    
    iter = 0
    nb_max_iter = 20        # max number of iterations
    
    while (K != [] and iter <= nb_max_iter):
        d += 2
        term = term * mE2 / d * (d + 1)
        x0 = x
        x = x - term
        # the inequality test below IS intentional and should NOT be replaced by a check with a small tolerance
        K = K.index(x != x0)
        iter += 1
    
    if K != []:
        logging.WARNING('Maximum number of iterations reached')
     
    return x


def TEME2ECI(teme_pos, teme_vel):
    """
    CIRS = ECI
    Converts the TEME positions to ECI

    Parameters
    ----------
    teme_pos : array
        List of all the [x, y, z] positions (TEME).
    teme_vel : array
        List of all the [vx, vy, vz] velocities (TEME).

    Returns
    -------
    eci_pos : array
        List of all the [x, y, z] positions (ECI).
    eci_vel : array
        List of all the [vx, vy, vz] velocities (ECI).

    """
    
    now = Time('2018-03-14 23:48:00') #  We don't care of the date used
    pos = teme_pos*u.km
    vel = teme_vel*u.km/u.s
    
    teme = TEME(x=[item[0] for item in pos],
                y=[item[1] for item in pos],
                z=[item[2] for item in pos],
                v_x=[item[0] for item in vel],
                v_y=[item[1] for item in vel],
                v_z=[item[2] for item in vel],
                representation_type='cartesian',
                differential_type='cartesian',
                obstime=now)
    cirs = teme.transform_to(CIRS(obstime=now))
    
    eci_pos = []
    eci_vel = []
    for i in range(len(cirs.cartesian)):
        eci_pos.append([cirs.cartesian[i].x.value,
                        cirs.cartesian[i].y.value,
                        cirs.cartesian[i].z.value])
        eci_vel.append([cirs.cartesian[i].differentials.get("s").d_x.value,
                        cirs.cartesian[i].differentials.get("s").d_y.value,
                        cirs.cartesian[i].differentials.get("s").d_z.value])

    return eci_pos, eci_vel


def ECI2ECF(SatInfo):
    """
    Converts the ECI positions to ECF

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    i = 0
    # conversion eci to ecef then cart_to_sphe
    for values in SatInfo.DATE_VALUES:
        date_astropy = datetime(SatInfo.TLES[values.k].epoch_year, values.month, values.day, values.hour, values.minute, values.SEC)
        stringbis = eci2ecef(SatInfo.ECI_POSITION[i][0], SatInfo.ECI_POSITION[i][1], SatInfo.ECI_POSITION[i][2], date_astropy)  # here eci to ecef
        SatInfo.POS_X_ECEF.append(stringbis[0])
        SatInfo.POS_Y_ECEF.append(stringbis[1])  # phi = lat , theta=long , rho = distance
        SatInfo.POS_Z_ECEF.append(stringbis[2])
        i += 1
    # positions now in ecef
    # saved all cartesian parameters


def GetPosVelSGP4(SatInfo, k, month, day, hour, minute, SEC):
    """
    NOT OPTIMIZED - UNUSED, PROBABLY TO REMOVE

    Parameters
    ----------
    SatInfo : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    hour : TYPE
        DESCRIPTION.
    minute : TYPE
        DESCRIPTION.
    SEC : TYPE
        DESCRIPTION.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    satellite = Satrec.twoline2rv(SatInfo.L_1[k], SatInfo.L_2[k])

    date_astropy = datetime(SatInfo.TLES[k].epoch_year, month, day, hour, minute, SEC)

    t = Time(date_astropy, format='datetime')
    error_code, teme_p, teme_v = satellite.sgp4(t.jd1, t.jd2)  # in km and km/s
    if error_code != 0:
        raise RuntimeError(SGP4_ERRORS[error_code])

    SatInfo.TEME_POSITION.append(teme_p)
    SatInfo.TEME_SPEED.append(teme_v)
    
    eci_p, eci_v = TEME2ECI(teme_p, teme_v)
    SatInfo.POS_X.append(eci_p[0])  # in km and km/s
    SatInfo.POS_Y.append(eci_p[1])
    SatInfo.POS_Z.append(eci_p[2])
    SatInfo.SPEED_X.append(eci_v[0])  # in km
    SatInfo.SPEED_Y.append(eci_v[1])
    SatInfo.SPEED_Z.append(eci_v[2])
    
    ECI2ECF(SatInfo, k, month, day, hour, minute, SEC, eci_p)


def GetIncXYEccXY(SatInfo):
    """
    Calculates the Inc X & Y and the Ecc X & Y of the last TLE from the keplerian parameters

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    SatInfo.INCLINATION_X.append(sin(deg2rad(float(SatInfo.INCLINATION[-1])) / 2) * cos(deg2rad(float(SatInfo.RAAN[-1]))))
    SatInfo.INCLINATION_Y.append(sin(deg2rad(float(SatInfo.INCLINATION[-1])) / 2) * sin(deg2rad(float(SatInfo.RAAN[-1]))))
    SatInfo.ECC_X.append(float(SatInfo.ECC[-1]) * cos(deg2rad(float(SatInfo.ARGP[-1])) + deg2rad(float(SatInfo.RAAN[-1]))))
    SatInfo.ECC_Y.append(float(SatInfo.ECC[-1]) * sin(deg2rad(float(SatInfo.ARGP[-1])) + deg2rad(float(SatInfo.RAAN[-1]))))
    

def GetLatLong(SatInfo):
    """
    Get the lattitude and longitude of the satellite

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    # Calculate the longitude
    for i in range(len(SatInfo.POS_X)):
        string = cartesian_to_spherical(SatInfo.POS_X_ECEF[i], SatInfo.POS_Y_ECEF[i], SatInfo.POS_Z_ECEF[i])
        # Using Astropy for the conversion

        SatInfo.D_ECEF.append(string[0])
        SatInfo.LATITUDE_ECEF.append(rad2deg(float(string[1] / u.rad)))  # Deleting the unit Rad which is automatically added and -
        SatInfo.LONGITUDE_ECEF.append(rad2deg(float(string[2] / u.rad)))  # - converting into Degrees

    SatInfo.D_ECEF = squeeze(SatInfo.D_ECEF)  # After deleting the unit, we have something like (<Quantity> 1000 ), deleting it -
    SatInfo.LATITUDE_ECEF = squeeze(SatInfo.LATITUDE_ECEF)  # - with np.squeeze to have [ 1000 ]
    SatInfo.LONGITUDE_ECEF = squeeze(SatInfo.LONGITUDE_ECEF)


def PROPAGATE(L1, L2, date_end):
    """
    Propagates the TLE to the end date. Positions in TEME.

    Parameters
    ----------
    L1 : string
        Line 1 of the TLE.
    L2 : string
        Line 2 of the TLE.
    date_end : datetime
        End date of the propagation.

    Returns
    -------
    teme_pos : list
        List with the shape [x, y, z].
    teme_vel : list
        List with the shape [vx, vy, vz].

    """
    tle_obj = twoline2rv(L1, L2, wgs72)
    teme_pos, teme_vel = tle_obj.propagate(date_end.year, date_end.month,
                                           date_end.day, date_end.hour,
                                           date_end.minute, date_end.second)
    
    return teme_pos, teme_vel