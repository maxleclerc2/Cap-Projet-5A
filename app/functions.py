"""
This file defines some general functions used across the app.
"""
from os import walk
from os.path import abspath, join

from sgp4.api import days2mdhms
from tletools import TLE
from datetime import datetime
from random import randrange
import pandas as pd
from astropy.time import Time
from bokeh.embed import components
from beyond.frames.local import to_qsw
from math import pi, sin, isclose
from numpy import deg2rad, cross, exp, sqrt
import pytz
import logging

from .classes import SatClass, Date_Values
from .api import CompleteInfoFromFile
from .satellite_czml import satellite_czml, satellite
from .positions import GetPosVelSGP4, kep2car, GetIncXYEccXY, PROPAGATE, TEME2ECI, ECI2ECF


# -----------------------------------------------------------------------------
# General functions
# -----------------------------------------------------------------------------

def GetSatInfoFile(filename):
    """
    Retrieve the satellite's TLE from a file.
    2 lines and 3 lines TLEs are accepted.
    Each line of the TLEs must begin with a 1 or a 2 (or nothing for the sat name)
    
    Parameters
    ----------
    filename : string
        Path to the uploaded file

    Returns
    -------
    SatInfo : SatClass
        Satellite object
    """
    # Creation of the satellite object
    SatInfo = SatClass()

    # Loading the file
    file = open(filename)
    lines = file.readlines()

    SatName = "Unknown "  # Default name
    L_1 = []  # list of lines 1
    L_2 = []  # list of lines 2

    # Adding each line from the file to the corresponding list
    for line in lines:
        if line.startswith('1'):
            L_1.append(line[:69])
        elif line.startswith('2'):
            L_2.append(line[:69])
        else:
            SatName = line

    # List of TLEs
    TLES = []
    k = 0
    # Processing until the last TLE
    while k < len(L_1):
        tle = TLE.from_lines(SatName, L_1[k], L_2[k])
        TLES.append(tle)

        k += 1

    # Deleting the "\n" at the end of the name
    length = len(SatName)
    Name = SatName[:length - 1]

    # Retrieving the dates
    month, day, hour, minute, second = days2mdhms(TLES[0].epoch_year, TLES[0].epoch_day)
    DBY = int(TLES[0].epoch_year)  # DateBeginYear
    DBM = '%02d' % int(month)  # DateBeginMonth
    DBD = '%02d' % int(day)  # DateBeginDay

    lastTLE = len(TLES) - 1
    month, day, hour, minute, second = days2mdhms(TLES[lastTLE].epoch_year, TLES[lastTLE].epoch_day)
    DEY = int(TLES[lastTLE].epoch_year)  # DateEndYear
    DEM = '%02d' % int(month)  # DateEndMonth
    DED = '%02d' % int(day)  # DateEndDay

    # Reformat the dates (AAAA-MM-DD)
    DateBegin = "".join([str(DBY), "-", str(DBM), "-", str(DBD)])
    DateEnd = "".join([str(DEY), "-", str(DEM), "-", str(DED)])

    # Saving data
    SatInfo.TLES = TLES
    SatInfo.NAME = Name
    SatInfo.ID = TLES[0].norad
    SatInfo.SAT_NUM = 0
    SatInfo.L_1 = L_1
    SatInfo.L_2 = L_2
    SatInfo.DATE_BEGIN = DateBegin
    SatInfo.DATE_END = DateEnd
    SatInfo.FILE = filename
    
    SatInfo = CompleteInfoFromFile(SatInfo)
    
    SatInfo.COMPLETED = True

    return SatInfo


def czml(SatList, nb_sat, origin):
    """
    Function to generate a CZML file for the orbit simulation

    Parameters
    ----------
    SatList : list
        List of satellites
    nb_sat : int
        Number of satellites
    origin : string
        To know where to save the generated file
    """
    logging.info('Generating .czml file')

    tle_list = [[str(SatList[i].NAME), str(SatList[i].L_1[0]), str(SatList[i].L_2[0])] for i in range(nb_sat)]

    multiple_sats = []
    i = 0
    for tle in tle_list:
        sat = satellite(tle,
                        description='Station: ' + tle[0],
                        color=[randrange(256) for x in range(3)],
                        use_default_image=True,
                        start_time=datetime.strptime(SatList[i].DATE_BEGIN, '%Y-%m-%d').replace(tzinfo=pytz.UTC),
                        end_time=datetime.strptime(SatList[i].DATE_END, '%Y-%m-%d').replace(tzinfo=pytz.UTC),
                        show_label=True,
                        show_path=True,
                        L_1=SatList[i].L_1,
                        L_2=SatList[i].L_2
                        )
        multiple_sats.append(sat)
        i += 1

    czml_obj = satellite_czml(satellite_list=multiple_sats)
    czml_string = czml_obj.get_czml()

    with open("app/static/czml/" + origin + "/sat.czml", "w") as fo:
        fo.write("".join([czml_string]))

    logging.info('.czml file complete')


def create_plot_list(SatList, nb_sat, origin):
    """
    Plot each graph of the satellite.
    
    Parameters
    ----------
    SatList : list
        The list of satellites, the first one is the one with the plots
    nb_sat : int
        Number of satellites in the list
    origin : string
        "ai" or "threshold" to know where to save the plots
    """
    if origin == "proximity":
        SatList[0].make_plot_proximity(SatList)
        SatList[0].JS_PLOTS, SatList[0].DIV_PROXIMITY = components((SatList[0].PLOT_PROXIMITY))
    else:
        SatList[0].make_plot_list("Energy", SatList, nb_sat, origin)
        SatList[0].make_plot_list("Energy Delta", SatList, nb_sat, origin)
        SatList[0].make_plot_list("I", SatList, nb_sat, origin)
        SatList[0].make_plot_list("ECC", SatList, nb_sat, origin)
        SatList[0].make_plot_list("SMA", SatList, nb_sat, origin)
        SatList[0].make_plot_list("ARGP", SatList, nb_sat, origin)
        SatList[0].make_plot_list("RAAN", SatList, nb_sat, origin)
        SatList[0].make_plot_list("MA", SatList, nb_sat, origin)
        SatList[0].make_plot_list("Longitude", SatList, nb_sat, origin)
        SatList[0].make_plot_mahalanobis("INC_SMA", origin)
        SatList[0].make_plot_mahalanobis("LON_INC", origin)
    
        # JS common to all Bokeh plots
        SatList[0].JS_PLOTS, (
            SatList[0].DIV_ENERGY, SatList[0].DIV_DELTA_ENERGY, SatList[0].DIV_SMA, SatList[0].DIV_I, SatList[0].DIV_ECC,
            SatList[0].DIV_ARGP, SatList[0].DIV_RAAN,
            SatList[0].DIV_MA, SatList[0].DIV_LONGITUDE) = components((
                SatList[0].PLOT_ENERGY, SatList[0].PLOT_DELTA_ENERGY, SatList[0].PLOT_SMA,
                SatList[0].PLOT_I, SatList[0].PLOT_ECC, SatList[0].PLOT_ARGP, SatList[0].PLOT_RAAN, SatList[0].PLOT_MA,
                SatList[0].PLOT_LONGITUDE))


def ProcessTLES(SatInfo, date_filter, origin):
    """
    Common function to process TLES for all services

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    date_filter : int
        Integer to calculate the MJD between 2 TLES (for example, 
        0.5 means we want 12 hours seperating each TLES,
        0.25 means we want 6 hours seperating each TLES).
    origin : string
        Which service is calling the function.

    Raises
    ------
    RuntimeError
        If the origin is wrong.

    Returns
    -------
    None.

    """
    NbTLE = 0
    k = 0

    # Initialising each list from the first TLE
    # Date of the first TLE
    month, day, hour, minute, second = days2mdhms(SatInfo.TLES[0].epoch_year, SatInfo.TLES[0].epoch_day)

    # Separating seconds and milliseconds
    SEC, MILI = GetSecMilli(second)

    # Converting the date into MJD format
    t = Time(datetime(SatInfo.TLES[0].epoch_year, month, day, hour, minute, SEC, MILI))
    t.format = 'mjd'

    SatInfo.TEMPS_MJD.append(t.value)

    # Calling our function to fill the first data
    if origin == "Threshold":
        ProcessingThreshold(SatInfo, 0, month, day, hour, minute, second)
    elif origin == "AI":
        ProcessingAI(SatInfo, t, 0, month, day, hour, minute, second)
    elif origin == "Proximity":
        ProcessingProximity(SatInfo, 0, month, day, hour, minute, second)
    else:
        raise RuntimeError("Wrong origin")

    NbTLE = NbTLE + 1

    threads = []  # TODO Add multithreading

    # Processing until the last TLE
    while k < len(SatInfo.TLES):
        # Retrieving the date
        month, day, hour, minute, second = days2mdhms(SatInfo.TLES[k].epoch_year, SatInfo.TLES[k].epoch_day)

        t = Time(datetime(SatInfo.TLES[k].epoch_year, month, day, hour, minute, SEC, MILI))
        t.format = 'mjd'

        # Saving the date to select only TLEs that are at least 12 hours apart
        SatInfo.TEMPS_MJD.append(t.value)

        if t.value - SatInfo.TEMPS_MJD[-2] > date_filter:

            # TODO Multithreading might break the results & lists orders
            '''
            # Processing through threads
            th = Thread(target = threading_function, args = (SatInfo, k, month, day, hour, minute, second, ))
            th.start()
            threads.append(th)
            '''
            if origin == "Threshold":
                ProcessingThreshold(SatInfo, k, month, day, hour, minute, second)
            elif origin == "AI":
                ProcessingAI(SatInfo, t, k, month, day, hour, minute, second)
            elif origin == "Proximity":
                ProcessingProximity(SatInfo, k, month, day, hour, minute, second)
            else:
                raise RuntimeError("Wrong origin")
            NbTLE = NbTLE + 1

        # Looping through the list until the last TLE
        k += 1

    # TODO Waiting for all the threads to finish
    for th in threads:
        th.join()
    
    # Classification
    SatInfo.CLASSIFICATION = SatInfo.TLES[0].classification


def ProcessingThreshold(SatInfo, k, month, day, hour, minute, second):
    """
    Threshold function to process each TLE

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    k : int
        Index of the TLE
    month : int
        Month of the TLE
    day : int
        Day of the TLE
    hour : int
        Hour of the TLE
    minute : int
        Minute of the TLE
    second : int
        Second of the TLE (with milliseconds)
    """
    # Saving all datas in the Satellite object
    GetKep(SatInfo, SatInfo.TLES[k])
    
    # Calculate the energy only for the first satellite
    if SatInfo.SAT_NUM == 0:
        GetEnergy(SatInfo)

    # SGPA SATREC position
    #GetPosVelSGP4(SatInfo, k, month, day, hour, minute, SEC) # TODO remove ?
    # KEPLERIAN TO CARTESIAN position
    kep2car(SatInfo)
    
    SEC, MILI = GetSecMilli(second)
    date_actuelle = datetime(SatInfo.TLES[k].epoch_year, month, day, hour, minute, SEC, MILI)
    SatInfo.DATE_TLE.append(date_actuelle)
    
    # Saving the values for ECI2ECF
    values = Date_Values(k, month, day, hour, minute, SEC)
    SatInfo.DATE_VALUES.append(values)


def ProcessingAI(SatInfo, t, k, month, day, hour, minute, second):
    """
    AI function to process each TLE

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    t : datetime
        Date of the TLE
    k : int
        Index of the TLE
    month : int
        Month of the TLE
    day : int
        Day of the TLE
    hour : int
        Hour of the TLE
    minute : int
        Minute of the TLE
    second : int
        Second of the TLE (with milliseconds)
    """
    # Adding the date for the AI
    SatInfo.DATE_MJD.append(t.value)
    
    # Saving all datas in the Satellite object
    GetKep(SatInfo, SatInfo.TLES[k])
    GetIncXYEccXY(SatInfo)
    GetEnergy(SatInfo)
    
    # SGPA SATREC position
    #GetPosVelSGP4(SatInfo, k, month, day, hour, minute, SEC) # TODO remove ?
    # KEPLERIAN TO CARTESIAN position
    kep2car(SatInfo)

    SEC, MILI = GetSecMilli(second)
    date_now = datetime(SatInfo.TLES[k].epoch_year, month, day, hour, minute, SEC, MILI)
    SatInfo.DATE_TLE.append(date_now)
    
    # Saving the values for ECI2ECF
    values = Date_Values(k, month, day, hour, minute, SEC)
    SatInfo.DATE_VALUES.append(values)

    if k != 0:
        teme_pos, teme_vel = PROPAGATE(SatInfo.L_1[SatInfo.LAST_TLE], SatInfo.L_2[SatInfo.LAST_TLE], date_now)
        
        SatInfo.TMP_TEME_POSITION.append(teme_pos)
        SatInfo.TMP_TEME_SPEED.append(teme_vel)
    
    SatInfo.LAST_TLE = k


def ProcessingProximity(SatInfo, k, month, day, hour, minute, second):
    """
    Proximity function to process each TLE

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    k : int
        Index of the TLE
    month : int
        Month of the TLE
    day : int
        Day of the TLE
    hour : int
        Hour of the TLE
    minute : int
        Minute of the TLE
    second : int
        Second of the TLE (with milliseconds)
    """
    # Saving all datas in the Satellite object
    GetKep(SatInfo, SatInfo.TLES[k])
    
    # KEPLERIAN TO CARTESIAN position
    kep2car(SatInfo)

    SEC, MILI = GetSecMilli(second)
    date_actuelle = datetime(SatInfo.TLES[k].epoch_year, month, day, hour, minute, SEC, MILI)
    SatInfo.DATE_TLE.append(date_actuelle)


def GetSecMilli(second):
    """
    Seperates seconds and milliseconds.

    Parameters
    ----------
    second : int
        Seconds var.

    Returns
    -------
    SEC : int
        Seconds.
    MILLI : int
        Milliseconds.

    """
    try:
        p = str(second)
        SEC = int(p.partition(".")[0])
        MILLI = int(p.partition(".")[2])
    except Exception: # In case there are no seconds
        SEC = 00
        MILLI = 0000
    
    return SEC, MILLI


def GetKep(SatInfo, TLE):
    """
    Saves the keplerian parameters of the satellite

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.
    TLE : tletools.TLE
        The current TLE.

    Returns
    -------
    None.

    """
    SatInfo.MEAN_MOTION.append(TLE.n)
    SatInfo.INCLINATION.append(float(TLE.inc))
    SatInfo.ECC.append(float(TLE.ecc))
    SatInfo.ARGP.append(float(TLE.argp))
    SatInfo.RAAN.append(float(TLE.raan))
    SatInfo.MEAN_ANOMALY.append(float(TLE.M))
    SatInfo.SMA.append(TLE.a)


def GetEnergy(SatInfo):
    """
    Calculates the energy of the satellite

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    MU = 3.986004415 * 10 ** 14 # mu = 398 600,441 km**3/s**2
    SatInfo.ENERGY.append((-1 / 2) * (MU * 2 * pi * (float(SatInfo.MEAN_MOTION[-1]) / 86400)) ** (2 / 3))


def ConvertCoord(SatInfo):
    """
    Converts the coordinates from TEME to ECI then ECI to ECF

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    SatInfo.ECI_POSITION, SatInfo.ECI_SPEED = TEME2ECI(SatInfo.TEME_POSITION, SatInfo.TEME_SPEED)
    SatInfo.POS_X = [item[0] for item in SatInfo.ECI_POSITION]
    SatInfo.POS_Y = [item[1] for item in SatInfo.ECI_POSITION]
    SatInfo.POS_Z = [item[2] for item in SatInfo.ECI_POSITION]
    
    SatInfo.SPEED_X = [item[0] for item in SatInfo.ECI_SPEED]
    SatInfo.SPEED_Y = [item[1] for item in SatInfo.ECI_SPEED]
    SatInfo.SPEED_Z = [item[2] for item in SatInfo.ECI_SPEED]
    
    ECI2ECF(SatInfo)


def GetQSW(SatInfo):
    """
    Get the Along Track and Out of Plane from the QSW

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    # Beggins at the second coord
    eci_pos, eci_vel = TEME2ECI(SatInfo.TMP_TEME_POSITION, SatInfo.TMP_TEME_SPEED)
    
    for i in range(len(SatInfo.POS_X) - 1):
        delta_pos_meter = [(SatInfo.POS_X[i + 1] - eci_pos[i][0]) * 1000,
                           (SatInfo.POS_Y[i + 1] - eci_pos[i][1]) * 1000,
                           (SatInfo.POS_Z[i + 1] - eci_pos[i][2]) * 1000]
        SatInfo.DELTA_QSW.append(to_qsw([SatInfo.POS_X[i + 1] * 1000,
                                         SatInfo.POS_Y[i + 1] * 1000,
                                         SatInfo.POS_Z[i + 1] * 1000,
                                         SatInfo.SPEED_X[i + 1] * 1000,
                                         SatInfo.SPEED_Y[i + 1] * 1000,
                                         SatInfo.SPEED_Z[i + 1] * 1000]).T @ delta_pos_meter)
        
    SatInfo.DELTA_ALONG_TRACK = [item[1] for item in SatInfo.DELTA_QSW]
    SatInfo.DELTA_OUT_OF_PLANE = [item[2] for item in SatInfo.DELTA_QSW]


def GetMeanCross(SatInfo):
    """
    Get the Mean Cross of each coordinates

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    for i in range(len(SatInfo.POS_X)):
        SatInfo.MEAN_CROSS.append(cross(tuple([SatInfo.POS_X[i] * 1000,
                                               SatInfo.POS_Y[i] * 1000,
                                               SatInfo.POS_Z[i] * 1000]),
                                        tuple([SatInfo.SPEED_X[i] * 1000,
                                               SatInfo.SPEED_Y[i] * 1000,
                                               SatInfo.SPEED_Z[i] * 1000])))


def GetOrbType(SatInfo, origin):
    """
    Get the orbit type of the satellite depending of the mean motion of the first TLE

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.
    origin : string
        Parent function calling it.

    Returns
    -------
    None.

    """
    # Orbit type depending of n
    if SatInfo.TLES[0].n < 1:
        SatInfo.ORBIT = "HEO"
    elif 1.003 < SatInfo.TLES[0].n < 11:
        SatInfo.ORBIT = "MEO"
    elif 11 <= SatInfo.TLES[0].n < 17:
        SatInfo.ORBIT = "LEO"
    elif isclose(SatInfo.TLES[0].n, 1.0027, rel_tol=1e-3):
        SatInfo.ORBIT = "GEO"
    else:
        # Adding n to the log to know its value for later updates
        logging.error(origin + " - Error during orbit type recognition:")
        logging.error("".join([SatInfo.NAME, " has a n of ", str(SatInfo.TLES[0].n)]))
        SatInfo.MESSAGES.append("".join(["Error during orbit type recognition for ",
                                         SatInfo.NAME, ". Please check the logs."]))


def GetInitialMass(SatInfo, mass):
    """
    Get the mass of the satellite at the beggining date

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.
    mass : float
        Mass of the satellite at launch.

    Returns
    -------
    mass : float
        Calculated mass at the beggining date.

    """
    g0 = 9.81
    propulsion_chimique = 200

    difference_annee = int(SatInfo.DATE_BEGIN[0:4]) - int(SatInfo.LAUNCH_DATE[0:4])
    difference_mois = int(SatInfo.DATE_BEGIN[5:7]) - int(SatInfo.LAUNCH_DATE[5:7])
    coefficient = (difference_annee * 12 + difference_mois) / 12
    mass = mass * exp(-(50 * coefficient) / (g0 * propulsion_chimique))
    
    return mass


def GetDeltaVDeltaM(SatInfo, i, j, mass):
    """
    Get the mass of the satellite after a maneuver

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.
    i : int
        Index of the TLE (maneuver).
    j : int
        1 or 2. Gap to be sure we are at the end of the list.
    mass : float
        Mass before the maneuver.

    Returns
    -------
    mass : float
        Calculated mass at the given TLE.

    """
    mu = 398600
    g0 = 9.81
    chemical_propulsion = 200
    
    # Delta Va and Vi (SMA & inclination)
    SatInfo.Delta_Va.append((SatInfo.SMA[i + j] - SatInfo.SMA[i]) * sqrt(mu / SatInfo.SMA[i]) / 2 / SatInfo.SMA[i] * 1000)
    SatInfo.Delta_Vi.append(2 * sqrt(mu / SatInfo.SMA[i]) * sin(deg2rad((SatInfo.INCLINATION[i + j] - SatInfo.INCLINATION[i]) / 2) * 1000))

    # Delta V
    SatInfo.Delta_V.append(sqrt(SatInfo.Delta_Va[-1] * SatInfo.Delta_Va[-1] + SatInfo.Delta_Vi[-1] * SatInfo.Delta_Vi[-1]))

    # Delta M
    SatInfo.Delta_M.append(mass * (1 - exp(-SatInfo.Delta_V[-1] / (g0 * chemical_propulsion))))
    mass = mass - SatInfo.Delta_M[-1]
    
    return mass


def CreateLix(SatInfo):
    """
    Generates the dataframe from the satellite

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    Lix : dataframe
        The pandas dataframe object.

    """
    Lix = pd.DataFrame(
        columns=['Date', 'Delta-t', 'Delta Inc X', 'Delta Inc Y', 'Delta Ecc X',
                 'Delta Ecc Y', 'Delta Energy', 'Delta Energy Previous',
                 'Delta Energy Next', 'Delta Moment'])
    Lix['Date'] = SatInfo.DATE_TLE
    Lix['Delta-t'] = SatInfo.DELTA_T
    Lix['Delta Inc X'] = SatInfo.DELTA_IX
    Lix['Delta Inc Y'] = SatInfo.DELTA_IY
    Lix['Delta Ecc X'] = SatInfo.DELTA_EX
    Lix['Delta Ecc Y'] = SatInfo.DELTA_EY
    Lix['Delta Energy'] = SatInfo.ENERGY_DELTA
    Lix['Delta Energy Previous'] = SatInfo.DELTA_ENERGY_PREV
    Lix['Delta Energy Next'] = SatInfo.DELTA_ENERGY_NEXT
    Lix['Delta Moment'] = SatInfo.DELTA_MOMENT
    Lix['Delta Out-of-plane'] = SatInfo.DELTA_OUT_OF_PLANE
    Lix['Delta Along-track'] = SatInfo.DELTA_ALONG_TRACK
    Lix['Delta Along-track avg'] = SatInfo.DELTA_ALONG_TRACK_AVG
    
    return Lix


def AddFilesToZip(zip_file, path):
    for dirname, subdirs, files in walk(path):
        for filename in files:
            absname = abspath(join(dirname, filename))
            arcname = absname[len(path) + 1:]
            zip_file.write(absname, arcname)
