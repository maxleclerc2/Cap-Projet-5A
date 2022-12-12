"""
This file contains the code for processing with the threshold defined by the user.

These functions are used in routes.py
"""

from statistics import pstdev

import logging

from app.code.utils.positions import GetLatLong
from app.code.utils.functions import ConvertCoord, GetOrbType, GetInitialMass, GetDeltaVDeltaM, ProcessTLES

# -----------------------------------------------------------------------------
# Functions for the threshold search
# -----------------------------------------------------------------------------


def GetTleInfo(SatInfo, Seuil="", ManualVal=0, AutoVal=0):
    """
    Main function to retrieve the Satellite's data

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    Seuil : string
        Threshold method (automatic or manual)
    ManualVal : int
        Threshold limit in manual mode
    AutoVal : int
        Threshold limit in automatic mode

    Returns
    -------
    SatInfo : SatClass
        Filled satellite object
    """
    # Doing the threshold only for the first satellite
    if SatInfo.SAT_NUM == 0:
        # Adding the method to the satellite object
        # Selecting the matching value
        SatInfo.LIMIT_TYPE = Seuil
        if Seuil == "auto":
            SatInfo.LIMIT_VALUE = AutoVal
        else:
            SatInfo.LIMIT_VALUE = float(ManualVal)

    # Processing every 12 hours
    ProcessTLES(SatInfo, 0.5, "Threshold")
    # Convert TEME to ECI then ECI to ECF
    ConvertCoord(SatInfo)
    #♠ Get Lattitude and Longitude
    GetLatLong(SatInfo)
    # Get orbit type
    GetOrbType(SatInfo, "Threshold")

    # Detecting maneuvers only for the first satellite
    if SatInfo.SAT_NUM == 0:
        # Difference of energy between 2 TLEs
        for i in range(len(SatInfo.ENERGY) - 1):
            # Energy Delta
            SatInfo.ENERGY_DELTA.append(SatInfo.ENERGY[i + 1] - SatInfo.ENERGY[i])

        SatInfo.DATE_ENERGY_DELTA = SatInfo.DATE_TLE.copy()  # Copying the dates with 1 less date
        SatInfo.DATE_ENERGY_DELTA.pop(0)  # Removing the first date because there is no Delta

        # Calculate with the automatic threshold
        if Seuil == "auto":
            # Checking that there are multiple values
            try:
                # Calculate the Sigma of the Energy Delta
                Sigma_delta_energie = pstdev(SatInfo.ENERGY_DELTA)
            except:
                logging.error("Threshold - Error with Energy Delta")
                SatInfo.ERROR_MESSAGE = "".join(["Unable to calculate the Energy Delta within the dates ",
                                                 SatInfo.DATE_BEGIN, " - ",
                                                 SatInfo.DATE_END,
                                                 ". Please chose a larger date range after the launch date: ",
                                                 SatInfo.LAUNCH_DATE])
                return SatInfo

            SatInfo.SIGMA_INF = - int(AutoVal) * Sigma_delta_energie
            SatInfo.SIGMA_SUP = int(AutoVal) * Sigma_delta_energie
        # Calculate with the manual threshold
        else:
            SatInfo.SIGMA_INF = - int(ManualVal)
            SatInfo.SIGMA_SUP = int(ManualVal)

        # Detecting maneuvers that have an energy greater than the threshold
        mass = float(SatInfo.MASS[:-2])
        # We will calculate the mass of fuel used between the launch date of the satellite and the start of the values
        if SatInfo.ORBIT == "GEO":
            GetInitialMass(SatInfo, mass)
            
        SatInfo.MASS_BEGIN = mass

        for i in range(len(SatInfo.ENERGY_DELTA)):
            if SatInfo.ENERGY_DELTA[i] < SatInfo.SIGMA_INF or SatInfo.ENERGY_DELTA[i] > SatInfo.SIGMA_SUP:
                # Keeping the Delta
                SatInfo.DETECTION_MANEUVER.append(SatInfo.ENERGY_DELTA[i])
                # Keeping the maneuver date
                SatInfo.DATE_MANEUVER.append(SatInfo.DATE_TLE[i + 1])
                
                # Taking a gap of 1 after the maneuver to be sure it is finished in case we are not at the end of the list
                if i + 2 < len(SatInfo.SMA):
                    mass = GetDeltaVDeltaM(SatInfo, i, 2, mass)
                else:
                    mass = GetDeltaVDeltaM(SatInfo, i, 1, mass)
        SatInfo.Somme_Delta_V = sum(SatInfo.Delta_V)
        SatInfo.Somme_Delta_M = sum(SatInfo.Delta_M)

        SatInfo.MASS_END = mass

    # Number of dates
    SatInfo.NB_DATE = len(SatInfo.DATE_TLE)
    # Number of maneuvers
    SatInfo.NB_MANEUVER = len(SatInfo.DATE_MANEUVER)
    # Number of messages (if there is any)
    SatInfo.NB_MESSAGES = len(SatInfo.MESSAGES)

    return SatInfo