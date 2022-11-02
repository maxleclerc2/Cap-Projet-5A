"""
This file contains the functions to connect to the online APIs DISCOSweb and Space-Track
to retrieve the satellite's data.

These functions are used in routes.py
"""

from flask import current_app as app
from flask_caching import Cache

from requests import get, Session
from tletools import TLE
from datetime import datetime
from json import loads

from .classes import SatClass

cache = Cache(app)

# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------

# Space-Track request URL
uriBase = "https://www.space-track.org"
requestLogin = "/ajaxauth/login"
requestCmdAction = "/basicspacedata/query"
requestCustomSat1 = "/class/satcat/SATNAME/"
requestCustomSat2 = "/orderby/NORAD_CAT_ID asc/emptyresult/show"
requestFindCustom1 = "/class/gp_history/NORAD_CAT_ID/"
requestFindCustom2 = "/orderby/TLE_LINE1%20ASC/EPOCH/"
requestFindCustom3 = "/format/tle"

# DISCOSweb request URL
DWBase = 'https://discosweb.esoc.esa.int'

# Login credentials to the APIs
configUsr = "julien.jaussely@telespazio.com"  # TODO Change the user login
configPwd = "2d0b54e4TLE!!!!!"  # TODO
siteCred = {'identity': configUsr, 'password': configPwd}
DWToken = 'IjllYWViOTFmLTQwM2QtNGNjZS1iMTExLTkyMjA2ZWRkN2NiNyI.ahM4IeXyhQtYG5kzaFMvnCYRjqQ'  # TODO Change the token


# -----------------------------------------------------------------------------
# Functions to connect to the APIs
# -----------------------------------------------------------------------------

def TestAPI(customSat):
    """
    Used to know which API to use to get the satellite's NORAD ID.
    It allow the user to use the satellite's name from Space-Track or the one from DISCOSweb.

    Parameters
    ----------
    customSat : string
        Satellite's name

    Returns
    -------
    string
        The API to use OR an error message
    """
    # Replace the parenthesis by their encoding code to avoid problems with the URL
    satNameQuery = customSat.replace('(', "%28")
    satNameQuery = satNameQuery.replace(')', "%29")

    # Connection to DISCOSweb
    # Looking for a satellite with the wanted name
    response = get(
        f'{DWBase}/api/objects',
        headers={
            'Authorization': f'Bearer {DWToken}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': "eq(name,'" + satNameQuery + "')"
        }
    )

    # Checking that the name is valid if there is a response
    doc = response.json()
    if doc['data']:
        # In that case, we will use DISCOSweb
        # We add the satellite to our 'known satellites' list
        with open("app/static/db/known_sat.txt", "a") as SaveSat:
            SaveSat.write("".join([customSat, "::DW\n"]))
        return "DW"

    # If not, we check with Space-Track
    with Session() as session:
        # Connection to Space-Track
        resp = session.post("".join([uriBase, requestLogin]), data=siteCred)
        if resp.status_code != 200:
            # Error message if unable to connect (connection interrupted)
            return "POST login fail for Space-Track. Please check your Internet connection."

        # Checking that the name is valid
        resp = session.get("".join([uriBase, requestCmdAction, requestCustomSat1,
                                    satNameQuery, requestCustomSat2]))
        if resp.status_code != 200:
            # Error message if unable to connect (connection interrupted)
            return "GET fail on request for Space-Track. Please check your Internet connection."

    # Closing the session
    session.close()

    # Loading the response in JSON
    retData = loads(resp.text)

    # If the response is not empty, then the name is valid
    if retData:
        # We add the satellite to our 'known satellites' list
        with open("app/static/db/known_sat.txt", "a") as SaveSat:
            SaveSat.write("".join([customSat, "::ST\n"]))
        return "ST"

    # If none of the APIs recognize the satellite's name,
    # then we tell the user that the name is incorrect.
    # We add this name to our list of known satellites
    with open("app/static/db/known_sat.txt", "a") as SaveSat:
        SaveSat.write("".join(
            [customSat, "::Satellite name '", customSat, "' unknown. Please make sure that the name is correct.\n"]))
    return "Satellite name '" + customSat + "' unknown. Please make sure that the name is correct."


@cache.memoize(1500)  # Caching the API response
def GetSatInfoDW(customSat, DateBegin, DateEnd, num):
    """
    Function to get the NORAD ID from DISCOSweb

    Parameters
    ----------
    customSat : string
        Satellite's name
    DateBegin : string
        Beginning date of the search
    DateEnd : string
        Ending date of the search
    num : int
        Index of the satellite

    Returns
    -------
    SatInfo : SatClass
        Satellite object
    """
    # Replace the parenthesis by their encoding code to avoid problems with the URL
    satNameQuery = customSat.replace('(', "%28")
    satNameQuery = satNameQuery.replace(')', "%29")

    # Connection to DISCOSweb
    response = get(
        f'{DWBase}/api/objects',
        headers={
            'Authorization': f'Bearer {DWToken}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': "eq(name,'" + satNameQuery + "')"
        }
    )

    # Creation of the SatClass object that will contain all informations
    SatInfo = SatClass()

    # Retrieving data
    doc = response.json()
    h = str(doc['data'][0]['attributes']['height'])
    if h == "None":
        h = "0"
    w = str(doc['data'][0]['attributes']['width'])
    if w == "None":
        w = "0"
    d = str(doc['data'][0]['attributes']['depth'])
    if d == "None":
        d = "0"

    n = str(doc['data'][0]['attributes']['name'])
    if n == "None":
        n = customSat
    oc = str(doc['data'][0]['attributes']['objectClass'])
    if oc == "None":
        oc = "Unknown"
    m = str(doc['data'][0]['attributes']['mass'])
    if m == "None":
        m = "0"

    # Dimensions as a sentence
    dim = "".join(["Height: ", h, "m; Width: ", w, "m; Depth: ", d, "m"])

    # Saving informations into SatInfo
    SatInfo.NAME = n
    SatInfo.ID = doc['data'][0]['attributes']['satno']
    SatInfo.OBJECT_CLASS = oc
    SatInfo.MASS = "".join([m, "Kg"])
    SatInfo.DIMENSION = dim

    # URL to the launching data
    LaunchLink = doc['data'][0]['relationships']['launch']['links']['related']

    # Connection to DISCOSweb
    response = get(
        f'{DWBase}/{LaunchLink}',
        headers={
            'Authorization': f'Bearer {DWToken}',
            'DiscosWeb-Api-Version': '2',
        }
    )

    # Retrieving data
    doc = response.json()

    # We check that the response is not empty
    if doc['data']:
        try:
            # Converting the launch date from string to datetime
            date = doc['data']['attributes']['epoch'][:10]
            SatInfo.LAUNCH_DATE = datetime.strptime(date, "%Y-%m-%d").date()
        except:
            # Error message to tell that there are no launching data
            SatInfo.MESSAGES.append("".join(["Launch date unknown for ", SatInfo.NAME]))

        # URL to the launching site data
        LaunchSiteLink = doc['data']['relationships']['site']['links']['related']

        # Connection to DISCOSweb
        response = get(
            f'{DWBase}/{LaunchSiteLink}',
            headers={
                'Authorization': f'Bearer {DWToken}',
                'DiscosWeb-Api-Version': '2',
            }
        )

        # Retrieving data
        doc = response.json()

        # We check that the response is not empty
        if doc['data']:
            # Saving data
            SatInfo.LAUNCH_SITE = doc['data']['attributes']['name']
        else:
            # Error message to tell that there are no launching site data
            SatInfo.MESSAGES.append("".join(["Launch site unknown for ", SatInfo.NAME]))
    else:
        # Error message to tell that there are no launching data
        SatInfo.MESSAGES.append("".join(["Launch date & site unknown for ", SatInfo.NAME]))

    # We now use Space-Track to get additional informations and the TLEs

    # Converting the dates into datetime objects
    DateBegin = datetime.strptime(DateBegin, "%Y-%m-%d").date()
    DateEnd = datetime.strptime(DateEnd, "%Y-%m-%d").date()

    # Checking if the dates match the launching date
    if DateBegin < SatInfo.LAUNCH_DATE:
        # If the beginning date is before the launch, we indicate it to the user
        SatInfo.MESSAGES.append("".join(["Data start date changed from ",
                                         str(DateBegin), " to ", str(SatInfo.LAUNCH_DATE),
                                         " for ", SatInfo.NAME]))
        DateBegin = SatInfo.LAUNCH_DATE
    if DateEnd < SatInfo.LAUNCH_DATE:
        # If the ending date is before the launch, error
        SatInfo.ERROR_MESSAGE = "".join(
            ["Data end date is before the satellite's launch date. Please select a date after ",
             str(SatInfo.LAUNCH_DATE)])
        return SatInfo

    # Converting the dates back to strings
    DateBegin = str(DateBegin)
    DateEnd = str(DateEnd)

    # Connection to Space-Track
    with Session() as session:
        # Sending logging credentials
        resp = session.post("".join([uriBase, requestLogin]), data=siteCred)
        if resp.status_code != 200:
            # Error message if unable to connect (connection interrupted)
            SatInfo.ERROR_MESSAGE = "POST fail on login for Space-Track. Please check your Internet connection."
            return SatInfo

        # Retrieving the response from the NORAD ID
        resp = session.get("".join([uriBase, requestCmdAction, requestFindCustom1,
                                    str(SatInfo.ID), requestFindCustom2, DateBegin,
                                    "--", DateEnd, requestFindCustom3]))
        if resp.status_code != 200:
            # Error message if unable to connect (connection interrupted)
            SatInfo.ERROR_MESSAGE = "".join(["Failed to retrieve data for ",
                                             customSat, " from ", DateBegin,
                                             " to ", DateEnd, ". The satellite was launched the ",
                                             str(SatInfo.LAUNCH_DATE)])
            return SatInfo

        # Retrieving the response line by line
        retData = resp.text.splitlines()

    # Closing connection
    session.close()

    L_1 = []  # List of lines "1" of the TLEs
    L_2 = []  # List of lines "2" of the TLEs

    i = 1

    # For each line of the response, switching between 1 and 2
    for line in retData:
        j = i
        if i == 1:
            # If 1, adding the line to L_1
            L_1.append(line[:69])
            j = 2
        elif i == 2:
            # If 2, adding the line to L_2
            L_2.append(line[:69])
            j = 1
        # Swap
        i = j

    # List of TLEs
    TLES = []
    k = 0
    # Processing until the last TLE
    while k < len(L_1):
        # Recreating the TLE from lines 1 and 2
        tle = TLE.from_lines(customSat, L_1[k], L_2[k])
        TLES.append(tle)

        k += 1

    # Saving data
    SatInfo.SAT_NUM = num
    SatInfo.TLES = TLES
    SatInfo.DATE_BEGIN = DateBegin
    SatInfo.DATE_END = DateEnd
    SatInfo.L_1 = L_1
    SatInfo.L_2 = L_2
    SatInfo.COMPLETED = True

    # Return the satellite object
    return SatInfo


@cache.memoize(1500)  # Caching the API response
def GetSatInfoST(customSat, DateBegin, DateEnd, num):
    """
    Function to get the NORAD ID from Space-Track

    Parameters
    ----------
    customSat : string
        Satellite's name
    DateBegin : string
        Beginning date of the search
    DateEnd : string
        Ending date of the search
    num : int
        Index of the satellite

    Returns
    -------
    SatInfo : SatClass
        Satellite object
    """
    # Creation of the SatClass object that will contain all informations
    SatInfo = SatClass()

    # use requests package to drive the RESTful session with space-track.org
    with Session() as session:
        # run the session in a with block to force session to close if we exit

        # need to log in first. note that we get a 200 to say the web site got the data, not that we are logged in
        resp = session.post("".join([uriBase, requestLogin]), data=siteCred)
        if resp.status_code != 200:
            SatInfo.ERROR_MESSAGE = "POST fail on login"
            return SatInfo

        # Replace the parenthesis by their encoding code to avoid problems with the URL
        satNameQuery = customSat.replace('(', "%28")
        satNameQuery = satNameQuery.replace(')', "%29")

        # Request to get the NORAD ID
        resp = session.get(uriBase + requestCmdAction + requestCustomSat1 + satNameQuery + requestCustomSat2)
        if resp.status_code != 200:
            SatInfo.ERROR_MESSAGE = "".join(["GET fail on request for ",
                                             customSat])
            return SatInfo

            # use the json package to break the json formatted response text into a Python structure (a list of
            # dictionaries)
        retData = loads(resp.text)
        SatID = 0

        # If the response is empty, then the name is invalid (SHOULD NOT HAPPEN)
        if not retData:
            SatInfo.ERROR_MESSAGE = "".join(["Nom de satellite \"", customSat,
                                             "\" invalide"])
            return SatInfo

        for e in retData:
            # Retrieve the NORAD ID
            if not 'NORAD_CAT_ID' in e or len(e['NORAD_CAT_ID']) == 0:
                SatInfo.ERROR_MESSAGE = "".join(["Satellite \"", customSat,
                                                 "\" unknown"])
                return SatInfo
            SatID = e['NORAD_CAT_ID']

        # We now use DISCOSweb to get additional informations

        # Connection to DiscosWeb with the NORAD ID
        response = get(
            f'{DWBase}/api/objects',
            headers={
                'Authorization': f'Bearer {DWToken}',
                'DiscosWeb-Api-Version': '2',
            },
            params={
                'filter': "eq(satno," + SatID + ")"
            }
        )

        # Retrieving data
        doc = response.json()
        h = str(doc['data'][0]['attributes']['height'])
        if h == "None":
            h = "0"
        w = str(doc['data'][0]['attributes']['width'])
        if w == "None":
            w = "0"
        d = str(doc['data'][0]['attributes']['depth'])
        if d == "None":
            d = "0"

        n = str(doc['data'][0]['attributes']['name'])
        if n == "None":
            n = customSat
        oc = str(doc['data'][0]['attributes']['objectClass'])
        if oc == "None":
            oc = "Unknown"
        m = str(doc['data'][0]['attributes']['mass'])
        if m == "None":
            m = "0"

        # Dimensions as a sentence
        dim = "".join(["Height: ", h, "m; Width: ", w, "m; Depth: ", d, "m"])

        # Saving informations into SatInfo
        SatInfo.NAME = n
        SatInfo.OBJECT_CLASS = oc
        SatInfo.MASS = "".join([m, "Kg"])
        SatInfo.DIMENSION = dim

        # URL to the launching data
        LaunchLink = doc['data'][0]['relationships']['launch']['links']['related']

        # Connection to DISCOSweb
        response = get(
            f'{DWBase}/{LaunchLink}',
            headers={
                'Authorization': f'Bearer {DWToken}',
                'DiscosWeb-Api-Version': '2',
            }
        )

        # Retrieving data
        doc = response.json()

        # We check that the response is not empty
        if doc['data']:
            try:
                # Converting the launch date from string to datetime
                date = doc['data']['attributes']['epoch'][:10]
                SatInfo.LAUNCH_DATE = datetime.strptime(date, "%Y-%m-%d").date()
            except:
                # Error message to tell that there are no launching data
                SatInfo.MESSAGES.append("".join(["Launch date unknown for ", SatInfo.NAME]))

            # URL to the launching site data
            LaunchSiteLink = doc['data']['relationships']['site']['links']['related']

            # Connection to DISCOSweb
            response = get(
                f'{DWBase}/{LaunchSiteLink}',
                headers={
                    'Authorization': f'Bearer {DWToken}',
                    'DiscosWeb-Api-Version': '2',
                }
            )

            # Retrieving data
            doc = response.json()

            # We check that the response is not empty
            if doc['data']:
                # Saving data
                SatInfo.LAUNCH_SITE = str(doc['data']['attributes']['name'])
            else:
                # Error message to tell that there are no launching site data
                SatInfo.MESSAGES.append("".join(["Launch site unknown for ", SatInfo.NAME]))
        else:
            # Error message to tell that there are no launching data
            SatInfo.MESSAGES.append("".join(["Launch date & site unknown for ", SatInfo.NAME]))

        # Converting the dates into datetime objects
        DateBegin = datetime.strptime(DateBegin, "%Y-%m-%d").date()
        DateEnd = datetime.strptime(DateEnd, "%Y-%m-%d").date()

        # Checking if the dates match the launching date
        if DateBegin < SatInfo.LAUNCH_DATE:
            # If the beginning date is before the launch, we indicate it to the user
            SatInfo.MESSAGES.append("".join(["Data start date changed from ",
                                             str(DateBegin), " to ", str(SatInfo.LAUNCH_DATE),
                                             " for ", SatInfo.NAME]))
            DateBegin = SatInfo.LAUNCH_DATE
        if DateEnd < SatInfo.LAUNCH_DATE:
            # If the ending date is before the launch, error
            SatInfo.ERROR_MESSAGE = "".join(
                ["Data end date is before the satellite's launch date. Please select a date after ",
                 str(SatInfo.LAUNCH_DATE)])
            return SatInfo

        # Converting the dates back to strings
        DateBegin = str(DateBegin)
        DateEnd = str(DateEnd)

        # Retrieving the response from the NORAD ID
        resp = session.get("".join([uriBase, requestCmdAction, requestFindCustom1,
                                    SatID, requestFindCustom2, DateBegin, "--",
                                    DateEnd, requestFindCustom3]))
        if resp.status_code != 200:
            SatInfo.ERROR_MESSAGE = "".join(["Failed to retrieve data for ",
                                             customSat, " from ", DateBegin,
                                             " to ", DateEnd, ". The satellite was launched the ",
                                             str(SatInfo.LAUNCH_DATE)])
            return SatInfo

        # Retrieving the response line by line
        retData = resp.text.splitlines()

    # Closing connection
    session.close()

    L_1 = []  # List of lines "1" of the TLEs
    L_2 = []  # List of lines "2" of the TLEs

    i = 1

    # For each line of the response, switching between 1 and 2
    for line in retData:
        j = i
        if i == 1:
            # If 1, adding the line to L_1
            L_1.append(line[:69])
            j = 2
        elif i == 2:
            # If 2, adding the line to L_2
            L_2.append(line[:69])
            j = 1
        # Swap
        i = j

    # List of TLEs
    TLES = []
    k = 0
    # Processing until the last TLE
    while k < len(L_1):
        # Recreating the TLE from lines 1 and 2
        tle = TLE.from_lines(customSat, L_1[k], L_2[k])
        TLES.append(tle)

        k += 1

    # Saving data
    SatInfo.SAT_NUM = num
    SatInfo.TLES = TLES
    SatInfo.ID = SatID
    SatInfo.DATE_BEGIN = DateBegin
    SatInfo.DATE_END = DateEnd
    SatInfo.L_1 = L_1
    SatInfo.L_2 = L_2
    SatInfo.COMPLETED = True

    # Return the satellite object
    return SatInfo

# TODO Uses the code from above: should be in the same function
@cache.memoize(1500)  # Caching the API response
def CompleteInfoFromFile(SatInfo):
    """
    Function to complete the informations of a satellite loaded from a file
    
    Parameters
    ----------
    SatInfo : SatClass
        Initial satellite object

    Returns
    -------
    SatInfo : SatClass
        Filled satellite object
    """
    
    # Connection to DiscosWeb with the NORAD ID
    response = get(
        f'{DWBase}/api/objects',
        headers={
            'Authorization': f'Bearer {DWToken}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': "eq(satno," + SatInfo.ID + ")"
        }
    )

    # Retrieving data
    doc = response.json()
    h = str(doc['data'][0]['attributes']['height'])
    if h == "None":
        h = "0"
    w = str(doc['data'][0]['attributes']['width'])
    if w == "None":
        w = "0"
    d = str(doc['data'][0]['attributes']['depth'])
    if d == "None":
        d = "0"

    n = str(doc['data'][0]['attributes']['name'])
    if n == "None":
        n = SatInfo.NAME
    oc = str(doc['data'][0]['attributes']['objectClass'])
    if oc == "None":
        oc = "Unknown"
    m = str(doc['data'][0]['attributes']['mass'])
    if m == "None":
        m = "0"

    # Dimensions as a sentence
    dim = "".join(["Height: ", h, "m; Width: ", w, "m; Depth: ", d, "m"])

    # Saving informations into SatInfo
    SatInfo.NAME = n
    SatInfo.OBJECT_CLASS = oc
    SatInfo.MASS = "".join([m, "Kg"])
    SatInfo.DIMENSION = dim

    # URL to the launching data
    LaunchLink = doc['data'][0]['relationships']['launch']['links']['related']

    # Connection to DISCOSweb
    response = get(
        f'{DWBase}/{LaunchLink}',
        headers={
            'Authorization': f'Bearer {DWToken}',
            'DiscosWeb-Api-Version': '2',
        }
    )

    # Retrieving data
    doc = response.json()

    # We check that the response is not empty
    if doc['data']:
        try:
            # Converting the launch date from string to datetime
            date = doc['data']['attributes']['epoch'][:10]
            SatInfo.LAUNCH_DATE = datetime.strptime(date, "%Y-%m-%d").date()
        except:
            # Error message to tell that there are no launching data
            SatInfo.MESSAGES.append("".join(["Launch date unknown for ", SatInfo.NAME]))

        # URL to the launching site data
        LaunchSiteLink = doc['data']['relationships']['site']['links']['related']

        # Connection to DISCOSweb
        response = get(
            f'{DWBase}/{LaunchSiteLink}',
            headers={
                'Authorization': f'Bearer {DWToken}',
                'DiscosWeb-Api-Version': '2',
            }
        )

        # Retrieving data
        doc = response.json()

        # We check that the response is not empty
        if doc['data']:
            # Saving data
            SatInfo.LAUNCH_SITE = str(doc['data']['attributes']['name'])
        else:
            # Error message to tell that there are no launching site data
            SatInfo.MESSAGES.append("".join(["Launch site unknown for ", SatInfo.NAME]))
    else:
        # Error message to tell that there are no launching data
        SatInfo.MESSAGES.append("".join(["Launch date & site unknown for ", SatInfo.NAME]))
        
    return SatInfo