"""
This file contains the code for processing with an AI.

These functions are used in routes.py
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from numpy import transpose, mean, linalg, cross, sqrt, array
from sgp4.earth_gravity import wgs72
import pandas as pd

import warnings
from datetime import datetime
import vg  # Calculate the angle between two vectors in 3D

from app.code.utils.positions import GetLatLong
from app.code.utils.functions import ConvertCoord, GetQSW, GetMeanCross, GetOrbType, GetInitialMass, GetDeltaVDeltaM, CreateLix, ProcessTLES


# -----------------------------------------------------------------------------
# Functions for the AI search
# -----------------------------------------------------------------------------

def preprocessing(df):
    """
    Separates the dataset into X and y

    Parameters
    ----------
    df : dataframe
        The initial dataframe

    Returns
    -------
    X : dataframe
        The dataframe without the 'Maneuver type' column
    y : dataframe
        The 'Maneuver type' column
    """
    X = df.drop('Maneuver type', axis=1)
    y = df['Maneuver type']
    return X, y


def evaluation(model, X_train_scaled, y_train):
    """
    Fits the model and returns it

    Parameters
    ----------
    model : AI model
        Model to fit
    X_train_scaled : dataframe
        The scaled training set
    y_train : dataframe
        The training results

    Returns
    -------
    model : AI model
        Fitted model
    """
    model.fit(X_train_scaled, y_train)
    return model


def detect_maneuvers_IA(dataframe, model):
    """
    Adds the detected maneuver type to the dataframe

    Parameters
    ----------
    dataframe : dataframe
        Satellite's data
    model : AI model
        The fitted model

    Returns
    -------
    dataframe : dataframe
        The dataframe with maneuvers prediction
    """
    df = dataframe.drop('Date', axis=1)
    sc = StandardScaler()
    scaler = sc.fit(df)
    df_scaled = scaler.transform(df)
    dataframe['Maneuver type'] = model.predict(df_scaled)
    return dataframe


def GetPredAI(Lix, filePath):
    """
    Main AI function

    Parameters
    ----------
    Lix : dataframe
        Dataframe with the satellite's data
    filePath : string
        File path to the training file

    Returns
    -------
    dataframe : dataframe
        Dataframe with the maneuvers prediction
    accuracy : list
        The precision of the model
    conf_matrix : list
        The confusion matrix of the model
    report : list
        The global report of the model
    """
    # Dataset options
    pd.set_option('display.max_row', 14)
    pd.set_option('display.max_column', 14)
    data = pd.read_excel(filePath)
    df = data.copy()
    df = df.drop('Date', axis=1)

    # Creation of the training set and the testing set
    #trainset, testset = train_test_split(df, test_size=0.358, shuffle=False)
    trainset, testset = train_test_split(df,random_state=6) # [1:1000]

    X_train, y_train = preprocessing(trainset)
    X_test, y_test = preprocessing(testset)
    sc = StandardScaler()
    scaler = sc.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Old model
    # mlp = MLPClassifier(hidden_layer_sizes=(120, 80, 40),
    #                     max_iter=100, activation='relu',
    #                     solver='adam', learning_rate='constant', alpha=0.0001)
    # AI model
    mlp = MLPClassifier(activation="tanh", hidden_layer_sizes=(50, 5), max_iter=5000, alpha=0.01, solver='adam', random_state=1)

    warnings.simplefilter('ignore')
    testset.drop(testset[testset['Maneuver type'] == 0].index, inplace=True)

    # Training model
    mlp = evaluation(mlp, X_train_scaled, y_train)

    accuracy = mlp.score(X_test_scaled, y_test)

    y_pred_test = mlp.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    labels = [0, 1, 2, 3]
    report = classification_report(y_test, y_pred_test, labels=labels, output_dict=True)

    # Returns the maneuvers prediction, precision, confusion matrix and report of the AI score
    return detect_maneuvers_IA(Lix, mlp), accuracy, conf_matrix, report


def GetSatInfoAI(SatInfo):
    """
    Main function to retrieve the Satellite's data

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object

    Returns
    -------
    SatInfo : SatClass
        Filled satellite object
    Lix : dataframe
        Dataframe with the satellite's data
    """
    # Processing every 6 hours
    ProcessTLES(SatInfo, 0.25, "AI")
    # Convert TEME to ECI then ECI to ECF
    ConvertCoord(SatInfo)
    
    # Get Inc X & Y and Ecc X & Y from positions
    #TMPFUNC(SatInfo) # TODO
    
    # Get orbit type
    GetOrbType(SatInfo, "AI")
    # Get QSW coordinates
    GetQSW(SatInfo)
    # Get Mean Cross
    GetMeanCross(SatInfo)

    # Fill the deltas
    FillDeltas(SatInfo)
    # Adding one last element so the lists have the same size
    FillLists(SatInfo)
    # Fill Energies lists
    FillEnergies(SatInfo)
    
    # Creating the dataframe
    Lix = CreateLix(SatInfo)

    # Number of messages (if there is any)
    SatInfo.NB_MESSAGES = len(SatInfo.MESSAGES)
    
    # TODO
    #Lix.to_csv("./man_ath_ssa.csv")

    return SatInfo, Lix


def PreparePlotsAI(SatInfo):
    """
    Clears the orbital parameters for the plots.
    Uses a 12 hours filter instead of 6h.

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    SatInfo : TYPE
        DESCRIPTION.

    """
    SatInfo.clear()
    # Using the Threshold search because we only need the kep
    ProcessTLES(SatInfo, 0.5, "Threshold")
    ConvertCoord(SatInfo)
    GetLatLong(SatInfo)
    
    # Number of dates
    SatInfo.NB_DATE = len(SatInfo.DATE_TLE)
    
    return SatInfo


def FillDeltas(SatInfo):
    """
    Fills the Deltas lists

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    max_energy = 1000
    half_window_size = 3

    for i in range(len(SatInfo.ENERGY) - 1):
        # Calculate the Energy Delta
        SatInfo.ENERGY_DELTA.append(min(max(SatInfo.ENERGY[i + 1] - SatInfo.ENERGY[i], -max_energy), max_energy))
        #SatInfo.DELTA_ENERGY_FINAL.append(float(min(max(SatInfo.ENERGY_DELTA[i], -max_energy), max_energy)))

        #-print(SatInfo.INCLINATION_X[i + 1], " - ", SatInfo.INCLINATION_X[i], " = ", SatInfo.INCLINATION_X[i + 1] - SatInfo.INCLINATION_X[i])
        SatInfo.DELTA_T.append(SatInfo.DATE_MJD[i + 1] - SatInfo.DATE_MJD[i])
        SatInfo.DELTA_MOMENT.append(vg.angle(SatInfo.MEAN_CROSS[i + 1], SatInfo.MEAN_CROSS[i], units='rad'))
        SatInfo.DELTA_IX.append(SatInfo.INCLINATION_X[i + 1] - SatInfo.INCLINATION_X[i])
        SatInfo.DELTA_IY.append(SatInfo.INCLINATION_Y[i + 1] - SatInfo.INCLINATION_Y[i])
        SatInfo.DELTA_EX.append(SatInfo.ECC_X[i + 1] - SatInfo.ECC_X[i])
        SatInfo.DELTA_EY.append(SatInfo.ECC_Y[i + 1] - SatInfo.ECC_Y[i])
        
        if i < half_window_size:
            SatInfo.DELTA_ALONG_TRACK_AVG.append(SatInfo.DELTA_ALONG_TRACK[i] - mean(SatInfo.DELTA_ALONG_TRACK[:i + 1]))
        elif i > len(SatInfo.DELTA_ALONG_TRACK) - half_window_size:
            SatInfo.DELTA_ALONG_TRACK_AVG.append(SatInfo.DELTA_ALONG_TRACK[i] - mean(SatInfo.DELTA_ALONG_TRACK[i - half_window_size:]))
        else:
            SatInfo.DELTA_ALONG_TRACK_AVG.append(SatInfo.DELTA_ALONG_TRACK[i] - mean(SatInfo.DELTA_ALONG_TRACK[i - half_window_size + 1:i + 1 + half_window_size]))

def FillEnergies(SatInfo):
    """
    Fills the Energy Deltas lists

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    max_energy = 1000
    # No energy before the first TLE
    SatInfo.DELTA_ENERGY_PREV.append(0)
    for i in (transpose(SatInfo.ENERGY_DELTA[:-1])):
        SatInfo.DELTA_ENERGY_PREV.append(min(max(i, -max_energy), max_energy))
    for i in (transpose(SatInfo.ENERGY_DELTA[1:])):
        SatInfo.DELTA_ENERGY_NEXT.append(min(max(i, -max_energy), max_energy))
    # No energy after the last TLE
    SatInfo.DELTA_ENERGY_NEXT.append(0)

            
def FillLists(SatInfo):
    """
    Adds 0 at the beggining of each delta's

    Parameters
    ----------
    SatInfo : SatClass
        The satellite object.

    Returns
    -------
    None.

    """
    SatInfo.DELTA_MOMENT.insert(0, 0)
    SatInfo.DELTA_T.insert(0, 0)
    SatInfo.ENERGY_DELTA.insert(0, 0)
    SatInfo.DELTA_IX.insert(0, 0)
    SatInfo.DELTA_IY.insert(0, 0)
    SatInfo.DELTA_EX.insert(0, 0)
    SatInfo.DELTA_EY.insert(0, 0)
    SatInfo.DELTA_OUT_OF_PLANE.insert(0, 0)
    SatInfo.DELTA_ALONG_TRACK.insert(0, 0)
    SatInfo.DELTA_ALONG_TRACK_AVG.insert(0, 0)
    
    
def GetManeuversAI(SatInfo, Prediction):
    """
    Function to get the maneuvers info

    Parameters
    ----------
    SatInfo : SatClass
        Satellite object
    Prediction : dataframe
        Maneuver prediction dataframe

    Returns
    -------
    SatInfo : SatClass
        Filled satellite object
    """
    type_pred = Prediction['Maneuver type'].tolist()

    SatInfo.DATE_ENERGY_DELTA = SatInfo.DATE_TLE.copy()  # Creation of a list for the Energy Delta with 1 less date
    SatInfo.DATE_ENERGY_DELTA.pop(0)  # Deleting the first date because there is no Delta

    mass = float(SatInfo.MASS[:-2])
    # We will calculate the mass of fuel used between the launch date of the satellite and the start of the values
    if SatInfo.ORBIT == "GEO":
        GetInitialMass(SatInfo, mass)
        
    SatInfo.MASS_BEGIN = mass

    #weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

    tab = []
    for i in Prediction['Date']:
        value = datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S.%f').weekday()
        if value != 6:
            tab.append(value)
        else:
            tab.append(0)

    Prediction['weekDays'] = tab

    SatInfo.AI_DATE = Prediction['Date']
    SatInfo.DAY_OF_WEEK = Prediction['weekDays']
    SatInfo.ALL_TYPE_MANEUVER = Prediction['Maneuver type']

    for i in range(len(SatInfo.DATE_TLE)):
        if type_pred[i] != 0:
            # Keeping the Delta
            SatInfo.DETECTION_MANEUVER.append(SatInfo.ENERGY_DELTA[i])
            # Keeping the maneuver date
            SatInfo.DATE_MANEUVER.append(SatInfo.DATE_TLE[i + 1])
            # Keeping the maneuver type
            SatInfo.TYPE_MANEUVER.append(type_pred[i])

            # Taking a gap of 1 after the maneuver to be sure it is finished in case we are not at the end of the list
            if i + 2 < len(SatInfo.SMA):
                mass = GetDeltaVDeltaM(SatInfo, i, 2, mass)
            else:
                mass = GetDeltaVDeltaM(SatInfo, i, 1, mass)

    SatInfo.Somme_Delta_V = sum(SatInfo.Delta_V)
    SatInfo.Somme_Delta_M = sum(SatInfo.Delta_M)

    SatInfo.MASS_END = mass

    # Number of maneuvers
    SatInfo.NB_MANEUVER = len(SatInfo.DATE_MANEUVER)

    return SatInfo


def TMPFUNC(SatInfo): # TODO remove ?
    """
    TEST CALCULATE IX,IY AND EX, EY
    NOT AS GOOD AS THE KEPLERIAN METHOD IN .POSITIONS

    Parameters
    ----------
    SatInfo : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Semi major axis
    r = [linalg.norm(item) for item in SatInfo.ECI_POSITION]
    #V = [linalg.norm(item) for item in SatInfo.ECI_SPEED]
    #a = []
    W = []
    #esinE = []
    
    for i in range(len(SatInfo.POS_X)):
        #a.append(r[i] / (2 - r[i] * V[i]**2/wgs72.mu))
        #a = r[i] / (2 - r[i] * V[i]**2/wgs72.mu)
        
        W.append(cross(tuple([SatInfo.POS_X[i],
                              SatInfo.POS_Y[i],
                              SatInfo.POS_Z[i]]),
                       tuple([SatInfo.SPEED_X[i],
                              SatInfo.SPEED_Y[i],
                              SatInfo.SPEED_Z[i]])))
        
        #esinE.append(SatInfo.ECI_POSITION[i] @ SatInfo.ECI_SPEED[i] / sqrt(wgs72.mu * a))
    
    # Inclination vector
    #Wu = []
    #for i in range(len(W)):
    #    Wu.append(W[i] / np.linalg.norm(W[i]))
    Wu = [item / linalg.norm(item) for item in W]
    
    ix = [-item[1] / (2 * sqrt((1 + item[2]) / 2)) for item in Wu]
    iy = [item[0] / (2 * sqrt((1 + item[2]) / 2)) for item in Wu]
    
    # Eccentricity vector
    # Coordinates in the "natural frame"
    # P = first column of matrix R
    # Q = second column of matrix R
    ex = []
    ey = []
    for i in range(len(ix)):
        c = sqrt(1 - ix[i]**2 - iy[i]**2)
        P = array([1 - 2 * iy[i]**2, 2 * ix[i] * iy[i], -2 * iy[i] * c])
        Q = array([2 * ix[i] * iy[i], 1 - 2 * ix[i]**2, 2 * ix[i] * c])
        
        X = array(SatInfo.ECI_POSITION[i]) @ P
        Y = array(SatInfo.ECI_POSITION[i]) @ Q
        VX = array(SatInfo.ECI_SPEED[i]) @ P
        VY = array(SatInfo.ECI_SPEED[i]) @ Q
        
        ex.append(linalg.norm(W[i]) * VY / wgs72.mu - X / r[i])
        ey.append(linalg.norm(W[i]) * VX / wgs72.mu - Y / r[i])
    
    SatInfo.INCLINATION_X = ix
    SatInfo.INCLINATION_Y = iy
    SatInfo.ECC_X = ex
    SatInfo.ECC_Y = ey
    