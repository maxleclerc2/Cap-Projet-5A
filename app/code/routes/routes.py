"""
This file defines the routes used by Flask and the front-end.
Each route defines the main logic of the app:
    - initialize the variables
    - if a form is submitted, contact the APIs and process the satellite data
    - return to the user a web page with all the informations
"""
import datetime
import os
from argparse import Namespace

from flask import (request, render_template, send_file, abort,
                   send_from_directory, jsonify)
from flask import current_app as app
from werkzeug.utils import secure_filename

from zipfile import ZipFile
from os import walk
from os.path import join, splitext, abspath
from io import BytesIO
import logging
import zipfile

from app.code.astrometry.SpotRemover import RemoveGreenStars
from app.code.klt.core.errors import ConfigurationError
from app.code.utils.api import TestAPI, GetSatInfoDW, GetSatInfoST
from app.code.astrometry.astrometry import SubmitAstrometry
from app.code.utils.classes import SatForm, AddSatForm, SatFile, AIForm, SatClass, ProcessImageForm
from app.code.utils.functions import GetSatInfoFile, czml, create_plot_list, AddFilesToZip
from app.code.utils.image_processing_progress import ImageProcessingProgress
from app.code.klt.core.configuration import Configuration
from app.code.klt.klt_matcher.matcher import match
from app.code.threshold.threshold import GetTleInfo
from app.code.ai.ai import GetSatInfoAI, GetPredAI, GetManeuversAI, PreparePlotsAI

# TODO from app.code.proximity.proximity import GetTleProximity, DetectProximity
# TODO from app.code.space_weather.SpaceWeather import run as SP_run

# Global variables
# TODO For development only, not working when there are multiple users simultaneously
# Threshold
global threshold_SatInfo
global threshold_nb_sat
global threshold_SatNames
global threshold_to_remove
threshold_SatInfo = []
threshold_nb_sat = 0
threshold_SatNames = []
threshold_to_remove = False
# AI
global ai_SatInfo
global ai_nb_sat
global ai_SatNames
global ai_to_remove
global Prediction
ai_SatInfo = []
ai_nb_sat = 0
ai_SatNames = []
ai_to_remove = False
Prediction = None

# KLT Processing progress
global klt_progress
# Astrometry Processing progress
global astro_progress


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

# favicon.ico
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame with no cache
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# AI
@app.route('/ai', methods=['GET', 'POST'])
def ai():
    # We use the global variables
    global ai_SatInfo
    global ai_nb_sat
    global ai_SatNames
    global ai_to_remove
    global Prediction

    # Tell if the information needs to be recalculated
    AddedSat = False

    # Set the satellites as not complete
    # If the list is empty (first load of the page), initialize it
    try:
        if ai_SatInfo[0].COMPLETED:
            for i in range(ai_nb_sat):
                ai_SatInfo[i].COMPLETED = False
    except:
        logging.info('AI - Page initialized')
        sat0 = SatClass()
        ai_SatInfo.append(sat0)

    # If the added satellite is already in the list, remove it
    if ai_to_remove:
        logging.info('AI - Satellite already search, removed')
        ai_SatInfo.pop()
        ai_nb_sat -= 1
        ai_to_remove = False

    # Create the forms variables
    SatName = None
    DateBegin = None
    DateEnd = None
    Visu = None

    # Create the forms
    formAi = AIForm()
    formAddSat = AddSatForm()

    # If the 'new search' form is completed and sent
    if formAi.submit.data and formAi.validate():
        logging.info('AI - Form submitted')

        # We reset the global variables
        ai_SatInfo = []
        ai_nb_sat = 0
        ai_SatNames = []
        Prediction = None

        # Saving the form data
        SatName = formAi.SatName.data
        DateBegin = formAi.DateBegin.data
        DateEnd = formAi.DateEnd.data
        Visu = formAi.Visu.data

        # Check if the user submitted a file
        if request.files['fileUpload']:
            # Saving the uploaded file
            uploaded_file = request.files['fileUpload']
            filename = secure_filename(uploaded_file.filename)
            if filename != '':
                file_ext = splitext(filename)[1]
                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    abort(400)
                uploaded_file.save(join(app.config['UPLOAD_PATH'], filename))
                logging.info('AI - Using uploaded learning file ' + filename)
        else:
            # Use default machine learning file
            logging.info('AI - Using default learning file')
            filename = "Manoeuvers_Ath_Scilab.xlsx"

        # Path to the uploaded file
        filePath = app.config['UPLOAD_PATH'] + "\\" + filename

        uploaded_file2 = request.files['fileUpload2']
        filename2 = secure_filename(uploaded_file2.filename)

        # Search from local files
        if filename2 != '' and SatName == '':
            logging.info('AI - Process from files')
            # Saving the uploaded file
            file_ext2 = splitext(filename2)[1]
            if file_ext2 not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file2.save(join(app.config['UPLOAD_PATH'], filename2))

            # Path to the uploaded file2
            filePath2 = app.config['UPLOAD_PATH'] + "\\" + filename2

            ai_SatInfo.append(GetSatInfoFile(filePath2))

        elif filename2 == '' and SatName != '':
            # If the end date is before the start date, we switch them
            if DateBegin > DateEnd:
                tmp = DateBegin
                DateBegin = DateEnd
                DateEnd = tmp
                logging.info('AI - Date swap : %s - %s', str(DateBegin), str(DateEnd))

            # Adding the satellite's name to the list
            ai_SatNames.append(SatName)

            # Open the file with the satellite's name already searched
            knownSat = []
            with open("app/static/db/known_sat.txt", "r") as SatSaved:
                doc = SatSaved.readlines()
                for line in doc:
                    # Reformat the line
                    tmp = line.split("::")
                    tmp[1] = tmp[1].replace("\n", "")
                    knownSat.append(tmp)

            # We check if the name of the satellite was already searched before calling the APIs
            needApi = True
            for known in knownSat:
                if known[0].lower() == ai_SatNames[0].lower():
                    # Satellite known, no need to call the APIs
                    api = known[1]
                    needApi = False
                    logging.info('AI - %s already known', ai_SatNames[0])
                    break

            # If the name is unknown, we test the APIs
            if needApi:
                logging.info('AI - Testing API call on %s', ai_SatNames[0])
                api = TestAPI(ai_SatNames[0])

            # Calling DISCOSweb
            if api == 'DW':
                logging.info('AI - Getting TLE of %s', ai_SatNames[0])
                ai_SatInfo.append(GetSatInfoDW(ai_SatNames[0], DateBegin, DateEnd, 0))
            # Calling Space-Track
            elif api == 'ST':
                logging.info('AI - Getting TLE of %s', ai_SatNames[0])
                ai_SatInfo.append(GetSatInfoST(ai_SatNames[0], DateBegin, DateEnd, 0))
            # If the name is unknown, error
            else:
                logging.error('AI - %s', api)
                return render_template('error.html', errorMessage=api)

            # If there is an error, we display it to the user
            if ai_SatInfo[0].ERROR_MESSAGE:
                logging.error('AI - %s', ai_SatInfo[0].ERROR_MESSAGE)
                return render_template('error.html', errorMessage=ai_SatInfo[0].ERROR_MESSAGE)

        else:
            logging.error('AI - Form not filled')
            return render_template('error.html', errorMessage='Please fill the form')

        # We have 1 satellite in our list
        ai_nb_sat += 1

        # Saving the 3D visualization value
        ai_SatInfo[0].VISU = Visu
        # Saving the Machine Learning training file
        ai_SatInfo[0].FILE = filename

    # If the 'add satellite' form is completed and sent
    if formAddSat.submit.data and formAddSat.validate():
        logging.info('AI - Adding satellite')

        # Saving the name
        SatNameAdd = formAddSat.SatNameAdd.data

        # Open the file with the satellite's name already searched
        knownSat = []
        with open("app/static/db/known_sat.txt", "r") as SatSaved:
            doc = SatSaved.readlines()
            for line in doc:
                # Reformat the line
                tmp = line.split("::")
                tmp[1] = tmp[1].replace("\n", "")
                knownSat.append(tmp)

        # We check if the name of the satellite was already searched before calling the APIs
        needApi = True
        for known in knownSat:
            if known[0].lower() == SatNameAdd.lower():
                # Satellite known, no need to call the APIs
                api = known[1]
                needApi = False
                logging.info('AI - %s already known', SatNameAdd)
                break

        # If the name is unknown, we test the APIs
        if needApi:
            logging.info('AI - Testing API call on %s', SatNameAdd)
            api = TestAPI(SatNameAdd)

        # DateBegin
        d_b = ai_SatInfo[0].DATE_BEGIN
        # DateEnd
        d_e = ai_SatInfo[0].DATE_END

        # Calling DISCOSweb
        if api == 'DW':
            logging.info('AI - Getting TLE of %s', SatNameAdd)
            ai_SatInfo.append(GetSatInfoDW(SatNameAdd, d_b, d_e, 1))
        # Calling Space-Track
        elif api == 'ST':
            logging.info('AI - Getting TLE of %s', SatNameAdd)
            ai_SatInfo.append(GetSatInfoST(SatNameAdd, d_b, d_e, 1))
        # If the name is unknown, error
        else:
            logging.error('AI - %s', api)
            return render_template('error.html', errorMessage=api)

        # If there is an error, we display it to the user
        if ai_SatInfo[ai_nb_sat].ERROR_MESSAGE:
            logging.error('AI - %s', ai_SatInfo[ai_nb_sat].ERROR_MESSAGE)
            return render_template('error.html', errorMessage=ai_SatInfo[ai_nb_sat].ERROR_MESSAGE)

        for i in range(ai_nb_sat):
            # Changing values to not recalculate the data of the first satellite
            ai_SatInfo[i].COMPLETED = True
            # We check that the name of the new satellite is not in the list
            if ai_SatInfo[i].NAME == ai_SatInfo[ai_nb_sat].NAME:
                # We need to remove the added satellite, we tell it to the user
                logging.info('AI - Same satellite name, skipping second satellite')
                ai_SatInfo[ai_nb_sat].COMPLETED = False
                ai_SatInfo[ai_nb_sat].MESSAGES.append(
                    "The added satellite is already in the visualization. Please use another satellite for comparison.")
                ai_SatInfo[ai_nb_sat].NB_MESSAGES = 1
                ai_to_remove = True
        ai_nb_sat += 1
        AddedSat = True

    # Global try-except in case there is an unknown error outside the processing functions
    try:
        for i in range(ai_nb_sat):
            # Converting the launch date to string
            ai_SatInfo[i].LAUNCH_DATE = str(ai_SatInfo[i].LAUNCH_DATE)

            if ai_SatInfo[i].LAUNCH_DATE == "1900-01-01":
                # We remove the default value
                ai_SatInfo[i].LAUNCH_DATE = "Unknown"

            # If the search is complete
            if ai_SatInfo[i].COMPLETED:
                # If there is no added satellite, we recalculate everything
                if not AddedSat or i == (ai_nb_sat - 1):
                    logging.info('AI - Getting TLE informations from %s', ai_SatInfo[i].NAME)

                    # Global try-except in case there is an unknown error outside the processing functions
                    try:
                        # Retrieving the data
                        results = GetSatInfoAI(ai_SatInfo[i])
                        ai_SatInfo[i] = results[0]

                        # We save the prediction dataframe only for the first satellite of the list
                        if i == 0:
                            Lix = results[1]
                    except Exception as e:
                        logging.exception('AI - %s', e)
                        ai_SatInfo[i].ERROR_MESSAGE = "Something went wrong, please send the ssa.log to the devs"
                    # If there is an error, we display it to the user
                    if ai_SatInfo[i].ERROR_MESSAGE:
                        return render_template('error.html', errorMessage=ai_SatInfo[i].ERROR_MESSAGE)
                    # Getting the prediction only for the first satellite
                    if i == 0:
                        # AI function
                        Prediction, ai_SatInfo[i].ACCURACY, ai_SatInfo[i].CONFUSION_MATRIX, ai_SatInfo[
                            i].REPORT = GetPredAI(Lix, filePath)
                        # Saving the maneuvers
                        ai_SatInfo[i] = GetManeuversAI(ai_SatInfo[i], Prediction)

                        # Exporting data
                        ai_SatInfo[i].exportData("ai")

                        # Plot maneuvers
                        ai_SatInfo[i].make_plot_AI()

        if ai_SatInfo[0].COMPLETED:
            for i in range(ai_nb_sat):
                ai_SatInfo[i] = PreparePlotsAI(ai_SatInfo[i])
            # Plot the orbital parameters
            create_plot_list(ai_SatInfo, ai_nb_sat, "ai")

            if ai_SatInfo[0].VISU == 'yes':
                # Generate the CZML file only if needed
                czml(ai_SatInfo, ai_nb_sat, "ai")

            logging.info('AI - Complete')
    except Exception as e:
        logging.exception('AI - %s', e)
        errorMessage = "Something went wrong, please send the ssa.log to the devs"
        return render_template('error.html', errorMessage=errorMessage)

    # Clear the form
    formAi.SatName.data = ''
    formAi.DateBegin.data = ''
    formAi.DateEnd.data = ''
    formAi.Visu.data = 'no'
    formAi.fileUpload.data = None
    formAi.fileUpload2.data = None

    return render_template('ai.html', formAi=formAi, formAddSat=formAddSat, SatList=ai_SatInfo,
                           nb_sat=ai_nb_sat)


# Threshold
@app.route('/threshold', methods=['GET', 'POST'])
def threshold():
    # We use the global variables
    global threshold_SatInfo
    global threshold_nb_sat
    global threshold_SatNames
    global threshold_to_remove

    # Create the forms variables
    SatName = None
    SatName2 = None
    DateBegin = None
    DateEnd = None
    Seuil = None
    Visu = None
    ManualVal = None
    AutoVal = None

    # Tell if the information needs to be recalculated
    AddedSat = False

    # Set the satellites as not complete
    # If the list is empty (first load of the page), initialize it
    try:
        if threshold_SatInfo[0].COMPLETED:
            for i in range(threshold_nb_sat):
                threshold_SatInfo[i].COMPLETED = False
    except:
        logging.info('Threshold - Page initialized')
        sat0 = SatClass()
        threshold_SatInfo.append(sat0)

    # If the added satellite is already in the list, remove it
    if threshold_to_remove:
        logging.info('Threshold - Satellite already search, removed')
        threshold_SatInfo.pop()
        threshold_nb_sat -= 1
        threshold_to_remove = False

    # Create the forms
    formSat = SatForm()
    formAddSat = AddSatForm()
    formFile = SatFile()

    # If the 'new search' form is completed and sent
    if formSat.submit.data and formSat.validate():
        logging.info('Threshold - Form submitted')

        # We reset the global variables
        threshold_SatInfo = []
        threshold_nb_sat = 0
        threshold_SatNames = []

        # Saving the form data
        SatName = formSat.SatName.data
        SatName2 = formSat.SatName2.data
        DateBegin = formSat.DateBegin.data
        DateEnd = formSat.DateEnd.data
        Seuil = formSat.Seuil.data
        Visu = formSat.Visu.data
        ManualVal = formSat.ManualVal.data
        AutoVal = formSat.AutoVal.data

        # If the end date is before the start date, we switch them
        if DateBegin > DateEnd:
            tmp = DateBegin
            DateBegin = DateEnd
            DateEnd = tmp
            logging.info('Threshold - Date swap : %s - %s', str(DateBegin), str(DateEnd))

        # Adding the satellite's name to the list
        threshold_SatNames.append(SatName)
        # If the second satellite's name is not empty, we add it
        if SatName2 != "":
            threshold_SatNames.append(SatName2)

        # Open the file with the satellite's name already searched
        knownSat = []
        with open("app/static/db/known_sat.txt", "r") as SatSaved:
            doc = SatSaved.readlines()
            for line in doc:
                # Reformat the line
                tmp = line.split("::")
                tmp[1] = tmp[1].replace("\n", "")
                knownSat.append(tmp)

        i = 0
        # Retrieving the informations for each satellite
        for name in threshold_SatNames:
            # We check if the name of the satellite was already searched before calling the APIs
            needApi = True
            for known in knownSat:
                if known[0].lower() == name.lower():
                    # Satellite known, no need to call the APIs
                    api = known[1]
                    needApi = False
                    logging.info('Threshold - %s already known', name)
                    break

            # If the name is unknown, we test the APIs
            if needApi:
                logging.info('Threshold - Testing API call on %s', name)
                api = TestAPI(name)

            # Calling DISCOSweb
            if api == 'DW':
                logging.info('Threshold - Getting TLE of %s', name)
                threshold_SatInfo.append(GetSatInfoDW(name, DateBegin, DateEnd, i))
            # Calling Space-Track
            elif api == 'ST':
                logging.info('Threshold - Getting TLE of %s', name)
                threshold_SatInfo.append(GetSatInfoST(name, DateBegin, DateEnd, i))
            # If the name is unknown, error
            else:
                logging.error('Threshold - %s', api)
                return render_template('error.html', errorMessage=api)

            # If there is an error, we display it to the user
            if threshold_SatInfo[i].ERROR_MESSAGE:
                logging.error('Threshold - %s', threshold_SatInfo[i].ERROR_MESSAGE)
                return render_template('error.html', errorMessage=threshold_SatInfo[i].ERROR_MESSAGE)

            # Next sat
            i += 1
            threshold_nb_sat += 1

        try:
            # Saving the 3D visualization value
            threshold_SatInfo[0].VISU = Visu

            # We check that the name of the second satellite is not in the list
            if threshold_SatInfo[0].NAME == threshold_SatInfo[1].NAME:
                # We need to remove the second satellite, we tell it to the user
                logging.info('Threshold - Same satellite name, skipping second satellite')
                threshold_SatInfo[1].COMPLETED = False
                threshold_SatInfo[1].MESSAGES.append(
                    "The second satellite is the same as the first one. Please use another satellite for comparison.")
                threshold_SatInfo[1].NB_MESSAGES = 1
                threshold_to_remove = True
        except:
            logging.info('Threshold - No second satellite')

    # If the 'add satellite' form is completed and sent
    if formAddSat.submit.data and formAddSat.validate():
        logging.info('Threshold - Adding satellite')

        # Saving the name
        SatNameAdd = formAddSat.SatNameAdd.data

        # Open the file with the satellite's name already searched
        knownSat = []
        with open("app/static/db/known_sat.txt", "r") as SatSaved:
            doc = SatSaved.readlines()
            for line in doc:
                # Reformat the line
                tmp = line.split("::")
                tmp[1] = tmp[1].replace("\n", "")
                knownSat.append(tmp)

        # We check if the name of the satellite was already searched before calling the APIs
        needApi = True
        for known in knownSat:
            if known[0].lower() == SatNameAdd.lower():
                # Satellite known, no need to call the APIs
                api = known[1]
                needApi = False
                logging.info('Threshold - %s already known', SatNameAdd)
                break

        # If the name is unknown, we test the APIs
        if needApi:
            logging.info('Threshold - Testing API call on %s', SatNameAdd)
            api = TestAPI(SatNameAdd)

        # DateBegin
        d_b = threshold_SatInfo[0].DATE_BEGIN
        # DateEnd
        d_e = threshold_SatInfo[0].DATE_END

        # Calling DISCOSweb
        if api == 'DW':
            logging.info('Threshold - Getting TLE of %s', SatNameAdd)
            threshold_SatInfo.append(GetSatInfoDW(SatNameAdd, d_b, d_e, 1))
        # Calling Space-Track
        elif api == 'ST':
            logging.info('Threshold - Getting TLE of %s', SatNameAdd)
            threshold_SatInfo.append(GetSatInfoST(SatNameAdd, d_b, d_e, 1))
        # If the name is unknown, error
        else:
            logging.error('Threshold - %s', api)
            return render_template('error.html', errorMessage=api)

        # If there is an error, we display it to the user
        if threshold_SatInfo[threshold_nb_sat].ERROR_MESSAGE:
            logging.error('Threshold - %s', threshold_SatInfo[threshold_nb_sat].ERROR_MESSAGE)
            return render_template('error.html', errorMessage=threshold_SatInfo[threshold_nb_sat].ERROR_MESSAGE)

        for i in range(threshold_nb_sat):
            # Changing values to not recalculate the data of the first satellite
            threshold_SatInfo[i].COMPLETED = True
            # We check that the name of the new satellite is not in the list
            if threshold_SatInfo[i].NAME == threshold_SatInfo[threshold_nb_sat].NAME:
                # We need to remove the added satellite, we tell it to the user
                logging.info('Threshold - Same satellite name, skipping second satellite')
                threshold_SatInfo[threshold_nb_sat].COMPLETED = False
                threshold_SatInfo[threshold_nb_sat].MESSAGES.append(
                    "The added satellite is already in the visualization. Please use another satellite for comparison.")
                threshold_SatInfo[threshold_nb_sat].NB_MESSAGES = 1
                threshold_to_remove = True
        threshold_nb_sat += 1
        AddedSat = True

    # If the 'file' form is completed and sent
    if formFile.submit.data and formFile.validate():
        logging.info('Threshold - File submitted')

        threshold_SatInfo = []
        threshold_nb_sat = 0

        Seuil = formFile.SeuilBis.data
        ManualVal = formFile.ManualValBis.data
        AutoVal = formFile.AutoValBis.data

        # Saving the uploaded file
        uploaded_file = request.files['fileUpload']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file.save(join(app.config['UPLOAD_PATH'], filename))

        # Path to the uploaded file
        filePath = app.config['UPLOAD_PATH'] + "\\" + filename

        threshold_SatInfo.append(GetSatInfoFile(filePath))
        threshold_nb_sat += 1

    # Global try-except in case there is an unknown error outside the processing functions
    try:
        for i in range(threshold_nb_sat):
            # Converting the launch date to string
            threshold_SatInfo[i].LAUNCH_DATE = str(threshold_SatInfo[i].LAUNCH_DATE)

            if threshold_SatInfo[i].LAUNCH_DATE == "1900-01-01":
                # We remove the default value
                threshold_SatInfo[i].LAUNCH_DATE = "Unknown"

            # If the search is complete
            if threshold_SatInfo[i].COMPLETED:
                # If there is no added satellite, we recalculate everything
                if not AddedSat or i == (threshold_nb_sat - 1):
                    logging.info('Threshold - Getting TLE informations from %s', threshold_SatInfo[i].NAME)

                    # Global try-except in case there is an unknown error outside the processing functions
                    try:
                        # Retrieving the data
                        threshold_SatInfo[i] = GetTleInfo(threshold_SatInfo[i], Seuil, ManualVal, AutoVal)
                    except Exception as e:
                        logging.exception('Threshold - %s', e)
                        threshold_SatInfo[i].ERROR_MESSAGE = "Something went wrong, please send the ssa.log to the devs"
                    # If there is an error, we display it to the user
                    if threshold_SatInfo[i].ERROR_MESSAGE:
                        return render_template('error.html', errorMessage=threshold_SatInfo[i].ERROR_MESSAGE)
                    # Exporting data of the first satellite
                    if i == 0:
                        threshold_SatInfo[i].exportData("threshold")

        if threshold_SatInfo[0].COMPLETED:
            # Plot the orbital parameters
            create_plot_list(threshold_SatInfo, threshold_nb_sat, "threshold")
            if threshold_SatInfo[0].VISU == 'yes':
                # Generate the CZML file only if needed
                czml(threshold_SatInfo, threshold_nb_sat, "threshold")

            logging.info('Threshold - Complete')
    except Exception as e:
        logging.exception('Threshold - %s', e)
        errorMessage = "Something went wrong, please send the ssa.log to the devs"
        return render_template('error.html', errorMessage=errorMessage)

    # Clear the forms
    formSat.SatName.data = ''
    formSat.SatName2.data = ''
    formSat.DateBegin.data = ''
    formSat.DateEnd.data = ''
    formSat.Seuil.data = 'auto'
    formSat.Visu.data = 'no'
    formSat.ManualVal.data = '10'
    formSat.AutoVal.data = '3'
    formAddSat.SatNameAdd.data = ''
    formFile.SeuilBis.data = 'auto'
    formFile.ManualValBis.data = '10'
    formFile.AutoValBis.data = '3'
    formFile.fileUpload.data = None

    return render_template('threshold.html', formSat=formSat, formFile=formFile,
                           formAddSat=formAddSat, SatList=threshold_SatInfo,
                           nb_sat=threshold_nb_sat)


# Download link for a .zip file
@app.route('/export&<request>', methods=['GET'])
def export(request):
    # Default values
    fileType = "application/zip"
    fileName = "data.zip"
    ml_test = False
    ml_file = False

    # Path to the .txt and plots
    path_data_threshold = abspath("app/export/general/threshold/")  # General info
    path_manoeuvre_threshold = abspath("app/export/maneuver/threshold/")  # Maneuvers detected
    path_images_param_threshold = abspath("app/static/images/param/threshold/")  # Plots param
    path_images_maneuver_threshold = abspath("app/static/images/maneuver/threshold/")  # Plots maneuver
    path_images_mahalanobis_threshold = abspath("app/static/images/mahalanobis/threshold/")  # Plot mahalanobis

    path_data_ai = abspath("app/export/general/ai/")  # General info
    path_manoeuvre_ai = abspath("app/export/maneuver/ai/")  # Maneuvers detected
    path_images_param_ai = abspath("app/static/images/param/ai/")  # Plot param
    path_images_maneuver_ai = abspath("app/static/images/maneuver/ai/")  # Plots maneuver
    path_images_mahalanobis_ai = abspath("app/static/images/mahalanobis/ai/")  # Plot mahalanobis

    path_klt = abspath("app/static/images/klt")  # Astrometry results
    path_astrometry = abspath("app/static/images/astrometry")  # Astrometry results

    path_training_file = abspath("app/uploads/")  # Machine Learning training file
    data = BytesIO()

    # Generate the .zip
    with ZipFile(data, "w") as z:
        if "threshold" in request:
            AddFilesToZip(z, path_data_threshold)
        elif "ai" in request:
            AddFilesToZip(z, path_data_ai)

        # Selecting the right files to include
        if request == "threshold-maneuver":
            # .txt
            AddFilesToZip(z, path_manoeuvre_threshold)
            # Energy and Energy Delta
            AddFilesToZip(z, path_images_maneuver_threshold)
        elif request == "threshold-parameters":
            # Orbital parameters
            AddFilesToZip(z, path_images_param_threshold)
        elif request == "threshold-mahalanobis":
            # Mahalanobis
            AddFilesToZip(z, path_images_mahalanobis_threshold)

        elif request == "ai-maneuver":
            # .txt
            AddFilesToZip(z, path_manoeuvre_ai)
            # Energy and Energy Delta (not displayed on the web page)
            AddFilesToZip(z, path_images_maneuver_ai)
        elif request == "ai-parameters":
            # Orbital parameters
            AddFilesToZip(z, path_images_param_ai)
        elif request == "ai-mahalanobis":
            # Mahalanobis
            AddFilesToZip(z, path_images_mahalanobis_ai)
        else:
            if "astrometry" in request:
                folder = request.split("=")[1]
                AddFilesToZip(z, path_astrometry + "/" + folder)
            elif "klt" in request:
                folder = request.split("=")[1]
                AddFilesToZip(z, path_klt + "/" + folder)
            else:
                # Machine Learning training file
                ml_test = True
                for dirname, subdirs, files in walk(path_training_file):
                    for filename in files:
                        if filename == request:
                            ml_file = True
                            absname = abspath(join(dirname, filename))
                            arcname = absname[len(path_training_file) + 1:]
                            z.write(absname, arcname)
                            # AI sat info
                            AddFilesToZip(z, path_data_ai)
                            break

    data.seek(0)

    # Sending the .zip
    if ml_test:
        if not ml_file:
            logging.error('Download - Tried to download %s', request)
            return render_template('error.html', errorMessage="No file to download with this link")
    return send_file(data, mimetype=fileType, as_attachment=True, download_name=fileName)


# KLT main page
@app.route('/klt-processing', methods=['GET', 'POST'])
def klt_processing():
    # Create the form
    formProcess = ProcessImageForm()

    return render_template('klt_processing.html', formProcess=formProcess)


# KLT image processing
@app.route('/klt-processing-submit', methods=['POST'])
def klt_processing_submit():
    global klt_progress
    logging.info('KLT Processing - Request received')
    klt_progress = ImageProcessingProgress()
    files = []
    names = []

    # Saving the uploaded file
    uploaded_file = request.files['fileUpload']
    filename = secure_filename(uploaded_file.filename).lower()
    if filename != '':
        file_ext = splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            klt_progress.setStatus("error")
            abort(400)
        klt_progress.setStatus("file saved")
        uploaded_file.save(join(app.config['UPLOAD_PATH'], filename))
        logging.info('KLT Processing - File saved as ' + filename)
    else:
        klt_progress.setStatus("error")
        logging.error('KLT Processing - Unable to save file ' + filename)
        return jsonify({"response": "pas ok"})

    klt_progress.setStatus("checking file")
    uploaded_filename, uploaded_file_extension = splitext(filename)
    if uploaded_file_extension.__eq__(".zip"):
        logging.info('KLT Processing - ZIP file received, extracting content')
        zipped = zipfile.ZipFile(join(app.config['UPLOAD_PATH'], filename))
        extracted_path = join(app.config['UPLOAD_PATH'], uploaded_filename)
        zipped.extractall(path=extracted_path)

        for dirname, subdirs, zippped_files in walk(extracted_path):
            for filename in zippped_files:
                absname = abspath(join(dirname, filename))

                secured_filename = secure_filename(filename).lower()
                if secured_filename != '':
                    file_ext = splitext(secured_filename)[1]
                    if file_ext not in app.config['IMAGES_EXTENSIONS']:
                        logging.warning('KLT Processing - Extracted file ' + filename + ' is not an image')
                        continue
                    try:
                        os.remove(join(app.config['UPLOAD_PATH'], secured_filename))
                    except Exception as e:
                        logging.info('KLT Processing - No file to overwrite for ' + secured_filename)
                    os.rename(absname, join(app.config['UPLOAD_PATH'], secured_filename))
                    files.append(secured_filename)
                    logging.info('KLT Processing - File saved as ' + secured_filename)
                else:
                    logging.warning('KLT Processing - Extracted file ' + filename + ' does not have a valid name')

        logging.info('KLT Processing - Finished extracting ZIP file')
    else:
        klt_progress.setStatus("error")
        logging.warning('KLT Processing - Single image to process')
        return jsonify({"response": "pas ok"})

    if len(files) < 2:
        klt_progress.setStatus("error")
        logging.error('KLT Processing - Not enough images to process')
        return jsonify({"response": "pas ok :("})

    saving_folder = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")

    for index in range(len(files) - 1):
        file1 = files[index]
        file2 = files[index + 1]
        klt_progress.setStatus('processing ' + splitext(file1)[0] + ' and ' + splitext(file2)[0])
        logging.info('KLT Processing - Processing ' + splitext(file1)[0] + ' and ' + splitext(file2)[0] + ' files')
        names.append(splitext(file1)[0] + "_" + splitext(file2)[0])

        saving_path = "app/static/images/klt/" + saving_folder + "/" + splitext(file1)[0] + "_" + splitext(file2)[
            0] + "/"
        os.makedirs(saving_path)

        args = Namespace(mon=file1,
                         ref=file2,
                         mask=None,
                         conf="app/code/klt/configuration/processing_configuration.json",
                         out=saving_path,
                         resume=False)

        try:
            # set up configuration :
            conf = Configuration(args)
        except ConfigurationError as e:
            klt_progress.setStatus("error")
            return jsonify({"response": "pas ok :("})

        try:
            # run matcher :
            match(args.mon, args.ref, conf, args.resume, args.mask)
        except Exception as e:
            klt_progress.setStatus("cannot process " + splitext(file1)[0] + " and " + splitext(file2)[0])
            logging.error(
                'KLT Processing - Error while processing ' + splitext(file1)[0] + ' and ' + splitext(file2)[0])
            logging.error(e)
            with open(saving_path + "README.txt", 'w') as f:
                f.write("Unable to finnish processing of " + splitext(file1)[0] + " and " + splitext(file2)[0])
            continue
    else:
        names.append(files[len(files) - 1])

    klt_progress.setStatus('moving files')
    logging.info('KLT Processing - Moving files')
    saving_path = "app/static/images/klt/" + saving_folder + "/inputs"
    os.makedirs(saving_path)
    for file in files:
        filePath = join(app.config['UPLOAD_PATH'], file)
        os.rename(filePath, saving_path + "/" + file)

    klt_progress.setStatus("success")
    logging.info('KLT Processing - Complete')
    return jsonify({"response": "ok", "folder": saving_folder, "names": names})


# Check the KLT processing progress
@app.route('/check-klt-progress')
def check_klt_progress():
    global klt_progress
    status = klt_progress.getStatus()
    return jsonify({"status": status})


# Astrometry.net main page
@app.route('/astrometry-processing', methods=['GET', 'POST'])
def astrometry_processing():
    # Create the form
    formProcess = ProcessImageForm()

    return render_template('astrometry_processing.html', formProcess=formProcess)


# Astrometry.net image processing
@app.route('/astrometry-processing-submit', methods=['POST'])
def astrometry_processing_submit():
    global astro_progress
    logging.info('Astrometry Processing - Request received')
    astro_progress = ImageProcessingProgress()
    files = []
    names = []

    # Saving the uploaded file
    uploaded_file = request.files['fileUpload']
    filename = secure_filename(uploaded_file.filename).lower()
    if filename != '':
        file_ext = splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            astro_progress.setStatus("error")
            abort(400)
        astro_progress.setStatus("file saved")
        uploaded_file.save(join(app.config['UPLOAD_PATH'], filename))
        logging.info('Astrometry Processing - File saved as ' + filename)
    else:
        astro_progress.setStatus("error")
        logging.error('Astrometry Processing - Unable to save file ' + filename)
        return jsonify({"response": "pas ok"})

    astro_progress.setStatus("checking file")
    uploaded_filename, uploaded_file_extension = splitext(filename)
    if uploaded_file_extension.__eq__(".zip"):
        logging.info('Astrometry Processing - ZIP file received, extracting content')
        zipped = zipfile.ZipFile(join(app.config['UPLOAD_PATH'], filename))
        extracted_path = join(app.config['UPLOAD_PATH'], uploaded_filename)
        zipped.extractall(path=extracted_path)

        for dirname, subdirs, zippped_files in walk(extracted_path):
            for filename in zippped_files:
                absname = abspath(join(dirname, filename))

                secured_filename = secure_filename(filename).lower()
                if secured_filename != '':
                    file_ext = splitext(secured_filename)[1]
                    if file_ext not in app.config['IMAGES_EXTENSIONS']:
                        logging.warning('Astrometry Processing - Extracted file ' + filename + ' is not an image')
                        continue
                    try:
                        os.remove(join(app.config['UPLOAD_PATH'], secured_filename))
                    except Exception as e:
                        logging.info('Astrometry Processing - No file to overwrite for ' + secured_filename)
                    os.rename(absname, join(app.config['UPLOAD_PATH'], secured_filename))
                    files.append(secured_filename)
                    logging.info('Astrometry Processing - File saved as ' + secured_filename)
                else:
                    logging.warning(
                        'Astrometry Processing - Extracted file ' + filename + ' does not have a valid name')

        logging.info('Astrometry Processing - Finished extracting ZIP file')
    else:
        logging.info('Astrometry Processing - Single image to process')
        if uploaded_file_extension not in app.config['IMAGES_EXTENSIONS']:
            astro_progress.setStatus("error")
            logging.error('Astrometry Processing - File ' + filename + ' is not an image')
            return jsonify({"response": "pas ok :("})
        files.append(filename)

    if len(files) == 0:
        astro_progress.setStatus("error")
        logging.error('Astrometry Processing - No image to process')
        return jsonify({"response": "pas ok :("})
    saving_folder = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")

    first_file_ext = ""
    for filename in files:
        astro_progress.setStatus("processing " + filename)
        logging.info('Astrometry Processing - Processing ' + filename)
        try:
            name, extension = splitext(filename)
            if filename.__eq__(files[0]):
                first_file_ext = extension
            names.append(name)
            output = SubmitAstrometry(saving_folder, name, extension, astro_progress)
        except Exception as e:
            logging.error(e)
            astro_progress.setStatus("error")
            logging.error('Astrometry Processing - Error with astrometry.net')
            return jsonify({"response": "pas ok :("})
        if output.__eq__("error"):
            astro_progress.setStatus("error")
            logging.error('Astrometry Processing - Unable to process the image ' + filename + ' with astrometry.net')
            return jsonify({"response": "pas ok :("})

        try:
            result = RemoveGreenStars(saving_folder, name, extension, astro_progress)
        except Exception as e:
            logging.error(e)
            astro_progress.setStatus("error")
            logging.error('Astrometry Processing - Error while removing stars for the image ' + filename)
            return jsonify({"response": "pas ok :("})
        if result.__eq__("nok"):
            astro_progress.setStatus("error")
            logging.error('Astrometry Processing - Error while creating the mask for the image ' + filename)
            return jsonify({"response": "pas ok :("})

        astro_progress.setStatus(filename + " complete")
        logging.info('Astrometry Processing - ' + filename + ' complete')

    astro_progress.setStatus("success")
    logging.info('Astrometry Processing - Complete')
    return jsonify({"response": "ok", "folder": saving_folder, "names": names, "extension": first_file_ext})


# Check the Astrometry processing progress
@app.route('/check-astrometry-progress')
def check_astrometry_progress():
    global astro_progress
    status = astro_progress.getStatus()
    return jsonify({"status": status})


# Space weather
@app.route('/space-weather')
def space_weather():
    # TODO
    return render_template('index.html')


# Proximity detection
@app.route('/proximity', methods=['GET', 'POST'])
def proximity():
    # TODO
    return render_template('index.html')
