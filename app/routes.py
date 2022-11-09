"""
This file defines the routes used by Flask and the front-end.
Each route defines the main logic of the app:
    - initialize the variables
    - if a form is submitted, contact the APIs and process the satellite data
    - return to the user a web page with all the informations
"""

from flask import (request, render_template, send_file, abort,
                   send_from_directory, jsonify)
from flask import current_app as app
from werkzeug.utils import secure_filename

from zipfile import ZipFile
from os import walk
from os.path import join, splitext, abspath
from io import BytesIO
import logging

from .api import TestAPI, GetSatInfoDW, GetSatInfoST
from .astrometry import SubmitAstrometry
from .classes import SatForm, AddSatForm, SatFile, AIForm, ProximityForm, SatClass, TESTProcessForm
from .functions import GetSatInfoFile, czml, create_plot_list
from .op_progress import Progress
from .remover import RemoveGreenStars
from .threshold import GetTleInfo
from .ai import GetSatInfoAI, GetPredAI, GetManeuversAI, PreparePlotsAI
# TODO from .proximity import GetTleProximity, DetectProximity
# TODO from .SpaceWeather import run as SP_run

# Global variables
# TODO For development only, not optimized when there are multiple users simultaneously
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
# Optical Processing progress
global progress


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
    
    path_training_file = abspath("app/uploads/") # Machine Learning training file
    data = BytesIO()

    # Generate the .zip
    with ZipFile(data, "w") as z:
        if "threshold" in request:
            for dirname, subdirs, files in walk(path_data_threshold):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_data_threshold) + 1:]
                    z.write(absname, arcname)
        elif "ai" in request:
            for dirname, subdirs, files in walk(path_data_ai):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_data_ai) + 1:]
                    z.write(absname, arcname)

        # Selecting the right files to include
        if request == "threshold-maneuver":
            # .txt
            for dirname, subdirs, files in walk(path_manoeuvre_threshold):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_manoeuvre_threshold) + 1:]
                    z.write(absname, arcname)
            # Energy and Energy Delta
            for dirname, subdirs, files in walk(path_images_maneuver_threshold):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_maneuver_threshold) + 1:]
                    z.write(absname, arcname)
        elif request == "threshold-parameters":
            # Orbital parameters
            for dirname, subdirs, files in walk(path_images_param_threshold):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_param_threshold) + 1:]
                    z.write(absname, arcname)
        elif request == "threshold-mahalanobis":
            # Mahalanobis
            for dirname, subdirs, files in walk(path_images_mahalanobis_threshold):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_mahalanobis_threshold) + 1:]
                    z.write(absname, arcname)

        elif request == "ai-maneuver":
            # .txt
            for dirname, subdirs, files in walk(path_manoeuvre_ai):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_manoeuvre_ai) + 1:]
                    z.write(absname, arcname)
            # Energy and Energy Delta (not displayed on the web page)
            for dirname, subdirs, files in walk(path_images_maneuver_ai):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_maneuver_ai) + 1:]
                    z.write(absname, arcname)
        elif request == "ai-parameters":
            # Orbital parameters
            for dirname, subdirs, files in walk(path_images_param_ai):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_param_ai) + 1:]
                    z.write(absname, arcname)
        elif request == "ai-mahalanobis":
            # Mahalanobis
            for dirname, subdirs, files in walk(path_images_mahalanobis_ai):
                for filename in files:
                    absname = abspath(join(dirname, filename))
                    arcname = absname[len(path_images_mahalanobis_ai) + 1:]
                    z.write(absname, arcname)
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
                        for dirname_2, subdirs_2, files_2 in walk(path_data_ai):
                            for filename_2 in files_2:
                                absname_2 = abspath(join(dirname_2, filename_2))
                                arcname_2 = absname_2[len(path_data_ai) + 1:]
                                z.write(absname_2, arcname_2)
                        break

    data.seek(0)

    # Sending the .zip
    if ml_test:
        if not ml_file:
            logging.error('Download - Tried to download %s', request)
            return render_template('error.html', errorMessage="No file to download with this link")
    return send_file(data, mimetype=fileType, as_attachment=True, attachment_filename=fileName)        


# Space image processing
@app.route('/image-processing', methods=['GET', 'POST'])
def image_processing():
    # Create the form
    formProcessTEST = TESTProcessForm()

    return render_template('image_processing.html', formProcessTEST=formProcessTEST)


@app.route('/test-processing-submit', methods=['POST'])
def test_processing_submit():
    global progress
    logging.info('OPTICAL PROCESSING - Request received')
    progress = Progress()

    # Saving the uploaded file
    uploaded_file = request.files['fileUpload']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            progress.setStatus("error")
            abort(400)
        progress.setStatus("file saved")
        uploaded_file.save(join(app.config['UPLOAD_PATH'], filename))
        logging.info('OPTICAL PROCESSING - Images saved as ' + filename)
    else:
        progress.setStatus("error")
        logging.info('OPTICAL PROCESSING - Unable to save image ' + filename)
        return jsonify({"response": "pas ok"})

    try:
        folderId = SubmitAstrometry(filename, progress)
    except Exception as e:
        print(e)
        progress.setStatus("error")
        logging.info('OPTICAL PROCESSING - Error with astrometry.net')
        return jsonify({"response": "pas ok :("})

    if folderId.__eq__("error"):
        logging.info('OPTICAL PROCESSING - Error while processing the image ' + filename + ' with astrometry.net')
        return jsonify({"response": "pas ok :("})

    result = RemoveGreenStars(folderId)
    if result.__eq__("nok"):
        progress.setStatus("error")
        logging.info('OPTICAL PROCESSING - Error while creating the mask for the image ' + filename)
        return jsonify({"response": "pas ok :("})

    progress.setStatus("success")
    return jsonify({"response": "ok", "jobId": folderId})


# Check the optical processing progress
@app.route('/check-progress')
def check_progress():
    global progress
    status = progress.getStatus()
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
