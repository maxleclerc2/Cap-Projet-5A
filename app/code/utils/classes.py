#
#
# INIT SATCLASS
#
#

"""
This file defines the object used by the app.
There are forms for the front-end and the SatClass object containing all informations
about the satellites.
"""

from flask import current_app as app
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, SubmitField, RadioField, SelectField
from wtforms.validators import DataRequired

from numpy import array, transpose, mean, cov, linalg, ones, mat, sqrt, diag

from bokeh.models import (PanTool, BoxZoomTool, WheelZoomTool, ResetTool,
                          SaveTool, HoverTool, Span)
from bokeh.plotting import figure, output_file
from bokeh.palettes import Dark2_5 as palette
from bokeh.resources import CDN
from bokeh.io import save

from datetime import datetime
from threading import Thread

import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse as Ellipse_plt
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import ListedColormap
from matplotlib import patches


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

# Threshold 'new search' form
class SatForm(FlaskForm):
    SatName = StringField('Satellite\'s name:', validators=[DataRequired()])
    SatName2 = StringField('Second satellite\'s name for orbit comparison:')
    DateBegin = StringField('Data start date:', validators=[DataRequired()])
    DateEnd = StringField('Data end date:', validators=[DataRequired()])
    Visu = RadioField('Display the orbit visualization:', choices=[('yes', 'Yes'), ('no', 'No')], default='no',
                      validators=[DataRequired()])
    Seuil = RadioField('Limit of maneuver detection:', choices=[('manual', 'Manual'), ('auto', 'Automatic')],
                       default='auto', validators=[DataRequired()])
    ManualVal = StringField('Value of the limit in J/Kg:', validators=[DataRequired()])
    AutoVal = SelectField('Value of sigma:', choices=[('3', '3'), ('6', '6')], validators=[DataRequired()])
    submit = SubmitField('Submit')


# 'Add satellite' form
class AddSatForm(FlaskForm):
    SatNameAdd = StringField('New satellite\'s name for comparison:', validators=[DataRequired()])
    submit = SubmitField('Add')


# Threshold 'load TLEs from a file' form
class SatFile(FlaskForm):
    fileUpload = FileField('File', validators=[DataRequired()])
    SeuilBis = RadioField('Limit of maneuver detection:', choices=[('manuel', 'Manuel'), ('auto', 'Automatique')],
                          default='auto', validators=[DataRequired()])
    ManualValBis = StringField('Value of the limit in J/Kg:', validators=[DataRequired()])
    AutoValBis = SelectField('Value of sigma:', choices=[('3', '3'), ('6', '6')], validators=[DataRequired()])
    submit = SubmitField('Submit')


# AI 'new search' form
class AIForm(FlaskForm):
    SatName = StringField('Satellite\'s name:')
    DateBegin = StringField('Data start date:')
    DateEnd = StringField('Data end date:')
    fileUpload = FileField('File')
    fileUpload2 = FileField('TLE file')
    Visu = RadioField('Display the orbit visualization:', choices=[('yes', 'Yes'), ('no', 'No')], default='no',
                      validators=[DataRequired()])
    submit = SubmitField('Submit')


# Proximity 'new search' form
class ProximityForm(FlaskForm):
    SatName = StringField('First satellite\'s name:')
    SatName2 = StringField('Second satellite\'s name:')
    DateBegin = StringField('Data start date:')
    DateEnd = StringField('Data end date:')
    Visu = RadioField('Display the orbit visualization:', choices=[('yes', 'Yes'), ('no', 'No')], default='no',
                      validators=[DataRequired()])
    fileUpload = FileField('TLE file 1')
    fileUpload2 = FileField('TLE file 2')
    submit = SubmitField('Submit')


# Image processing form
class ProcessImageForm(FlaskForm):
    fileUpload = FileField('File')
    submit = SubmitField('Submit')


class Date_Values:
    def __init__(self, k, month, day, hour, minute, SEC):
        self.k=k
        self.month=month
        self.day=day
        self.hour=hour
        self.minute=minute
        self.SEC=SEC


# Satellite object with all the informations needed
class SatClass:
    # Initialization
    def __init__(self):
        # General info
        self.SAT_NUM = 0  # Know which satellite it is
        self.COMPLETED = False  # If the search is finished
        self.VISU = 'yes'  # Display the orbit visualization
        self.TLES = []  # All the TLEs
        self.L_1 = []  # Lines 1 of the TLEs
        self.L_2 = []  # Lines 2 of the TLEs
        self.FILE = ""  # File with the TLEs (when loading from a file)
        self.MESSAGES = []  # Messages to display if needed
        self.NB_MESSAGES = 0  # Number of messages
        self.ERROR_MESSAGE = ""  # Error message to display

        # Satellite info
        self.NAME = ""
        self.ID = ""  # NORAD ID
        self.CLASSIFICATION = ""
        self.ORBIT = "/"  # Orbit type
        self.OBJECT_CLASS = ""
        self.LAUNCH_DATE = datetime.strptime("1900-01-01", "%Y-%m-%d").date()  # Default launch date
        self.LAUNCH_SITE = "Unknown"  # Default launch site
        self.MASS = "0Kg"
        self.MASS_BEGIN = 0  # Mass at the beginning date of the search
        self.MASS_END = 0  # Mass at the end date of the search
        self.DIMENSION = ""
        self.DATE_BEGIN = ""  # Date begin of the search
        self.DATE_END = ""  # Date end of the search
        self.LIMIT_TYPE = "manual"  # Threshold manual or automatic
        self.LIMIT_VALUE = 10  # Threshold value
        self.SIGMA_SUP = 0
        self.SIGMA_INF = 0

        # Dates
        self.DATE_TLE = []  # Date of each TLE
        self.NB_DATE = 0  # Number of TLE
        self.DATE_ENERGY_DELTA = []
        self.DATE_MANEUVER = []
        self.DETECTION_MANEUVER = []  # TLE of maneuvers detected
        self.NB_MANEUVER = 0
        self.DATE_MJD = []
        self.TEMPS_MJD = []
        self.DATE_VALUES = []

        # Orbital parameters
        self.MEAN_MOTION = []
        self.ENERGY = []
        self.ENERGY_DELTA = []
        self.ECC = []
        self.ARGP = []
        self.RAAN = []
        self.MEAN_ANOMALY = []
        self.INCLINATION = []
        self.SMA = []
        self.Delta_V = []
        self.Delta_Va = []
        self.Delta_Vi = []
        self.Delta_M = []
        self.Somme_Delta_V = 0
        self.Somme_Delta_M = 0

        # Coordinates
        self.TMP_TEME_SPEED = []
        self.TEME_SPEED = []
        self.ECI_SPEED = []
        self.TMP_TEME_POSITION = []
        self.TEME_POSITION = []
        self.ECI_POSITION = []
        self.POS_X = []
        self.POS_Y = []
        self.POS_Z = []
        self.POS_X_ECEF = []
        self.POS_Y_ECEF = []
        self.POS_Z_ECEF = []
        self.LONGITUDE_ECEF = []
        self.LATITUDE_ECEF = []
        self.D_ECEF = []
        
        # Proximity
        self.DATE_PROXIMITY = []
        self.DIST_PROXIMITY = []
        self.ALT_PERIGEE = []
        self.ORB_PERIOD_PERIGEE = []
        self.ALT_APOGEE = []
        self.ORB_PERIOD_APOGEE = []

        # AI
        self.ALL_TYPE_MANEUVER = None
        self.TYPE_MANEUVER = []
        self.REPORT = None
        self.CONFUSION_MATRIX = None
        self.ACCURACY = 0
        self.AI_DATE = None
        self.DAY_OF_WEEK = None
        self.LAST_TLE = 0
        self.DELTA_T = []
        self.INCLINATION_X = []
        self.INCLINATION_Y = []
        self.ECC_X = []
        self.ECC_Y = []
        self.DELTA_IX = []
        self.DELTA_IY = []
        self.DELTA_EX = []
        self.DELTA_EY = []
        self.DELTA_ENERGY_PREV = []
        self.DELTA_ENERGY_NEXT = []
        self.MEAN_CROSS = []
        self.SPEED_X = []
        self.SPEED_Y = []
        self.SPEED_Z = []
        self.DELTA_MOMENT = []
        self.ALONG_TRACK = []
        self.OUT_OF_PLANE = []
        self.DELTA_ALONG_TRACK = []
        self.DELTA_OUT_OF_PLANE = []
        self.DELTA_ALONG_TRACK_AVG = []
        self.DELTA_QSW = []
        self.DELTA_INERT = []

        # Plot resources
        self.JS_RESSOURCES = CDN.render_js()  # Global JavaScript
        self.CSS_RESSOURCES = CDN.render_css()  # Global CSS
        self.JS_PLOTS = None  # Common JavaScript
        # Orbital parameters plots
        self.PLOT_SMA = None  # Plot
        self.DIV_SMA = None  # HTML div
        self.PLOT_I = None
        self.DIV_I = None
        self.PLOT_ECC = None
        self.DIV_ECC = None
        self.PLOT_ARGP = None
        self.DIV_ARGP = None
        self.PLOT_RAAN = None
        self.DIV_RAAN = None
        self.PLOT_MA = None
        self.DIV_MA = None
        self.PLOT_LONGITUDE = None
        self.DIV_LONGITUDE = None
        self.PLOT_ENERGY = None
        self.DIV_ENERGY = None
        self.PLOT_DELTA_ENERGY = None
        self.DIV_DELTA_ENERGY = None
        self.PLOT_PROXIMITY = None
        self.DIV_PROXIMITY = None

    def make_plot_list(self, orbit_param, SatList, nb_sat, origin):
        """
        Function to plot the orbital parameters.

        Parameters
        ----------
        orbit_param : string
            The orbital parameter to plot
        SatList : list
            List of SatClass object
        nb_sat : int
            Number of satellites in the list
        origin : string
            "ai" or "threshold"
        """
        setTitle = ""  # Default title
        setYLabel = ""  # Default label
        valueFormat = '@y{0,0.00}'  # Default format
        htmlPath = "app/static/images/"  # Default file path
        htmlName = "tmp.html"  # Default file name
        param_list = ""  # List to plot
        date_list = "DATE_TLE"  # Default date list

        # Adding each sat name to the comparison
        comparison = ""
        for i in range(nb_sat):
            if not (i == 0):
                if i == 1:
                    comparison = "".join([" - Comparison with ",
                                          SatList[i].NAME])
                elif i < 3:
                    comparison = "".join([comparison, ", ", SatList[i].NAME])
                else:
                    # Stopping at 3 sat
                    continue

        # Changing the title, label and format for each orbital parameter
        if orbit_param == "Energy":
            setTitle = "".join([self.NAME, " - Change of energy"])
            setYLabel = "Energy (J/Kg)"
            htmlName = "maneuver/" + origin + "/energy.html"
            param_list = "ENERGY"
        elif orbit_param == "Energy Delta":
            # Sigma automatic or manual
            if self.LIMIT_TYPE == "auto":
                setTitle = "".join([self.NAME, " - Energy Delta: maneuver detection\nDetection threshold ",
                                    str(self.LIMIT_VALUE), "-\u03C3"])
            else:
                setTitle = "".join([self.NAME, " - Energy Delta: maneuver detection\nDetection threshold ",
                                    str(self.LIMIT_VALUE), " J/Kg"])
            setYLabel = "Energy Delta (J/Kg)"
            htmlName = "maneuver/" + origin + "/energy_delta.html"
            param_list = "ENERGY_DELTA"
            date_list = "DATE_ENERGY_DELTA"
        elif orbit_param == "SMA":
            setTitle = "".join([self.NAME, " - Change of SMA", comparison])
            setYLabel = "SMA (Km)"
            htmlName = "param/" + origin + "/sma.html"
            param_list = "SMA"
        elif orbit_param == "I":
            setTitle = "".join([self.NAME, " - Change of Inclination", comparison])
            setYLabel = "I (deg)"
            htmlName = "param/" + origin + "/inc.html"
            param_list = "INCLINATION"
        elif orbit_param == "ECC":
            setTitle = "".join([self.NAME, " - Change of Eccentricity", comparison])
            setYLabel = "ECC"
            htmlName = "param/" + origin + "/ecc.html"
            valueFormat = '@y{0.0000000}'
            param_list = "ECC"
        elif orbit_param == "ARGP":
            setTitle = "".join([self.NAME, " - Change of ARGP", comparison])
            setYLabel = "ARGP (deg)"
            htmlName = "param/" + origin + "/argp.html"
            param_list = "ARGP"
        elif orbit_param == "RAAN":
            setTitle = "".join([self.NAME, " - Change of RAAN", comparison])
            setYLabel = "RAAN (deg)"
            htmlName = "param/" + origin + "/raan.html"
            param_list = "RAAN"
        elif orbit_param == "MA":
            setTitle = "".join([self.NAME, " - Change of Mean Anomaly", comparison])
            setYLabel = "MA (deg)"
            htmlName = "param/" + origin + "/mean_anomaly.html"
            param_list = "MEAN_ANOMALY"
        elif orbit_param == "Longitude":
            setTitle = "".join([self.NAME, " - Change of longitude", comparison])
            setYLabel = "Longitude (deg)"
            htmlName = "param/" + origin + "/longitude.html"
            param_list = "LONGITUDE_ECEF"

        # Creating the figure
        p = figure(
            output_backend="webgl",
            title=setTitle,
            x_axis_label="Date (YYYY-MM-DD)",
            x_axis_type='datetime',
            y_axis_label=setYLabel,
            toolbar_location="right",
            tools=[
                PanTool(),  # Move
                BoxZoomTool(),  # Zoom (left click)
                WheelZoomTool(),  # Zoom (mouse wheel)
                ResetTool(),  # Reset the view
                SaveTool(),  # Download the plot
                HoverTool(  # When hovering with the cursor
                    tooltips=[  # Displayed values
                        ('Sat', '$name'),  # Sat name (CAN BE REMOVED FOR BETTER READABILITY WITH MULTIPLE SATS)
                        (setYLabel, valueFormat),  # Orbital parameter and its format
                        ('Date', '@x{%Y-%m-%d %H:%M:%S}'),  # Date
                    ],
                    formatters={'@x': 'datetime'},  # Indicates that the abscissa is a date

                    # display a tooltip whenever the cursor is vertically in line with a glyph
                    mode='vline'
                )
            ],
            sizing_mode="stretch_width",
            max_width=1000,
            height=500,
        )

        # add renderers
        if orbit_param == "Energy Delta":
            # Detected maneuvers
            p.circle(self.DATE_MANEUVER, self.DETECTION_MANEUVER,
                     fill_color="red", size=6, name=self.NAME)
            
            if app.config['DEBUG_MANEUVERS']:  # TODO Change in config.py
                print("\n##### CHECK MANEUVERS #####\n")
                print("Debug menu for checking the number of maneuvers of supported satellites.")
                print("Yellow triangles are added on the Energy Delta plot to display the real maneuvers of the satellite.")
                print("While debugging, be sure to use the same dates for better readability.\n")
                print("Change the value of DEBUG_MANEUVERS in config.py to disable this.\n")
                
                def create_dates(YEARS, DAYS, HOURS):
                    DATES = []
                    
                    for y in range(len(YEARS)):
                        for d in range(len(DAYS[y])):
                            DATES.append(datetime.strptime(str(YEARS[y]) + ' ' + str(DAYS[y][d]) + ' ' + str(HOURS[y][d]), '%Y %j %H'))
                    
                    return DATES
                
                YEARS = []
                days_man = []
                hours_man = []
                
                if self.NAME == "Sentinel-3B":
                    print("----- SENTINEL 3B -----")
                    YEARS = [2018]
                    
                    days_man.append([120, 121, 122, 124, 128, 130, 144, 149, 150, 156, 157, 222, 235, 242, 288, 289, 297, 302, 308, 324, 324, 326, 327, 331, 346])
                    
                    hours_man.append([9, 11, 15, 10, 9, 11, 13, 5, 13, 4, 14, 7, 4, 9, 3, 10, 9, 15, 10, 13, 16, 20, 20, 8, 13])
                
                elif self.NAME == "SARAL":
                    print("----- SARAL -----")
                    YEARS = [2013, 2014, 2015, 2016, 2017, 2019]
                    
                    days_man.append([58, 58, 60, 61, 62, 64, 72, 82, 103, 135, 158, 175, 195, 210, 212, 219, 238, 276, 280, 282, 301, 311, 329, 346, 361])
                    days_man.append([28, 62, 85, 108, 139, 90, 225, 255, 279, 283, 289, 308, 325, 344, 360])
                    days_man.append([22, 52, 90, 99, 146, 146, 167, 189, 223, 236, 294, 316, 330])
                    days_man.append([7, 78, 98, 186])
                    days_man.append([336])
                    days_man.append([152])
                    
                    hours_man.append([13, 14, 15, 6, 2, 1, 0, 12, 12, 12, 12, 13, 14, 0, 14, 13, 13, 13, 13, 14, 14, 13, 14, 13, 14])
                    hours_man.append([12, 14, 12, 12, 14, 12, 12, 13, 12, 12, 12, 12, 13, 13, 13])
                    hours_man.append([14, 13, 13, 12, 10, 12, 13, 13, 12, 12, 13, 13, 13])
                    hours_man.append([14, 15, 13, 12])
                    hours_man.append([12])
                    hours_man.append([13])
                    
                elif self.NAME == "Jason-3":
                    print("----- JASON-3 -----")
                    YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
                    
                    days_man.append([19, 21, 28, 31, 33, 34, 35, 38, 40, 42, 54, 95, 140, 207, 258, 357])
                    days_man.append([102, 249, 346])
                    days_man.append([94, 231, 352])
                    days_man.append([104, 167, 218, 337])
                    days_man.append([72, 205, 303])
                    days_man.append([35])
                    
                    hours_man.append([22, 22, 22, 21, 3, 20, 21, 22, 23, 22, 2, 1, 20, 2, 20, 21])
                    hours_man.append([23, 16, 19])
                    hours_man.append([0, 17, 17])
                    hours_man.append([20, 19, 18, 15])
                    hours_man.append([0, 21, 20])
                    hours_man.append([0])
                    
                elif self.NAME == "Envisat":
                    print("----- ENVISAT -----")
                    YEARS = [2002, 2003, 2004, 2005, 2006, 2007]
                    
                    days_man.append([109, 129, 152, 187, 218, 238, 252, 254, 268, 290, 311, 333, 352, 352])
                    days_man.append([14, 42, 52, 62, 64, 134, 140, 158, 192, 227, 273, 301, 304, 322, 349, 360])
                    days_man.append([21, 26, 35, 36, 55, 98, 105, 128, 182, 230, 245, 246, 265, 268, 296, 317, 352])
                    days_man.append([5, 7, 49, 76, 89, 91, 140, 159, 221, 250, 279, 336])
                    days_man.append([4, 10, 19, 87, 90, 152, 171, 255, 256, 319, 354])
                    days_man.append([23, 24, 53, 93, 95, 136, 166, 191, 198, 199, 221, 270, 271, 298, 333, 338, 341])
                    
                    hours_man.append([4, 5, 10, 3, 3, 17, 23, 1, 22, 19, 18, 3, 4, 22])
                    hours_man.append([0, 23, 3, 23, 0, 22, 4, 1, 0, 1, 0, 4, 1, 23, 21, 21])
                    hours_man.append([23, 22, 4, 11, 11, 20, 4, 1, 8, 2, 22, 23, 4, 3, 3, 1, 1])
                    hours_man.append([23, 4, 1, 4, 22, 22, 0, 19, 22, 5, 2, 0])
                    hours_man.append([5, 5, 23, 5, 5, 0, 1, 2, 5, 3, 23])
                    hours_man.append([4, 2, 3, 4, 3, 3, 1, 3, 4, 3, 2, 5, 3, 3, 3, 4, 2])
                    
                elif self.NAME == "Sentinel-3A":
                    print("----- SENTINEL-A -----")
                    YEARS = [2016, 2017, 2018, 2019]
                    
                    days_man.append([53, 54, 55, 57, 62, 67, 81, 83, 104, 110, 154, 203, 244, 266, 306, 336, 349])
                    days_man.append([54, 74, 117, 143, 193, 249, 270, 333, 347])
                    days_man.append([59, 73, 144, 213, 241, 332, 353])
                    days_man.append([58, 72, 164, 240, 331, 345])
                    
                    hours_man.append([9, 10, 19, 9, 12, 12, 11, 13, 9, 12, 11, 12, 7, 7, 12, 8, 8])
                    hours_man.append([9, 7, 10, 14, 9, 10, 8, 9, 8])
                    hours_man.append([10, 8, 8, 8, 7, 13, 9])
                    hours_man.append([9, 8, 8, 12, 8, 11])
                    
                elif self.NAME == "TOPEX-Poseidon":
                    print("----- TOPEX-POSEIDON -----")
                    YEARS = [1992, 1993, 1994, 1995, 1996, 1998, 1999, 2000, 2001, 2002, 2003, 2004]
                    
                    days_man.append([230, 233, 240, 246, 252, 258, 265, 286, 356])
                    days_man.append([89, 218])
                    days_man.append([31, 140, 279])
                    days_man.append([142])
                    days_man.append([15])
                    days_man.append([335])
                    days_man.append([159, 289])
                    days_man.append([12, 111, 190, 270])
                    days_man.append([13, 122, 261, 330])
                    days_man.append([35, 124, 227, 231, 235, 253, 256, 259, 352])
                    days_man.append([116])
                    days_man.append([216, 267, 267, 267, 322, 322])
                    
                    hours_man.append([18, 17, 18, 18, 18, 17, 19, 23, 9])
                    hours_man.append([12, 10])
                    hours_man.append([20, 23, 18])
                    hours_man.append([22])
                    hours_man.append([19])
                    hours_man.append([20])
                    hours_man.append([7, 22])
                    hours_man.append([10, 15, 23, 6])
                    hours_man.append([7, 10, 5, 14])
                    hours_man.append([0, 6, 19, 18, 18, 21, 20, 19, 8])
                    hours_man.append([5])
                    hours_man.append([18, 18, 19, 20, 18, 20])
                
                elif self.NAME == "CRYOSAT 2":
                    print("----- CRYOSAT 2 -----")
                    YEARS = [2010, 2011, 2012]
                    
                    days_man.append([105, 123, 124, 125, 126, 138, 138, 140, 140, 147, 148, 166, 169, 201, 275, 301, 350])
                    days_man.append([21, 70, 104, 152, 209, 244, 272, 294, 313, 333, 355])
                    days_man.append([47, 75, 82, 110, 136, 164, 194, 208, 226, 264, 292, 334])
                    
                    hours_man.append([17, 17, 18, 17, 18, 0, 23, 0, 23, 0, 1, 3, 1, 12, 3, 6, 3])
                    hours_man.append([10, 8, 9, 6, 4, 11, 8, 9, 12, 7, 9])
                    hours_man.append([6, 14, 14, 11, 11, 9, 8, 6, 7, 8, 7, 12])
                    
                else:
                    print("No maneuvers registered for ", self.NAME)
                
                dates_man = create_dates(YEARS, days_man, hours_man)
                
                print("Number of real maneuvers registered: ", len(dates_man))
                print("Number of maneuvers detected: ", self.NB_MANEUVER)
                if self.LIMIT_TYPE == "auto":
                    print("Threshold: ", self.SIGMA_SUP, " J/Kg")
                else:
                    print("Threshold: ", self.LIMIT_VALUE, " J/Kg")
                if self.NB_MANEUVER > len(dates_man):
                    print("More maneuvers are detected. Try with a higher threshold.")
                elif self.NB_MANEUVER < len(dates_man):
                    print("Less maneuvers are detected. Try with a lower threshold.")
                else:
                    print("The same amount of maneuvers are detected.")
                
                energy = [0 for tmp in dates_man]
                p.triangle(dates_man, energy, fill_color="yellow", size=6, name=self.NAME)
                print("\n##### END CHECK #####\n")
            
            # Adding threshold limit for the Energy Delta
            if self.LIMIT_TYPE == "auto":
                hline1 = Span(location=self.SIGMA_SUP, dimension='width',
                              line_color='red', line_width=1)
                hline2 = Span(location=self.SIGMA_INF, dimension='width',
                              line_color='red', line_width=1)
            else:
                hline1 = Span(location=self.LIMIT_VALUE, dimension='width',
                              line_color='red', line_width=1)
                hline2 = Span(location=-self.LIMIT_VALUE, dimension='width',
                              line_color='red', line_width=1)
            # Extending the lines into infinite
            p.renderers.extend([hline1, hline2])

        # On trace le graphe
        if orbit_param == "Energy" or orbit_param == "Energy Delta":
            # Energy and Energy Delta only for the first satellite
            p.line(getattr(self, date_list), getattr(self, param_list),
                   legend_label=self.NAME, line_width=2, name=self.NAME)
        else:
            # create a color iterator
            colors = itertools.cycle(palette)
            # Plotting for each satellite
            for i, color in zip(range(nb_sat), colors):
                if i < 3:
                    p.line(getattr(SatList[i], date_list), getattr(SatList[i], param_list),
                           legend_label=SatList[i].NAME, line_color=color, line_width=2,
                           name=SatList[i].NAME)
                else:
                    # Breaking the loop if there are more than 3 sat
                    continue

        # Saving the plot in the right variable
        if orbit_param == "Energy":
            self.PLOT_ENERGY = p  # Graphe
        elif orbit_param == "Energy Delta":
            self.PLOT_DELTA_ENERGY = p
        elif orbit_param == "SMA":
            self.PLOT_SMA = p
        elif orbit_param == "I":
            self.PLOT_I = p
        elif orbit_param == "ECC":
            self.PLOT_ECC = p
        elif orbit_param == "ARGP":
            self.PLOT_ARGP = p
        elif orbit_param == "RAAN":
            self.PLOT_RAAN = p
        elif orbit_param == "MA":
            self.PLOT_MA = p
        elif orbit_param == "Longitude":
            self.PLOT_LONGITUDE = p

        # Saves the plot as HTML
        output_file("".join([htmlPath, htmlName]))
        save(p, title=orbit_param)

    def make_plot_mahalanobis(self, param, origin):
        """
        Draw the Mahalanobis plots

        Parameters
        ----------
        param : string
            Which plot to draw
        origin : string
            "ai" or "threshold"
        """
        savePath = "".join(["app/static/images/mahalanobis/", origin, "/"])

        if param == "INC_SMA":
            x_list = "SMA"
            y_list = "INCLINATION"
        elif param == "LON_INC":
            x_list = "INCLINATION"
            y_list = "LONGITUDE_ECEF"

        x = array(getattr(self, x_list))
        x = transpose(x)

        y = array(getattr(self, y_list))
        y = transpose(y)

        r = [x, y]
        mu = [mean(r[0]), mean(r[1])]
        mat_cov = cov(transpose(x), transpose(y))
        sigma = linalg.inv(mat_cov)

        delta1 = x - ones(len(x), dtype=float) * mu[0]
        delta2 = y - ones(len(y), dtype=float) * mu[1]
        delta = transpose([delta1, delta2])

        delta_prime = [delta1, delta2]

        mat_mah = mat(delta) * mat(sigma) * mat(delta_prime)
        dist_mah = sqrt(diag(mat_mah))
        ind1 = []
        ind3 = []
        ind5 = []
        ind_ano = []

        for i in range(len(dist_mah)):
            if dist_mah[i] < 1:
                ind1.append(i)
            if 3 > dist_mah[i] >= 1:
                ind3.append(i)
            if 5 > dist_mah[i] >= 3:
                ind5.append(i)
            if dist_mah[i] >= 3:
                ind_ano.append(i)

        fig, ax = plt.subplots(figsize=(9, 6))

        def threading_func_1():
            for i in ind1:
                ax.plot(getattr(self, x_list)[i], getattr(self, y_list)[i], 'go', mfc='none')

        def threading_func_2():
            for i in ind3:
                ax.plot(getattr(self, x_list)[i], getattr(self, y_list)[i], 'bo', mfc='none')

        def threading_func_3():
            for i in ind_ano:
                ax.plot(getattr(self, x_list)[i], getattr(self, y_list)[i], 'ro', mfc='none')

        def threading_func_4():
            for i in ind5:
                ax.plot(getattr(self, x_list)[i], getattr(self, y_list)[i], 'ro', mfc='none')

        thread_1 = Thread(target=threading_func_1)
        thread_2 = Thread(target=threading_func_2)
        thread_3 = Thread(target=threading_func_3)
        thread_4 = Thread(target=threading_func_4)

        thread_1.start()
        thread_2.start()
        thread_3.start()
        thread_4.start()

        # ellipse
        ax.scatter(x, y, s=0.5)

        confidence_ellipse(x, y, ax, n_std=1, label=r'1$\sigma$', edgecolor='blue')
        # confidence_ellipse(x, y, ax, n_std=2, label=r'2$\sigma$', edgecolor='fuchsia', linestyle='--')
        confidence_ellipse(x, y, ax, n_std=3.05, label=r'3$\sigma$', edgecolor='red')

        # ax.scatter(mu[0], mu[1], c='red', s=3)
        ax.set_title('Different standard deviations')
        ax.legend()

        if param == "INC_SMA":
            ax.set_title("Changes in inclination depending on SMA of " + self.NAME, fontsize=20)
            ax.set_xlabel('SMA (km)')
            ax.set_ylabel('Inclination (deg)')
            saveName = "".join([savePath, 'mahalanobis_inc_sma.jpg'])
        if param == "LON_INC":
            ax.set_title("Changes in longitude depending on inclination of " + self.NAME, fontsize=20)
            ax.set_xlabel('Inclination (deg)')
            ax.set_ylabel('Longitude (deg)')
            saveName = "".join([savePath, 'mahalanobis_lon_inc.jpg'])
        ax.grid()

        thread_1.join()
        thread_2.join()
        thread_3.join()
        thread_4.join()

        # Save the plot (with ellipses) in the folder:
        # static/images/mahalanobis/ai
        fig.savefig(saveName, bbox_inches='tight', dpi=150)

    def make_plot_AI(self):
        """
        Make the 'recap maneuvers' plot for the AI search
        """
        plt.figure(figsize=(15, 10))

        classes = []
        class_colours = []

        # Chek how many maneuver type there is
        if 0 in self.ALL_TYPE_MANEUVER.unique():
            classes.append('No Maneuver')
            class_colours.append('w')
        if 1 in self.ALL_TYPE_MANEUVER.unique():
            classes.append('East/West')
            class_colours.append('b')
        if 2 in self.ALL_TYPE_MANEUVER.unique():
            classes.append('North/South')
            class_colours.append('g')
        if 3 in self.ALL_TYPE_MANEUVER.unique():
            classes.append('Orbit relocation')
            class_colours.append('r')

        plt.ylim(self.AI_DATE.iloc[-1], self.AI_DATE[0])
        x = [0, 1, 2, 3, 4, 5, 6]
        labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        plt.scatter(self.DAY_OF_WEEK, self.AI_DATE, c=self.ALL_TYPE_MANEUVER, linewidths=8, marker="_",
                    cmap=ListedColormap(class_colours), s=400 * 35)
        plt.xticks(x, labels, fontsize=11.5)

        recs = []
        for i in range(0, len(class_colours)):
            recs.append(patches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
        plt.legend(recs, classes, bbox_to_anchor=(1, 1))
        plt.title("Recap chart of the maneuvers of " + self.NAME + " from " + self.DATE_BEGIN + " to " + self.DATE_END,
                  fontsize=20)

        # Save the plot in the folder:
        # static/images/maneuver/ai
        plt.savefig("app/static/images/maneuver/ai/recap.jpg", bbox_inches='tight', dpi=150)

    def make_plot_proximity(self, SatList):
        """
        Function to plot the orbital distance betwen 2 satellites.

        Parameters
        ----------
        SatList : list
            List with 2 SatClass objects
        """
        #if self.ORBIT == "GEO":
        # Distance plot only for GEO
        # See the second method in .proximity.DetectProximity to enable it for LEO ssatellites
        setTitle = "".join([SatList[0].NAME, " approch by ", SatList[1].NAME])
        setYLabel = "Distance (Km)"
        valueFormat = '@y{0,0.00}'  # Default format
        fileName = "app/static/images/proximity/proximity.html"

        # Creating the figure
        p = figure(
            output_backend="webgl",
            title=setTitle,
            x_axis_label="Date (YYYY-MM-DD)",
            x_axis_type='datetime',
            y_axis_label=setYLabel,
            toolbar_location="right",
            tools=[
                PanTool(),  # Move
                BoxZoomTool(),  # Zoom (left click)
                WheelZoomTool(),  # Zoom (mouse wheel)
                ResetTool(),  # Reset the view
                SaveTool(),  # Download the plot
                HoverTool(  # When hovering with the cursor
                    tooltips=[  # Displayed values
                        (setYLabel, valueFormat),  # Orbital parameter and its format
                        ('Date', '@x{%Y-%m-%d %H:%M:%S}'),  # Date
                    ],
                    formatters={'@x': 'datetime'},  # Indicates that the abscissa is a date

                    # display a tooltip whenever the cursor is vertically in line with a glyph
                    mode='vline'
                )
            ],
            sizing_mode="stretch_width",
            max_width=1000,
            height=500,
        )

        p.line(self.DATE_PROXIMITY, self.DIST_PROXIMITY, line_width=2)  # Line with all the values
        val, idx = min((val, idx) for (idx, val) in enumerate(self.DIST_PROXIMITY))  # Get the index of the min value
        p.circle(self.DATE_PROXIMITY[idx], self.DIST_PROXIMITY[idx], fill_color="red", size=8)  # Red dot at the min distance
        
        self.PLOT_PROXIMITY = p

        # Saves the plot as HTML
        output_file(fileName)
        save(p)
        """
        else:
            # Gabbard diagram for LEO satellites
            # Check .proximity.DetectProximity
            setTitle = "".join(["Gabbard diagram of ", SatList[0].NAME, " and ", SatList[1].NAME])
            setXLabel = "Orbital period (min)"
            setYLabel = "Altitude (Km)"
            valueXFormat = '@x{0,0.00}'  # Default format
            valueYFormat = '@y{0,0.00}'  # Default format
            fileName = "app/static/images/proximity/proximity.html"

            # Creating the figure
            p = figure(
                output_backend="webgl",
                title=setTitle,
                x_axis_label=setXLabel,
                y_axis_label=setYLabel,
                toolbar_location="right",
                tools=[
                    PanTool(),  # Move
                    BoxZoomTool(),  # Zoom (left click)
                    WheelZoomTool(),  # Zoom (mouse wheel)
                    ResetTool(),  # Reset the view
                    SaveTool(),  # Download the plot
                    HoverTool(  # When hovering with the cursor
                        tooltips=[  # Displayed values
                            (setYLabel, valueYFormat),
                            (setXLabel, valueXFormat),
                        ],
                    )
                ],
                sizing_mode="stretch_width",
                max_width=1000,
                height=500,
            )
            
            # First satellite
            p.scatter(self.ORB_PERIOD_APOGEE, self.ALT_APOGEE, fill_alpha=0.4,
                     marker='hex', color='red', legend_label="".join([self.NAME, " - Apogee"]))
            p.scatter(self.ORB_PERIOD_PERIGEE, self.ALT_PERIGEE, fill_alpha=0.4,
                     marker='hex', color='blue', legend_label="".join([self.NAME, " - Perigee"]))
            
            # Second satellite
            p.scatter(SatList[1].ORB_PERIOD_APOGEE, SatList[1].ALT_APOGEE, fill_alpha=0.4,
                     marker='triangle', color='red', legend_label="".join([SatList[1].NAME, " - Apogee"]))
            p.scatter(SatList[1].ORB_PERIOD_PERIGEE, SatList[1].ALT_PERIGEE, fill_alpha=0.4,
                     marker='triangle', color='blue', legend_label="".join([SatList[1].NAME, " - Perigee"]))

        self.PLOT_PROXIMITY = p

        # Saves the plot as HTML
        output_file(fileName)
        save(p)
    """
        
    def exportData(self, origin):
        """
        Function to write all informations in files

        Parameters
        ----------
        origin : string
            "ai" or "threshold"
        """
        # General informations
        with open("app/export/general/" + origin + "/SatInfo.txt", "w") as fo:
            fo.write("".join(["Data of ", self.NAME, " :\n\n"]))
            fo.write("".join(["NORAD ID: ", str(self.ID), "\n"]))
            fo.write("".join(["Classification: ", self.CLASSIFICATION, "\n"]))
            fo.write("".join(["Orbit type: ", self.ORBIT, "\n"]))
            fo.write("".join(["Object class: ", self.OBJECT_CLASS, "\n"]))
            fo.write("".join(["Launch date: ", self.LAUNCH_DATE, "\n"]))
            fo.write("".join(["Launch site: ", self.LAUNCH_SITE, "\n"]))
            fo.write("".join(["Initial mass: ", self.MASS, "\n"]))
            fo.write("".join(["Dimension: ", self.DIMENSION, "\n"]))
            fo.write("".join(["Data recovery dates: ", self.DATE_BEGIN, " to ", self.DATE_END, "\n"]))
            fo.write("".join(["Number of maneuvers detected: ", str(self.NB_MANEUVER), "\n"]))

        # Dates of the maneuvers
        with open("app/export/maneuver/" + origin + "/maneuvers.txt", "w") as fo:
            strSeuil = "\n"

            if origin == "threshold":
                # Threshold limit
                if self.LIMIT_TYPE == "auto":
                    strSeuil = "".join(["Detection threshold ",
                                        str(self.LIMIT_VALUE), "-sigma\n\n"])
                else:
                    strSeuil = "".join(["Detection threshold ",
                                        str(self.LIMIT_VALUE), " J/Kg\n\n"])

            # Sat info
            fo.write("".join([self.NAME, " - Maneuvers from ", self.DATE_BEGIN,
                              " to ", self.DATE_END, "\n", strSeuil]))

            # Check if there are maneuvers
            if not self.DATE_MANEUVER:
                fo.write("No maneuver was detected in the period selected with this threshold.")
            else:
                # Write each maneuver date
                fo.write("".join(["Number of maneuvers detected: ", str(self.NB_MANEUVER), "\n\n"]))
                fo.write("".join(["N° man. |      Date of maneuvers      |"]))
                if origin == "ai":
                    fo.write("".join([" Man. type |"]))
                fo.write("".join(["   Delta V (m/s)   |    Delta M (kg)    \n"]))
                for i in range(self.NB_MANEUVER):
                    fo.write("".join([str(i + 1), "       | ", str(self.DATE_MANEUVER[i]), " | "]))
                    if origin == "ai":
                        if self.TYPE_MANEUVER[i] == 1:
                            fo.write("".join(["East/West | "]))
                        elif self.TYPE_MANEUVER[i] == 2:
                            fo.write("".join(["South/North | "]))
                        else:
                            fo.write("".join(["Relocation | "]))
                    fo.write("".join([str(self.Delta_V[i]), " | ", str(self.Delta_M[i]), "\n"]))
        
        
    def clear(self):
        """
        Clears all the field requiered for the plots.
        Used by the AI service to clear the 6h values to process the TLES with a 12h filter.

        Returns
        -------
        None.

        """
        self.MEAN_MOTION = []
        self.ENERGY = []
        self.ENERGY_DELTA = []
        self.ECC = []
        self.ARGP = []
        self.RAAN = []
        self.MEAN_ANOMALY = []
        self.INCLINATION = []
        self.SMA = []
        
        self.DATE_VALUES = []
        self.DATE_TLE = []
        self.TEMPS_MJD = []
        
        self.TEME_POSITION = []
        self.TEME_SPEED = []
        self.POS_X_ECEF = []
        self.POS_Y_ECEF = []
        self.POS_Z_ECEF = []


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same size")

    covariance = cov(x, y)
    pearson = covariance[0, 1] / sqrt(covariance[0, 0] * covariance[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = sqrt(1 + pearson)
    ell_radius_y = sqrt(1 - pearson)
    ellipse = Ellipse_plt((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = sqrt(covariance[0, 0]) * n_std
    mean_x = mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = sqrt(covariance[1, 1]) * n_std
    mean_y = mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
