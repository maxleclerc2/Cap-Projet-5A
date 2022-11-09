""" Flask configuration """

CACHE_TYPE = 'SimpleCache'  # Flask-Caching related configs
CACHE_DEFAULT_TIMEOUT = 1500
THREADS_PER_PAGE = 4  # Number of threads for the app
SECRET_KEY = 'R4nD0m$k3Y'  # TODO Secret key for the forms
UPLOAD_EXTENSIONS = ['.txt', '.csv', '.xlsx', '.png', '.jpg', '.jpeg', '.tiff']  # TODO Files accepted for upload
UPLOAD_PATH = 'app/uploads'  # Path to the folder 'upload'
DEBUG_MANEUVERS = False  # TODO Display the real maneuvers or not on the plots
