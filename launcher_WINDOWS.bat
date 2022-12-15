@echo on

cmd /k "%userprofile%\Anaconda3\Scripts\activate.bat %userprofile%\Anaconda3 & conda create --name ssa python=3.9 gdal -c conda-forge & conda activate ssa & pip install -r requirement.txt & set FLASK_APP = 'wsgi.py' & flask run"
