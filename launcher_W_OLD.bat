@echo off

if exist "ssa-venv\" (
  echo "Virtual environment already created. Checking requirements..."
) else (
  echo "Creating virtual environment..."
  python3.9 -m venv ssa-venv
)

cmd /k "ssa-venv\Scripts\activate & pip install -r requirement.txt & set FLASK_APP = 'wsgi.py' & flask run"

cmd /k "%appdata%\.anaconda\navigator\.anaconda\navigator\scripts\console_shortcut.bat & conda create --name ssa python=3.9 --file requirement.txt -c conda-forge & conda activate ssa & set FLASK_APP = 'wsgi.py' & flask run"