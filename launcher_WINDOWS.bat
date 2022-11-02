@echo off

if exist "ssa-venv\" (
  echo "Virtual environment already created. Checking requirements..."
) else (
  echo "Creating virtual environment..."
  python3.9 -m venv ssa-venv
)

cmd /k "ssa-venv\Scripts\activate & pip install -r requirement.txt & set FLASK_APP = 'wsgi.py' & flask run"
