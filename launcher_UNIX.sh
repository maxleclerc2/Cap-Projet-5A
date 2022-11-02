#!/bin/bash

if [ ! -d './ssa-venv/' ] ; then
 python3.9 -m venv ssa-venv
 chmod +x ./ssa-venv/bin/activate
fi

./ssa-venv/bin/activate
pip install -r ./requirement.txt
FLASK_APP=wsgi.py
export FLASK_APP=wsgi.py
flask run
