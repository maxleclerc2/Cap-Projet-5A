from setuptools import setup, find_packages
from os.path import dirname

# avoid package import
version = {}
with open(dirname(__file__) + "/klt_matcher/version.py") as fp:
    exec(fp.read(), version)

setup(name="klt_matcher", packages=find_packages(), version=version['__version__'])
