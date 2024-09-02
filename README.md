# Projects

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AIAlpha/blob/master/LICENSE) 
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This repository is mainly focused around the analysis of time-series financial security data, incorporating the main packages:
* dataload: modules for connecting to cloud MongoDB (non-relational/NoSQL database), for data read/write, and module for downloading stock price data from yahoo finance API [yfinance](https://github.com/ranaroussi/yfinance)
* jupyter-notebooks: notebooks on concepts for time-series analysis such as Auto-Regression, Efficient Frontier of a portfolio with multiple securities, Stationarity, and some tips & tricks in Python
* sql_zoo: the practice of using SQL from the online exercises [SQLZOO](https://www.sqlzoo.net/)
* tests: unit tests for the specific utils files (date, generic, lists)

# Virtual Environment
It's good practice to set up a virtual environment to keep the project requirements separate and enable others to reproduce your results.
Using the virtualenv package in Python, it's easy to set up a virtual environment.
- If not installed already, install venv:
``` pip install virtualenv```
- To use the virtual environment in the project, in the terminal, create a project folder, and change the directory to that project directory, then run the following:
```python<version> -m venv <virtual-environment-name>```
- To activate and then use the virtual environment in your project, run the following
```source env/bin/activate```
  

## To-do
- [X] Add a requirements.txt for compatibility (used pipreqs)
- [X] Re-word Auto-Regression notebook (in Jupyter Notebooks)

## Contributing
Pull requests are welcome.
