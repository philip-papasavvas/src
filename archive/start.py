# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 07:55:08 2018

@author: ppapasav
"""
import numpy as np
import pandas as pd
import os
import bs4
import quandl

basePath = 'C:\\Users\\ppapasav\\My Documents\\python'

inputPath = basePath + "\\input"
outputPath = basePath + "\\output"

if not os.path.exists(inputPath):
    os.mkdir(inputPath)

if not os.path.exists(outputPath):
    os.mkdir(outputPath)