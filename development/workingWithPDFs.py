"""
Script to extract text from a PDF document, from the AutomateTheBoringStuff
Python book
"""

import PyPDF2
import os
os.chdir("C:\\Users\\ppapasav\\Documents\\myPythonScripts")

pdfFile = open('meetingminutes1.pdf', 'rb')
reader = PyPDF2.PdfFileReader(pdfFile)
reader.numPages
page = reader.getPage(0)
page.extractText()

#get all text and print
for pageNum in range(reader.numPages):
    print(reader.getPage(pageNum).extractText())
