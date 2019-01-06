"""
Created on 21/12/2018
Author: Philip.P_adm

Organise files by extension and move to a folder
"""
import os
import shutil

sourcePath = "C:/Users/ppapasav/Documents/"
sourceFiles = os.listdir(sourcePath)

destPath = "C:/Users/ppapasav/Documents/Excel documents"
if os.path.exists(destPath):
    print("Excel dump exists: {}".format(destPath))
else:
    print("Creating directory: {}".format(destPath))
    os.mkdir(destPath)

# Sort into Folders --> Excel (xlsx) files, and also Photos (jpg)
for file in sourceFiles:
    if file.endswith(".xlsx"):
        shutil.move(os.path.join(sourcePath, file), os.path.join(destPath,file))
    elif file.endswith(".jpg"):
        if os.path.exists(sourcePath + "Pictures/"):
            "Directory for photos exists"
        else:
            print("Creating directory for photos {}".format(sourcePath + "Pictures"))
            os.mkdir(sourcePath + "Pictures")
            shutil.move(os.path.join(sourcePath, file), os.path.join(sourcePath + "Pictures", file))
    else:
        print("No more files to organise")


#### This is for the desktop stuff

# bad way of getting the file extension names
# [os.path.splitext(i)[1] for i in os.listdir("C:/Users/ppapasav/Desktop/")]
# most popular found to be .csv, .pdf, .PNG, .jpg

sourcePath = "C:/Users/ppapasav/Desktop/"
sourceFiles = os.listdir(sourcePath)

destList = [sourcePath+ "Docs", sourcePath+ "PDFs", sourcePath+ "Photos"]
destPath = "C:/Users/ppapasav/Documents/Excel documents"
if os.path.exists(destPath):
    print("Excel dump exists: {}".format(destPath))
else:
    print("Creating directory: {}".format(destPath))
    os.mkdir(destPath)

for file in sourceFiles:
    if file.endswith(".xlsx"):
        shutil.move(os.path.join(sourcePath, file), os.path.join(destPath, file))
    elif file.endswith(".jpg"):
        if os.path.exists(sourcePath + "Pictures/"):
            "Directory for photos exists"
        else:
            print("Creating directory for photos {}".format(sourcePath + "Pictures"))
            os.mkdir(sourcePath + "Pictures")
            shutil.move(os.path.join(sourcePath, file), os.path.join(sourcePath + "Pictures", file))
    else:
        print("No more files to organise")
