"""
Created on 21 Dec 2018 by Philip.P_adm
Organise files by extension and move to a folder
"""

import os
import shutil

source_path = "C:/Users/ppapasav/Documents/"
source_files = os.listdir(source_path)
dest_path = ""

# Sort into Folders --> Excel (xlsx) files, and also Photos (jpg)
for file in source_files:
    if file.endswith(".xlsx"):
        shutil.move(src=os.path.join(source_path, file),
                    dst=os.path.join(dest_path, file))
    elif file.endswith(".jpg"):
        if os.path.exists(source_path + "Pictures"):
            "Directory for photos exists"
        else:
            print(f'Creating directory for photos {source_path + "/Pictures"}')
            os.mkdir(source_path + "Pictures")
            shutil.move(src=os.path.join(source_path, file),
                        dst=os.path.join(source_path + "Pictures", file))
    else:
        print("No more files to organise")


# bad way of getting the file extension names
# [os.path.splitext(i)[1] for i in os.listdir("C:/Users/ppapasav/Desktop/")]
# most popular found to be .csv, .pdf, .PNG, .jpg

source_path = "C:/Users/ppapasav/Desktop/"
source_files = os.listdir(source_path)

destination_list = [source_path + "Docs", source_path + "PDFs", source_path + "Photos"]
destination_path = "C:/Users/ppapasav/Documents/Excel documents"
if os.path.exists(destination_path):
    print(f"Excel dump exists: {destination_path}".format())
else:
    print(f"Creating directory: {destination_path}")
    os.mkdir(destination_path)

for file in source_files:
    if file.endswith(".xlsx"):
        shutil.move(src=os.path.join(source_path, file),
                    dst=os.path.join(destination_path, file))
    elif file.endswith(".jpg"):
        if os.path.exists(source_path + "Pictures/"):
            "Directory for photos exists"
        else:
            print("Creating directory for photos {}".format(source_path + "Pictures"))
            os.mkdir(source_path + "Pictures")
            shutil.move(src=os.path.join(source_path, file),
                        dst=os.path.join(source_path + "Pictures", file))
    else:
        print("No more files to organise")
