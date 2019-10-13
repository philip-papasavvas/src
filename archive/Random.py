import os
import shutil

# move all of the log files into the log folder, and
# delete when finsihed

lst = lst = os.listdir(R"C:\Users\Philip\Desktop\upload")
log_lst = [x for x in lst if "Log Total" in x]

for x in log_lst:
    shutil.copy((R"C:\Users\Philip\Desktop\upload/" + x),(R"C:\Users\Philip\Desktop\upload\log charts/" + x))

for x in log_lst:
    os.remove((R"C:\Users\Philip\Desktop\upload/" + x))