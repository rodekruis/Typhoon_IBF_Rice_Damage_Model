import pandas as pd
import openpyxl
from os import listdir
from os.path import isfile, join


mypath = r"IBF_typhoon_model//data"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)


# df = pd.read_excel(path, engine="openpyxl")
# print(df.head())
