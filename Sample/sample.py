import sys
import os
#Adding the LCIA functions to the python path.
sys.path.append("../../")
from LCIA import load_images as li
from LCIA import extract_data as ed
from LCIA import plotting as pl
from LCIA import process_images as pi
from LCIA import auto_detection as ad
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import argparse
#===========================================
#Pointing to Loctions of Importance
#===========================================

LocOfData = './Data/'

LocOfOutput = './Analysis/'

NameOfKey = "./key.csv"

ExpCode = 1

Test1 = ["A"]

Filter= "YFP"

#===========================================
#Analysis
#===========================================

#Getting the key information and filtering by expiermental code
key = pd.read_csv(NameOfKey)
key = key[key["Expierment Code"]== ExpCode]

parser = argparse.ArgumentParser('Analyze Images')
parser.add_argument('--option', default=1, type=int, help='decide what to do')
args = parser.parse_args()

if args.option==1: 
    print("Loading Data")
    ed.AnnoteImagesAndSave(key, LocOfOutput, LocOfData,Filter,Test1)
elif args.option == 2: 
    for i in Test1: 
        ad.SaveAndSegmentByCellPose(i,Filter,key,LocOfData,LocOfOutput)
    print("Connecting cells between frames")
    ad.ConnectAndSaveTrajectories(Test1,LocOfOutput)

    print("Extracting Data")
    ad.ExtractAllData(Test1,["mCherry","CFP","YFP"],key,LocOfData,LocOfOutput,-1)
    ad.ExtractRatioData(Test1,"CFP","YFP",key,LocOfData,LocOfOutput)
    ad.SaveMeanTrajectoriesToExcel(LocOfOutput)
elif args.option ==3: 
    ad.ShowAutoDetection(Test1,Filter,key,LocOfData,LocOfOutput)




    





 
