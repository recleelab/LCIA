import os 
import matplotlib.pyplot as plt 
import numpy as np 
from aicsimageio import AICSImage

def ExtractFileNames(LocOfData,
                    Target='',
                    Extension=".vsi",
                    SubFolder=""):
    
    fullFilePath = os.path.join(LocOfData,SubFolder)
    
    fileContents = os.listdir(fullFilePath)
    
    fileContents = [i for i in fileContents if Extension in i]
    fileContents = [i for i in fileContents if Target in i]
    
    return fileContents 

def LoadBasedOnFileName(LocOfData,
                        Target='',
                        Extension=".vsi",
                        SubFolder="",
                        Identifier=0,
                        LoadAsDictionary = True):
    
    #Finding files that are in the known extension
    foundImages = ExtractFileNames(LocOfData,Target,Extension,SubFolder)
    print("The Following images have been found and are loaded:")
    
    if LoadAsDictionary:
        output = {}
    else: 
        output = []
    for i in foundImages: 
        print(i)
        locOfImage = os.path.join(LocOfData,SubFolder,i)
        tempLoad = AICSImage(locOfImage).dask_data[:,Identifier,:,:,:]
        if LoadAsDictionary:
            output[i] = tempLoad
        else: 
            output.append(tempLoad)
    
    return output

def GetAverageOfAverage(FlatFieldDictionary,ListToExclude=[]): 
    averageofaverage = []
    
    
    #Taking the avereage of each flat field set 
    for i in FlatFieldDictionary: 
        if i not in ListToExclude: 
            averageofaverage.append(FlatFieldDictionary[i].mean(0).compute())
            
    averageofaverage = np.concatenate(averageofaverage,0)
    
    
    averageofaverage = averageofaverage.mean(0)
    
    return averageofaverage


def PlotImageAndDiagonal(InputImage,Title=""):
    plt.figure(figsize=(15,5))

    plt.subplot(131)
    plt.imshow(InputImage)
    plt.title(Title)

    plt.subplot(132)
    plt.plot(np.diagonal(InputImage))
    plt.grid()
    plt.xlabel("Location Along the Diagonal",size=10)
    plt.ylabel("Pixel Intensity",size=20)
    plt.title(Title)
    
    plt.subplot(133)
    plt.hist(InputImage.flatten(),bins=np.arange(0,InputImage.max(),1),log=True)
    plt.title("Pixel Distribution")
    plt.xlabel("Bin",size=10)
    plt.grid()


def ObtainFlatFieldImages(LocOfData,
                          FlatFieldId = "FlatField",
                          DarkFieldID = "DarkField",
                          SubFolder="Folder_current",
                          Plot = True):

    darkFieldImages = LoadBasedOnFileName(LocOfData,SubFolder=SubFolder,Target = DarkFieldID)
    flatFieldImages = LoadBasedOnFileName(LocOfData,SubFolder=SubFolder,Target = FlatFieldId)

    flatField = GetAverageOfAverage(flatFieldImages)
    darkField = GetAverageOfAverage(darkFieldImages)
    
    if Plot: 
        PlotImageAndDiagonal(flatField,"Flat Field")
        PlotImageAndDiagonal(darkField,"Dark Field")
        
        
    return flatField,darkField 








