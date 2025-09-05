import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 

def InterpolateData(Data2Interpolate):
    '''
    Interpolates mising data points due to mis-segmentation within a track. 
    Only interpolates between the first and last reported value. 
    Input: 
        Data2Interpolate -> [nCells X TimePoints] numpy matrix
    
    Output: 
        interpolatedData -> [nCells X TimePoints] numpy matrix
    

    '''
    interpolatedData = np.ones_like(Data2Interpolate)*-100
    for i,ith in enumerate(Data2Interpolate):
        realValues = ith>-1E10
        xLoc, = np.where(realValues)
        if np.any(xLoc):
            minreal = min(xLoc)
            maxreal = max(xLoc)
            reduced = ith[minreal:maxreal+1]
            reduced = pd.Series(reduced).interpolate().to_numpy()
            interpolatedData[i,minreal:maxreal+1] = reduced
    interpolatedData[interpolatedData==-100] = np.nan
    
    return interpolatedData

def ConnectCustomTrajectories(UnConnectedData,Position,ManualTraj): 
    ''''
    This function creates the trajectory from the ManualTraj input 
    '''
    data2Connect = UnConnectedData[Position]
        
    acceptedPaths = ManualTraj
    allPaths = []
    for ith in acceptedPaths:  
        singleTraj = np.zeros(ith.shape[0])
        for i,j in enumerate(ith): 
            singleTraj[i] = data2Connect[j,i] 
        allPaths.append(singleTraj)
    
    return np.stack(allPaths)

def GetData(ithPosition,Path,Data):
    ''''
    This function loads the custom trajectories and then calls 
    ConnectCustomTrajectories to connect the data 
    '''
    #Input can either be a number or full position name if the label 
    #starts with Position 
    if type(ithPosition) == int:
        label = "Position {}".format(ithPosition)
    else: 
        label = ithPosition

    #Loading the connected path labels 
    connectedPath = label+"_connectedPaths.npy"
    fullPath = os.path.join(Path,connectedPath)
    connectedDict = np.load(fullPath,allow_pickle=True).item()
    connectedPaths = connectedDict["storedConnectedPaths"]

    #Connecting data based off of custom trajectories and data
    data = ConnectCustomTrajectories(Data,label,connectedPaths)
    return data

def GetAllCustomTrajData(ListOfPositions,Path,UnConnectedData,StopTime=-1):
    ''''
    Given a list of positions, the function will stitch together the manually corrected 
    trajectories. 
        - ListOfPositions - Python list of the position/well label 
        - Path to output of the connected data 
        - Dictionary of unconnected data 
        - StopTime: Length of trajectory to return. Default is full trajectory 
    '''
    
    connnectedData = {}
    positionTrackLoc = []
    for ith in ListOfPositions: 
        #Connect data based off of manual corrections 
        manuallyConnectedData = GetData(ith,Path,UnConnectedData)

        #Interpolating missing data 
        connnectedData[ith] = InterpolateData(manuallyConnectedData)[:,0:StopTime]

        ithKey = np.repeat(ith,  manuallyConnectedData.shape[0])
        ithKey = list((zip(ithKey,np.arange(manuallyConnectedData.shape[0]))))
        positionTrackLoc+=ithKey
        
    return connnectedData,positionTrackLoc


def CalculateandPlotFalsePostiveRate(InputData,XLabel = "HI"): 

    '''
    Given a set of input data, specifically negative data set, it will set threshold 
    to create to find postive cells 
    '''

    #Calculating the false postive rate 
    InputDataFlatten = InputData.flatten()
    threshold = np.linspace(0,np.nanmax(InputDataFlatten),10000)

    #Remove NAN from values 
    realValues = np.isnan(InputDataFlatten)==False
    InputDataFlatten = InputDataFlatten[realValues]

    nSamples = len(InputDataFlatten)


    falsePostiveRate = []
    for ith in threshold: 
        falsePostiveRate.append(np.sum(InputDataFlatten>ith)/nSamples)

    falsePostiveRate = np.array(falsePostiveRate)

    plt.figure(figsize=(15,15))
    plt.plot(threshold,falsePostiveRate,linewidth=8)
    plt.grid()
    plt.xticks(fontsize=30,rotation=45)
    plt.yticks(fontsize=30)
    plt.xlabel(XLabel, size=30)
    plt.ylabel("False Positive Rate",size=30)


    targetFalsePostiveRates = [0.2,0.1,0.05, 0.01]
    targetThresholds = []
    for ith in targetFalsePostiveRates:
        indexOfThreshold = np.argmin(abs(ith-falsePostiveRate))
        targetThresholds.append(threshold[indexOfThreshold])

    colors = ['r','g','k','c']
    for i,ith in enumerate(targetThresholds): 
        plt.axvline(ith,c=colors[i],label = "{}% False Postive Rate/Threshold {}".format(round(targetFalsePostiveRates[i]*100),round(ith)),
                   linewidth=8)
    plt.legend(fontsize=20,loc='best')


def CombinePositions(PositionList,DataDictionary): 
    '''
    Given a position list and dictionary of data, returns comebined values 
    '''
    output = []

    for ith in PositionList: 
        output.append(DataDictionary[ith])
    output = np.concatenate(output)
    return output


def MovingWindowCumulativeSum(Input,Window=5):
    '''
    This function takes the cumulative sum of a moving window. Input needs to be 
    a 1-dimensional array. I got the code from a message board and seems to be working 
    as expected. 

    Assumes the missing data is at the begining and end of the trajectory. 
    
    '''
    #Counting the number of times a moving window is above the threshold 
    a = np.array(Input)
    
    missingData = np.isnan(a)
    
    b = a[missingData==False].cumsum()
    b[Window:] = b[Window:] - b[:-Window]
    return b 


def FindPostiveCells(AllTraj,Threshold,Window=5):
    '''
    From a matrix of trajectories, find cells that are above the threshold for a given window
    '''
    timeSpentAboveThreshold = []
    for ith in AllTraj: 
        aboveThreshold = ith>Threshold
        timeSpentAboveMovingWindow = MovingWindowCumulativeSum(aboveThreshold,Window)
        timeSpentAboveThreshold.append(np.any(timeSpentAboveMovingWindow==Window))
    return np.array(timeSpentAboveThreshold)

def OrderByLength(InputData): 
    '''
    Ordering trajectories by length where the input is a 
    cellxtime numpy array 
    '''
    lengthOfTraj = np.sum(InputData>0,1)
    sortedTraj = np.argsort(lengthOfTraj)
    
    
    return InputData[sortedTraj,:]

def SlideData2TimeZero(Data2Correct):
    '''
    Setting all trajectories to start at time 0 
    '''
    slidedData = np.ones_like(Data2Correct)*-1

    for i,ith in enumerate(Data2Correct): 
        findNAN = np.isnan(ith)
        findReal = findNAN == False
        xLoc, = np.where(findReal)
        if np.any(xLoc):
            minX = min(xLoc)
            maxX = max(xLoc)
            onlyMeasuredPart = ith[minX:maxX+1]
            slidedData[i,0:(maxX-minX+1)] = onlyMeasuredPart
            slidedData[i,(maxX-minX+1):] = -1
    slidedData[slidedData==-1] = np.nan
        
    return slidedData