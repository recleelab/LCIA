import pandas as pd 
import matplotlib.pyplot as plt 
import os 

from LCIA import load_images as li
from LCIA import extract_data as ed 
from LCIA import auto_detection as ad
from LCIA import plotting as pl 

import numpy as np


from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans



def PCAOnTrajectories(NumComponents,FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift = False,NormalizeTraj=True): 
    #Getting the trajectories: 
    traj = ad.CombineReplicates(FunctionType,WellList,Channel,LocOfOutput,CorrectLinearShift,
                                ShowLinearShift=False)
    
    #Normalize Trajectory 
    if NormalizeTraj:
        #The input of normalize trajectory must be Time X Cells 
        traj = pl.NormalizeTrajectory(traj.T)

        #The PCA must be Cells X Time since time is assumed to be the feature for each timepoint 
        traj = traj.T
    
    #PCA cannot handle NAN inputs. Therefore they will be treated as 0's. 
    traj = np.nan_to_num(traj)

    pca = PCA(n_components=NumComponents)
    Out = pca.fit_transform(traj)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print("With {} components, the explained variance is = {}".format(NumComponents,explained_variance))

    return Out

def KMeanTrajectories(NumClusters,NumComponents,FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift = False,NormalizeTraj=True,StartValue=0):
    '''
    Clustering trajectories to find common patterns. PCA analysis is run first. 
    '''

    #Reducing the trajectories to the number of components 
    pca =  PCAOnTrajectories(NumComponents,FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift,NormalizeTraj)

    #Running Kmeans Algorthm 
    clusters = KMeans(NumClusters).fit(pca[:,StartValue:])

    return clusters
    