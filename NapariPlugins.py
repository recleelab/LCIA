from LCIA import load_images as li
from LCIA import extract_data as ed
from LCIA import plotting as pl
from LCIA import process_images as pi
from LCIA import auto_detection as ad
import napari
import numpy as np 
import scipy
def ShowTracksInNapari(WellLabel,Channel,KeyInformation,
                       LocOfData,LocOfOutput,CorrectShape=0,
                       ShowMasks=False,showRing = 0,Threshold=20): 
    
    
    #============================================================
    #Loading Images
    #============================================================
    print("LOadingImage")
    img,channels = li.LoadByAICSImageIOALL(WellLabel,KeyInformation,LocOfData,CorrectShape)
    print("Started Enhancing IMage")
    img = img.astype(np.float32)
    #img = pi.EnhanceContrast(img)
    if False:
        for jth in range(img.shape[0]):
            for ith in range(img.shape[1]):
                background,_ = scipy.stats.mode(img[jth,ith,:,:].flatten())
                img[jth,ith,:,:]=img[jth,ith,:,:] - background 
        img[img<0] = 0

    for ith in range(img.shape[1]): 
        img[:,ith,:,:] = pi.EnhanceContrast(img[:,ith,:,:])
    
    #===========================================================
    #Loading Info
    #===========================================================
    print("Loading Masks")
    info = ad.LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    
    
    #===========================================================
    #ReScaling Stack 
    #===========================================================

    masks = ad.DecompressImageStack(info['NucleusCompressedMask'])
    z,x,y = masks.shape
    if showRing>0:
        masks = ad.GetStackCytoplasm(masks,showRing)
    
    #Scaling Image to match the mask
    #img = ad.RescaleStack(img,x,y)

    #Masks
    paths = info['Paths']

    if ShowMasks:
        colorMasks = ad.CreateColorConnectedMaskV1(masks,paths)
        
        
        
    viewer = napari.Viewer(title=WellLabel)

    
    
    #Showing Image
    i = 0 
    legend = {"DAPI": "blue",
             "TRITC": "yellow",
             "CY5":"red",
             "POL":"gray"
    }
    for ith in channels.split("//"):
        viewer.add_image(img[:,i,:,:],name=ith,colormap=legend[ith],
                         opacity=0.5)
        i+=1
    
    if ShowMasks: 
        viewer.add_image(colorMasks,name='Masks',opacity=0.5,rgb=True)
        
    #showing tracks
    paths = info['NapariPaths']
    #Correcting paths to be 512X512: 
    if x<612:
        paths[:,2:] =  paths[:,2:]/2

    viewer.add_tracks(info['NapariPaths'],properties=info['Properties'],graph=info['Graph'])
    
    
    return viewer