import pandas as pd 
import os 
import napari 
import numpy as np 
from LCIA import load_images as li
from LCIA import process_images as pi
from LCIA import auto_detection as ad 

def CorrectAutomaticSegmentation(WellLabel,KeyInformation,
                       LocOfData,LocOfOutput,CorrectShape,LocOfCorrectedTraj,Background=None):

    #Loading All Images through AICS Package  
    print("Loading Images")
    img,channels = li.LoadByAICSImageIOALL(WellLabel,KeyInformation,LocOfData,CorrectShape)
    print("Finsihed Loading Images")

    print("Started Enhancing IMage")
    for ith in range(img.shape[1]): 
        img[:,ith,:,:] = pi.EnhanceContrast(img[:,ith,:,:])

    print("Shape of Image: ",img.shape)

    channels =channels.split("//")
    print(channels)

    info = ad.LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    masks = ad.DecompressImageStack(info['NucleusCompressedMask'])

    napariPaths = info["NapariPaths"]
    paths = info["Paths"]
    print("Loading Color Mask")
    colormasks = ad.CreateColorConnectedMaskV1(masks,paths)
    print("SUccess")

    locOfStorage = os.path.join(LocOfCorrectedTraj,"{}_connectedPaths.npy".format(WellLabel))
    try: 
        previousData = np.load(locOfStorage,allow_pickle=True).item()
        ambiguousList = previousData["ambiguousList"]
        failedList = previousData["failedList"]
        storedDivisions = previousData["storedDivisions"]
        storedConnectedPaths = previousData["storedConnectedPaths"]
        connectedNapariPaths  = previousData["connectedNapariPaths"]

    except:
        print("Failed to load Data")
        ambiguousList = []
        failedList = []
        storedDivisions = []
        numFrames,xrange,yrange = masks.shape
        storedConnectedPaths = np.zeros((1,numFrames),np.int32)

        connectedNapariPaths = np.zeros((numFrames,4))
        connectedNapariPaths[:,1] = np.arange(numFrames)
        connectedNapariPaths[:,3] = np.linspace(0,xrange,numFrames)


    viewer = napari.Viewer()
    for i,name in enumerate(channels): 
        viewer.add_image(img[:,i,:,:],name=name+"//{}".format(i+1))

    viewer.add_image(colormasks,name="masks",opacity=0.1)
    viewer.add_tracks(napariPaths,name='Automatic Paths')

    @viewer.bind_key("1")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,0)

    @viewer.bind_key("2")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,1)

    @viewer.bind_key("3")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,2)

    @viewer.bind_key("4")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,3)
    @viewer.bind_key("a")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,4)
    @viewer.bind_key("s")
    def turnOnLayer1(viewer): 
        turnOnLayer(viewer,5)



    connectedIDs = []
    otherStoredIDs = []
    viewer.title = WellLabel
    doubleClickStore = ["Connect Trajectories"]
    @viewer.mouse_double_click_callbacks.append
    def update_layer(layer, event):
        z,x,y = event.position 
        z = int(z)
        x = int(x) 
        y = int(y)
        masksID = masks[z,x,y]
        if masksID == 0: 
            print("Zero Mask ID, not valid")
        elif doubleClickStore[0]=="Connect Trajectories":
            pathID = np.where(paths[:,z]==masksID)[0][0]
            if len(connectedIDs)%2 == 1: 
                q,prevID,q1 = connectedIDs[-1]
                if prevID == pathID: 
                    connectedIDs.append((z,pathID,masksID))
                    print("Finished Track")
                else: 
                    print("Must click ID with {}".format(prevID))
            else: 
                print("HELLO")
                connectedIDs.append((z,pathID,masksID))
            print(connectedIDs)
        elif doubleClickStore[0] == "Other Store": 
            pathID = np.where(paths[:,z]==masksID)[0][0]
            otherStoredIDs.append((z,pathID,masksID))
            print("Other Stored IDs")
            print(otherStoredIDs)
            
    @viewer.bind_key('t')
    def toggle(viewer): 
        if doubleClickStore[0]=="Connect Trajectories": 
            doubleClickStore[0]= "Other Store"
            print("Toggled to Other Store")
        else: 
            doubleClickStore[0]="Connect Trajectories"
            print("Toggled to connect store")

    @viewer.bind_key('r')
    def delete_layer(viewer):
        if doubleClickStore[0]=="Connect Trajectories":
            connectedIDs.pop(-1)
            print(connectedIDs)
        elif doubleClickStore[0]=="Other Store": 
            otherStoredIDs.pop(-1)
            print(otherStoredIDs)
                
    @viewer.bind_key('d')
    def RecordDivision(Viewer): 
        UpdateList(Viewer,storedDivisions,"Stored Division",otherStoredIDs)
            
    @viewer.bind_key('g')
    def RecordDivision(Viewer): 
        UpdateList(Viewer,ambiguousList,"Ambiguous Segmentation",otherStoredIDs)
        
    @viewer.bind_key('f')
    def RecordDivision(Viewer): 
        UpdateList(Viewer,failedList,"Failed Segmentation",otherStoredIDs)

    viewer.add_tracks(connectedNapariPaths,name='ConnectedPaths')
    viewer.layers[0].metadata['NapariPaths'] = connectedNapariPaths
    viewer.layers[0].metadata['ConnectedPaths'] = storedConnectedPaths
    @viewer.bind_key('p')
    def StoreConnection(viewer): 

        new_path = ConstructPath(connectedIDs,paths)
        napariPath = FindNewTrack(new_path,masks,viewer.layers[0].metadata['ConnectedPaths'].shape[0] )

        

        viewer.layers[0].metadata['NapariPaths'] = np.concatenate((viewer.layers[0].metadata['NapariPaths'],napariPath))
        viewer.layers[0].metadata['ConnectedPaths'] = np.concatenate((viewer.layers[0].metadata['ConnectedPaths'] ,new_path[np.newaxis,:]))

        viewer.layers[-1].data = viewer.layers[0].metadata['NapariPaths']
        for ith in range(len(connectedIDs)): 
            connectedIDs.pop()

        
        saveData = {"ambiguousList":   ambiguousList,  
                    "failedList": failedList,
                    "storedDivisions": storedDivisions,
                    "storedConnectedPaths": viewer.layers[0].metadata['ConnectedPaths'] ,
                    "connectedNapariPaths":viewer.layers[0].metadata['NapariPaths'] 

        }

        np.save(locOfStorage,saveData)

    @viewer.bind_key('~')
    def RemoveLast(viewer): 
        
        
        npaths,nFrames = viewer.layers[0].metadata['ConnectedPaths'].shape
        viewer.layers[0].metadata['ConnectedPaths'] = viewer.layers[0].metadata['ConnectedPaths'][0:npaths-1,:]
        tokeep = viewer.layers[0].metadata['NapariPaths'][:,0]<(npaths-1)
        viewer.layers[0].metadata['NapariPaths'] = viewer.layers[0].metadata['NapariPaths'][tokeep]
        viewer.layers[-1].data = viewer.layers[0].metadata['NapariPaths']
        saveData = {"ambiguousList":   ambiguousList,  
                        "failedList": failedList,
                        "storedDivisions": storedDivisions,
                        "storedConnectedPaths": viewer.layers[0].metadata['ConnectedPaths'] ,
                        "connectedNapariPaths":viewer.layers[0].metadata['NapariPaths'] 

            }

        np.save(locOfStorage,saveData)

        print("Removed The following path: {}".format(npaths-1))

    napari.run()
    return viewer


def UpdateList(Viewer,List2Update,Name,otherStoredIDs): 
    if len(otherStoredIDs)==0: 
        for ith in List2Update[-1]:
            otherStoredIDs.append(ith)
        List2Update.pop()
            
    else: 
        List2Update.append(otherStoredIDs.copy())
        for ith in range(len(otherStoredIDs)): 
            otherStoredIDs.pop()
        
    print("Number of {} found: {}".format(Name,len(List2Update)))
    print("Other List",otherStoredIDs)
    print(Name,List2Update)

def turnOnLayer(Viewer,Layer): 
    Viewer.layers[Layer].visible = not Viewer.layers[Layer].visible
        

def ConstructPath(NewPath,AutomaticPaths): 
    numFrames = AutomaticPaths.shape[1]
    newPath = np.zeros(numFrames,dtype=np.int32)
    
    numConnectedPaths = len(NewPath)//2
    leftOvers = len(NewPath)%2
    
    
    for i in range(numConnectedPaths): 
        framei,pathi,maski = NewPath[i*2]
        frame,path,mask = NewPath[i*2+1]
        newPath[framei:frame+1] = AutomaticPaths[pathi,framei:frame+1]
    
    if leftOvers: 
        print("HI")
        frame,path,mask = NewPath[-1]
        print(AutomaticPaths[path,frame:])
        newPath[frame:] = AutomaticPaths[path,frame:] 
    return newPath

def FindNewTrack(NewTrack,Masks,PathID): 
    
    storedPaths = [] 
    for i,ith in enumerate(NewTrack): 
        if ith == 0: 
            pass 
        else: 
            x,y = np.where(ith==Masks[i])
            storedPaths.append((PathID,i,np.mean(x),np.mean(y)))
    return np.stack(storedPaths)