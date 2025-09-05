import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LCIA import load_images as li
from LCIA import process_images as pi
from LCIA import extract_data as ed 
from LCIA import  plotting as pl
import cv2
import time 
from os import path
import os
from collections import Counter
import matplotlib
import plotly.graph_objects as go
import multiprocess
import scipy
#Cell Pose is the main algorthm used to segment the data stacks 
from cellpose import models,io 
from collections import Counter
#import javabridge
#===========================================================
#Visualization
#===========================================================
def ShowAutoDetection(WellList,Channel,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc = True,BackgroundPercentile=0,Point=0,AutomaticBackgroundCorrection = False):
    if not isinstance(WellList,list): 
        WellList = [WellList]
    for ith_well in WellList: 
        _ShowAutoDetection(ith_well,Channel,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc,BackgroundPercentile,Point,AutomaticBackgroundCorrection)

    javabridge.kill_vm()

def _ShowAutoDetection(WellLabel,Channel,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc = True,BackgroundPercentile=0,Point=0,AutomaticBackgroundCorrection = False,ZStack = False): 
    #Loading the Image Stack
    image_stack = li.GetOnlyOneChannel(WellLabel,Channel,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,KillJavaBridge=True,BioFormats=False,CorrectShape=0 )

    #Loading the masks: 
    if SegmentBothCyAndNuc: 
        masks,cytoplasm_masks = GetNucAndCytMasks(WellLabel,LocOfOutput)
    else: 
        masks,cytoplasm_masks = GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,40)

    #Loading paths: 
    paths = GetPaths(WellLabel,LocOfOutput)

    #Remove Nucleus that were not tracked to have a complete trajectory
    masks = RemoveNucleusMasks(masks,paths)
    cytoplasm_masks = RemoveNucleusMasks(cytoplasm_masks,paths)

    #avgData = LoadMeanTrajectories(LocOfOutput)[WellLabel]
    WindowName = WellLabel+"///"+Channel 

    updatedPaths = DisplayMaskOverData(image_stack,paths,masks,masks,WindowName,BackgroundPercentile,None)

    #Saving the connection in the same location as the masks
    store_location = LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    store_location["Paths"] = updatedPaths
    np.save(os.path.join(LocOfOutput,WellLabel+".npy"),store_location)


def DisplayMaskOverData(ImageStack,Paths,Masks,OptionalMask,WindowName,BackgroundPercentile,AverageData):
    cell = 0 
    max_cell = Paths.shape[0]
    print(max_cell)
    
    #Rescaling Stack for speed 
    ImageStack = RescaleStack(ImageStack,1024,1024)

    #Dimensions of the input 
    n_frames,width,height = ImageStack.shape 

    if OptionalMask is None: 
        masks_rescaled = np.zeros(ImageStack.shape,dtype=np.uint8)
    else: 
        masks_rescaled = RescaleStack(OptionalMask,width,height)
    Masks = RescaleStack(Masks,width,height)

    #Generate Random Color Palette for display 
    def GenerateRandomColors(MaxCell):
        color_palette = []
        for ith_color in range(max_cell): 
            color_palette.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
        return color_palette
    color_palette = GenerateRandomColors(max_cell) 

    frac_image = 1.0
    frac_mask  = 0.5 
    def ShowImage(ImageStack,MaskStack,ImageToShow,WindowName): 
        nonlocal Paths,cell,Masks
        nonlocal frac_image, frac_mask,max_cell
        image = pi.EnhanceContrast(ImageStack[ImageToShow,:,:])
        mask  = MaskStack[ImageToShow,:,:].copy()
        cytostack = Masks[ImageToShow,:,:].copy()
        image3d = np.dstack((image,image,image))
        print("Current Frame Number: {}".format(ImageToShow))
        #To Highlight a specific trajectory. Must first find the specific 
        #coordinanents 
        if max_cell!=0:
            if Paths[cell,ImageToShow] !=0:
                x_spe,y_spec = np.where(mask==Paths[cell,ImageToShow])
                x_spe_c,y_spe_c = np.where(cytostack==Paths[cell,ImageToShow])
        x_loc,y_loc = np.where(mask>0)
        x_neg,y_neg = np.where(mask<0)
        mask = np.uint8(np.dstack((mask,mask,mask)))
        mask[x_loc,y_loc,:] = (255,0,0)
        mask[x_neg,y_neg,:] = (0,0,255)
        #mask[x_loc_c,y_loc_c] = (0,255,0)
        #x_loc_c,y_loc_c = np.where(cytostack>0)
        for ith_cell in range(max_cell): 
            cell_mask_identifier = Paths[ith_cell,ImageToShow]
            if cell_mask_identifier!=0:
                mask[cytostack==cell_mask_identifier,:] = color_palette[ith_cell]

        if max_cell!=0:
            if Paths[cell,ImageToShow] !=0:
                #Show Specific Trajectory: 
                mask[x_spe,y_spec,:] = (255,255,0)
                mask[x_spe_c,y_spe_c,:] = (0,165,255)

        background_cutoff  = np.percentile(ImageStack[ImageToShow,:,:],BackgroundPercentile)
        
        x,y = np.where(ImageStack[ImageToShow,:,:] <= background_cutoff)
        mask[x,y] = (0,255,255)

        display_image = cv2.addWeighted(image3d,frac_image,mask,frac_mask,1.0)

        #When I want to save a movie, I have to set this to True. Need to add 
        #functionality so I do not have to change the LCIA code in order to 
        #activate it. 
        if False: 
            org = (0,30)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            # fontScale
            fontScale = 1
            #display_image[0:40,0:200,:] = 0
            cv2.putText(display_image,'Time: {} Minutes'.format(ImageToShow*5),org,font,fontScale,(255,255,255),3)
            matplotlib.image.imsave("TestingTiff/tt_0{}.tiff".format(ImageToShow),display_image)
        cv2.imshow(WindowName,display_image)

    fig = go.FigureWidget()
    first = False
    CorrectPaths = False
    PlotOnClick = False
    path1 = -1
    path2 = -1
    mergeIndex = -1
    def MouseCommands(event,x,y,flags,param):
        nonlocal first,fig,CorrectPaths,path1,path2,mergeIndex,Paths,max_cell,color_palette
        if event == cv2.EVENT_MOUSEMOVE:
            print("X: {}, Y: {}, Intensity: {}".format(x,y,ImageStack[current_image,y,x])) 
        elif event == cv2.EVENT_LBUTTONUP: 
            maskLocation = masks_rescaled[current_image,y,x]
            
            identifiedPath, = np.where(Paths[:,current_image]==maskLocation)
            print("Identified Path: {}".format(identifiedPath))

            if CorrectPaths:
                if mergeIndex == -1: 
                    print("//////////////////////////////////////////////////")
                    print("              Merge Index Set                     ")
                    print("//////////////////////////////////////////////////")
                    mergeIndex = current_image
                    path1 = identifiedPath[0]

                    print("//////////////////////////////////////////////////")
                    print("              Path1: {}                           ".format(path1))
                    print("//////////////////////////////////////////////////")

                else: 
                    if (current_image - mergeIndex) >=1: 
                        path2 = identifiedPath[0]
                        print("Updating Path {} and {}".format(path1,path2))
                        Paths = MergePaths(Paths,path1,path2,current_image)
                    else: 
                        print("Reselect Paths")
                    path1 = -1
                    path2 = -1
                    mergeIndex = -1
                    max_cell = Paths.shape[0]
                    color_palette = GenerateRandomColors(max_cell) 

            else: 
                if maskLocation==0: 
                    identifiedPath = -1
                else: 
                    if PlotOnClick:
                        identifiedPath, = np.where(Paths[:,current_image]==maskLocation)
                        identifiedPath = identifiedPath[0]
                        #first  = pl.PlotTrajectory(fig,AverageData,identifiedPath,current_image,first)


    current_image = 0 
    ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
    cv2.setMouseCallback(WindowName,MouseCommands)

    while True:
        store_waitkey = cv2.waitKey(1000)
        if store_waitkey & 0xFF == ord('q'):
            break
        elif (store_waitkey & 0xFF == ord('f')): #Right Arrow
            current_image += 1 
            if current_image >= n_frames: 
                current_image = 0 
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('d')): #Left Arrow
            current_image -= 1
            if current_image<0:
                current_image = n_frames-1
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('s')): 
            cell += 1
            if cell>=max_cell:
                cell = 0 
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
            print(cell)
        elif (store_waitkey & 0xFF == ord('a')): 
            cell -= 1
            if cell<0:
                cell = max_cell-1
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
            print(cell)
        elif (store_waitkey & 0xFF == ord('r')):
            color_palette = GenerateRandomColors(max_cell) 
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('j')):
            frac_image -= 0.1
            if frac_image <0: 
                frac_image = 0
            print("Fraction of Image to Display: {}".format(frac_image))
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('k')):
            frac_image += 0.1
            if frac_image >1: 
                frac_image = 1
            print("Fraction of Image to Display: {}".format(frac_image))
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('l')):
            frac_mask -= 0.1
            if frac_mask <0: 
                frac_mask = 0
            print("Fraction of Mask to Display: {}".format(frac_mask))
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord(';')):
            frac_mask += 0.1
            if frac_mask >1: 
                frac_mask = 1
            print("Fraction of Mask to Display: {}".format(frac_mask))
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('i')):
            frac_image = 1.0
            frac_mask  = 0.5 
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('u')):
            frac_image = 0.1
            frac_mask  = 1.0 
            ShowImage(ImageStack,masks_rescaled,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('c')): 
            
            CorrectPaths = not CorrectPaths
            if CorrectPaths:
                print("Entering Path Correction Mode")
            else: 
                print("Exiting Path Correction Mode")
        elif (store_waitkey & 0xFF == ord('p')): 
            PlotOnClick = not PlotOnClick
            outString = "Activated Plot on Click" if PlotOnClick else "Deactivated Plot On Click"
            print(outString)

    cv2.destroyAllWindows()
    return Paths

def GetStackCytoplasm(MaskStack,KernelSize=5,Buffer=0): 
    n_slides, width,height = MaskStack.shape 

    output_cytoplasm = np.zeros(MaskStack.shape,dtype=np.uint16)

    for ith_slide in range(n_slides): 
        if (Buffer>0) & (Buffer<KernelSize): 
            bufferRegion = DilateToCytoplasm(MaskStack[ith_slide,:,:],Buffer)
        elif (Buffer>KernelSize): 
            print("Buffer not applied because KernelSize<BufferRegion")
            bufferRegion = 0 
        else: 
            bufferRegion = 0 
        output_cytoplasm[ith_slide,:,:] = DilateToCytoplasm(MaskStack[ith_slide,:,:],KernelSize) - bufferRegion

    return output_cytoplasm

def DilateToCytoplasm(Mask,KernelSize=5): 
    #Mask = np.uint16(Mask)
    #Finding locations of non-zero spots
    x,y = np.where(Mask>0)

    kernel = np.ones((KernelSize,KernelSize),dtype=np.uint8)

    dilated_mask = Mask.copy()

    dilated_mask[dilated_mask<0] = 0

    dilated_mask = np.uint16(dilated_mask)
    dilated_mask = cv2.dilate(dilated_mask,kernel)
    dilated_mask[x,y] = 0 

    return dilated_mask

def SeparateCombinedMask(CombinedCellMask): 
    '''
    Extract the cytoplasm and nucleus masks when segmentation was performed on both the nucleus and whole cell and the mask was saved as a special combined mask where nucleus are represented by an odd number, and the next even number represents the cytoplasm masks. The return will be separate masks for the cytoplasm and nucleus with the same numbering scheme. 
    '''

    cell_masks = CombinedCellMask.copy()

    #Cells are even numbers. Therefore divide by 2 to get the cells with values of 1,2,3...
    cell_masks[cell_masks%2==1] = 0 
    cell_masks = cell_masks/2 

    #Nucleus are the odd numbers, therefore convert the input from odd to increasing values 
    nucleus_mask = CombinedCellMask.copy()
    nucleus_mask[nucleus_mask%2==0] = 0 
    nucleus_mask = nucleus_mask/2 + 0.5
    nucleus_mask[nucleus_mask<1] = 0 

    #Return nucleus mask and cytoplasm mask where the numbers correspond to the same nucleus and cytoplasm 
    return nucleus_mask,cell_masks

def GetNucAndCytMasks(WellLabel,LocOfOutput): 
    info = LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    
    #Determine if compressed or combined mask are stored: 
    keys = info.keys()
    if ("NucleusCompressedMask" in keys) and ("CytoplasmCompressedMask" in keys): 
        nucleus_mask = DecompressImageStack(info["NucleusCompressedMask"])
        cytoplasm_masks = DecompressImageStack(info["CytoplasmCompressedMask"])

        #Remove Nucleus from recovered cell: 
        z,x,y = np.where(nucleus_mask>0)
        cytoplasm_masks[z,x,y]= 0 
    else: 
        #Cell masks are combined nucleus and cytoplasm masks 
        combined = info["CellMasks"]
        nucleus_mask,cytoplasm_masks = SeparateCombinedMask(combined)

    return nucleus_mask,cytoplasm_masks

def GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,KernelSize=5,Buffer=0): 
    info = LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    
    #Determine if compressed or combined mask are stored: 
    keys = info.keys()
    if ("NucleusCompressedMask" in keys): 
        nucleus_mask = DecompressImageStack(info["NucleusCompressedMask"])
    else: 
        nucleus_mask = info["NucleusMasks"].copy()

    cytoplasm_mask = GetStackCytoplasm(nucleus_mask,KernelSize=KernelSize,Buffer=Buffer)

    return nucleus_mask,cytoplasm_mask

def GetPaths(WellLabel,LocOfOutput): 
    info = LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)

    try: 
        paths = info["Paths"].copy()
    except: 
        paths = np.array([])

    return paths

def RemoveNucleusMasks(MaskStack,Paths):
    '''
    Removing masks that do not have a trajectory associated with it 
    ''' 
    #Number of frames: 
    n_frames = MaskStack.shape[0]

    #Removing Cell masks that are not part of a trajectory: 
    masks = MaskStack.copy()

    for ith_frame in range(n_frames): 
        found_paths = Paths[:,ith_frame]
        max_cell_in_frame  = int(masks[ith_frame,:,:].max())
        for ith_cell in range(1,max_cell_in_frame+1): 
            if ith_cell not in found_paths: 
                x_loc,y_loc = np.where(masks[ith_frame,:,:]==ith_cell)
                masks[ith_frame,x_loc,y_loc] = 0
    
    return masks

def GetColorPallet(Paths): 
    
    nCells,nFrames = Paths.shape 
    
    colorList = []
    
    for i in range(nCells): 
        c = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        colorList.append(c)
    return colorList 

def CreateColorConnectedMaskV1(Masks,Paths,ExistingColorPallet=None): 
    maxCells,maxFrames = Paths.shape
    
    z,x,y = Masks.shape 
    
    colorMasks = np.zeros((z,x,y,3))
    
    for ithcell in range(maxCells):
        chosenColor = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        chosenPath = Paths[ithcell,:]
        if np.sum(chosenPath>0)>0:
            for ithFrame in range(maxFrames): 
                if chosenPath[ithFrame]>0: 
                    x,y = np.where(Masks[ithFrame]==chosenPath[ithFrame])
                    colorMasks[ithFrame,x,y,:] = chosenColor
            
    
    
    return colorMasks.astype(np.uint8)

def UpdateFrame(Inputs): 
    currentFrame, currentFramePaths,generatedColors = Inputs
    x,y = currentFrame.shape
    colorFrame = np.zeros((x,y,3),np.uint8)        
    for ind,ithCell in enumerate(currentFramePaths):
        if ithCell>0: 
            xloc,yloc = np.where(currentFrame==ithCell)
            colorFrame[xloc,yloc,:] = generatedColors[ind]
    return colorFrame


def CreateColorConnectedMask(Masks,Paths): 
    maxCells,maxFrames = Paths.shape
    
    z,x,y = Masks.shape 
    
    generatedColors = [(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)) for i in range(maxCells)]
    
    combineInput = [ (Masks[i],Paths[:,i],generatedColors) for i in range(z)]
    
    with multiprocess.Pool(5) as p: 
        allColorImages = p.map(UpdateFrame,combineInput)
        p.close()
        p.join()
    return np.stack(allColorImages)

#===========================================================
#Functions for auto segmentation and storing the masks 
#===========================================================
def LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel):
    '''
    Function to laod the masks saved in the LocOfOutput. If the masks is not avaible then 
    an empty dictionary will be returned. 
    Input: 
        - WellLabel      : Corresponds to the well label as specified in the KeyInformation. Must be 
                           a string
                           
        - LocOfOutput    : Location of output to save the cellpose masks 
    '''
    try: 
        StoredMasks = np.load(path.join(LocOfOutput,WellLabel+".npy"),allow_pickle=True).item()
    except: 
        print("No Masks found")
        StoredMasks = {}
    
    return StoredMasks

def SaveAndSegmentByCellPose(WellLabel,Channel,KeyInformation,LocOfData,LocOfOutput,Model="nuclei",NucleusDiameter = None,CellDiameter=None,CellChannel=None,CellKernelSize = -1,InvertNucleus=True,SegmentCell=True,CompressImages=True,AutomaticBackgroundCorrection = False,CorrectShape=4,Point=0): 
    '''
    Function to both run the cell segmentation and store the output of a single stack. 
    The input images are assumed to be 1024X1024, and will be rescaled to 512X512 to 
    increase the speed of cell segmentation. This function will also save the masks outputted 
    by cellpose in a npy file in the LocOfOutput directory and save it in a file named 
    <WellLabel>.npy. 

    Input: 
        - WellLabel      : Corresponds to the well label as specified in the KeyInformation. Must be 
                           a string
                         
        - Channel        : Corresponds to the excitation channel that was used during the experiment. 
                           Needs to match one of the options in the corresponding row in the Channels 
                           column in the KeyInformation 
                         
        - KeyInformation : A pandas dataframe that contains the information for only the experiment wanting 
                           to be analyzed. Therefore, if loading a key file with multiple experiments, one
                           must filter the dataframe to only include one experiment. 
        
        - LocOfData      : Local path to the image stack. 

        - LocOfOutput    : Location of output to save the cellpose masks 

        - Point          : Optional argument if more than one point was given per image. This is not a common scenario 
                           and may not be functional in the current code.  
    '''
    #Segmentation:
    nuc_masks_n = SegmentNucleusStackByCellPose(WellLabel,Channel,KeyInformation,LocOfData,Model,InvertNucleus,NucleusDiameter,Point,AutomaticBackgroundCorrection,CorrectShape=CorrectShape)
    
    if SegmentCell: 
        if CellChannel is None: 
            cell_channel = Channel
        else:
            cell_channel = CellChannel 
        cell_masks_n = SegmentCytoplasmStackByCellPose(WellLabel,cell_channel,KeyInformation,LocOfData,nuc_masks_n,CellDiameter,CellKernelSize,Point,AutomaticBackgroundCorrection)


    #Set new dictionary key to auto segment 
    StoredMasks = {}
    #StoredMasks["AutoSegment"] = auto_segment
    if SegmentCell: 
        if CompressImages: 
            nucleus,cytoplasm    = SeparateCombinedMask(cell_masks_n)
            StoredMasks["NucleusCompressedMask"]    = CompressImageStack(nucleus)
            StoredMasks["CytoplasmCompressedMask"]  = CompressImageStack(cytoplasm)
        else: 
            StoredMasks["CellMasks"] = cell_masks_n
    else:  
        if CompressImages:
            StoredMasks["NucleusCompressedMask"]    = CompressImageStack(nuc_masks_n)
        else:   
            StoredMasks["NucleusMasks"] = nuc_masks_n

    np.save(path.join(LocOfOutput,WellLabel),StoredMasks)

def SegmentNucleusStackByCellPose(WellLabel,Channel,KeyInformation,LocOfData,Model="nuclei",invert=True,NucleusDiameter=None,Point=0,AutomaticBackgroundCorrection=False,ZStack = False,CorrectShape=4):
    '''
    Function to both run the nucleus segmentation and store the output of a single stack. 
    The input images are assumed to be 1024X1024, and will be rescaled to 512X512 to 
    increase the speed of cell segmentation. 

    Input: 
        - WellLabel      : Corresponds to the well label as specified in the KeyInformation. Must be 
                           a string
                         
        - Channel        : Corresponds to the excitation channel that was used during the experiment. 
                           Needs to match one of the options in the corresponding row in the Channels 
                           column in the KeyInformation 
                         
        - KeyInformation : A pandas dataframe that contains the information for only the experiment wanting 
                           to be analyzed. Therefore, if loading a key file with multiple experiments, one
                           must filter the dataframe to only include one experiment. 
        
        - LocOfData      : Local path to the image stack. 

        - Point          : Optional argument if more than one point was given per image. This is not a common scenario 
                           and may not be functional in the current code.  
    '''

    #Loading the Image Stack and specific channel as specified by the Channel input 
    image_stack = li.GetOnlyOneChannel(WellLabel,Channel,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,BioFormats=False,CorrectShape=CorrectShape )

    #Resizing each image to 512X512 using INTER_NEAREST interpolation 
    #image_stack = RescaleStack(image_stack,512,512)

    #CellPose expects the input to be a list. Therefore, changing the image stack to a list 
    image_stack = list(image_stack) 
    
    #Channels are assumed to be be grayscale and we need to define a channel for each image  
    channels =np.zeros((len(image_stack),2)) 

    #Model to Segment the nucleus 
    #from cellpose.contrib import openvino_utils
    print("Nuclei Model Used: {}".format(Model))
    nuclei_model = models.CellposeModel(gpu=True, model_type = Model)
    #nuclei_model = openvino_utils.to_openvino(nuclei_model)
    #Estimating the nuclei using the CellPose model:
    nuc_masks_n, _, _= nuclei_model.eval(image_stack,
                                         invert=invert, 
                                         diameter=NucleusDiameter, 
                                         channels=channels)
    #nuc_masks_n,_,_,_          = nuclei_model.eval(image_stack[1:],invert=True, diameter=diams_1, channels=channels)

    #nuc_masks_n = [nuc_masks_1] + nuc_masks_n

    #Model to segment whole cell:
    #cyto_model = models.Cellpose(gpu=False, model_type = 'cyto') 

    #cell_masks_n, _, _, diams_1 = cyto_model.eval(image_stack,invert=False, diameter=None, channels=channels)
    #cell_masks_n, _, _, _       = cyto_model.eval(image_stack[1:],invert=False, diameter=diams_1, channels=channels)

    #cell_masks_n = [cell_masks_1] + cell_masks_n    

    #Converting data into a multi-dimsional matrix 
    nuc_masks_n = np.stack(nuc_masks_n)
    #cell_masks_n = np.stack(cell_masks_n)

    #Combining Masks: 
    #combined_mask = CombineNucleiCytoplasmStack(cell_masks_n,nuc_masks_n)

    return nuc_masks_n

def SmoothStack(ImageStack,KernelSize): 
    image_out = np.zeros(ImageStack.shape)

    for ith in range(ImageStack.shape[0]): 
        image_out[ith,:,:] = cv2.blur(ImageStack[ith,:,:],(KernelSize,KernelSize))
    return image_out

def SegmentCytoplasmStackByCellPose(WellLabel,Channel,KeyInformation,LocOfData,NucleusMasks,Diameter=None,KernelSize = -1,Point=0,AutomaticBackgroundCorrection = False,ZStack = False):
    '''
    Function to both run the cytoplasm segmentation and store the output of a single stack. 
    The input images are assumed to be 1024X1024, and will be rescaled to 512X512 to 
    increase the speed of cell segmentation. 

    Input: 
        - WellLabel      : Corresponds to the well label as specified in the KeyInformation. Must be 
                           a string
                         
        - Channel        : Corresponds to the excitation channel that was used during the experiment. 
                           Needs to match one of the options in the corresponding row in the Channels 
                           column in the KeyInformation 
                         
        - KeyInformation : A pandas dataframe that contains the information for only the experiment wanting 
                           to be analyzed. Therefore, if loading a key file with multiple experiments, one
                           must filter the dataframe to only include one experiment. 
        
        - LocOfData      : Local path to the image stack. 

        - NucleusMasks   : Nucleus mask from the cellpose nucleus segmentation 

        - Point          : Optional argument if more than one point was given per image. This is not a common scenario 
                           and may not be functional in the current code.  
    '''

    #Loading the Image Stack and specific channel as specified by the Channel input 
    image_stack = li.GetOnlyOneChannel(WellLabel,Channel,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack)

    #Smooth image if Kernel size is greater than zero. 
    if KernelSize > 0: 
        image_stack = SmoothStack(image_stack,KernelSize)

    #Resizing each image to 512X512 using INTER_NEAREST interpolation 
    image_stack = RescaleStack(image_stack,512,512)

    #Transforming the grayscale image to a color image where the found nucleus are blue 
    image_stack = pi.ProcessingProcedure1(image_stack)

    #CellPose expects the input to be a list. Therefore, changing the image stack to a list 
    image_stack = list(image_stack) 


    
    #Channels are assumed to be be grayscale and and will have a nuclear mask in the blue channel 
    channels =np.zeros((len(image_stack),2)) 
    channels[:,1] = 3 #Sets nucleus mask channel to blue 

    #Model to segment whole cell:
    cyto_model = models.Cellpose(gpu=True, model_type = 'cyto') 

    cell_masks_n, _, _, diams_1 = cyto_model.eval(image_stack,invert=False, diameter=Diameter, channels=[0,0],flow_threshold = 0.8,cellprob_threshold= -1)
    #cell_masks_n, _, _, _       = cyto_model.eval(image_stack[1:],invert=False, diameter=diams_1, channels=channels)

    #cell_masks_n = [cell_masks_1] + cell_masks_n    

    #Converting data into a multi-dimsional matrix 
    cell_masks_n = np.stack(cell_masks_n)

    #Combining Masks: 
    combined_mask = CombineNucleiCytoplasmStack(cell_masks_n,NucleusMasks)

    return combined_mask

def CombineNucleiCytoplasmStack(CellStack,NucleiStack):
    '''
    Returning the combined stack of the cytoplasm and nucleus segmentation. Calls the CombineNucleiCytoplas
    '''
    CombinedMask = np.zeros(CellStack.shape)

    #Combining each slide
    for ith_slide in range(CombinedMask.shape[0]): 
        CombinedMask[ith_slide,:,:] = CombineNucleiCytoplasm(CellStack[ith_slide,:,:],NucleiStack[ith_slide,:,:])
    
    return CombinedMask 

def CombineNucleiCytoplasm(CytoplasmMasks,NucleiMasks):
    '''
    The CombineNucleiCytoplasm function works to combine the segmentation output from cellpose when 
    separating the nucleus and cytoplasm. The input is a single slide of segmentation and the output 
    will be the same segmentation but the nucleus is indicated by an odd number and the surrouding 
    cytoplasm is indciated by the next even number.  

    Note: Only nucleus and cytoplasm combos will be consdered as long as the number of pixels for the 
          whole cell to nuclus ratio is greater than 1.1. This will prevent large nucleus to small cytoplasm 
          area
    '''
    #Finding the maximum number of cytoplasm Mask 
    max_cytoplasm_masks = CytoplasmMasks.max()
    
    output_combined_mask = np.zeros(NucleiMasks.shape,dtype=np.uint16)
    counter_found_cells = 0 
    for ith_cell in range(1,max_cytoplasm_masks+1):
        #Finding the whole cell masks 
        x_loc,y_loc = np.where(CytoplasmMasks==ith_cell)
        
        #Assuming if more than one nucleus if found within the cell, the cell is tossed 
        list_of_found_values = NucleiMasks[x_loc,y_loc]
        
        #Removing the zeros
        list_of_found_values = list_of_found_values[list_of_found_values!=0]
        
        found_nuclei_within_cytoplasm_mask = list(Counter(list_of_found_values).keys())

        if len(found_nuclei_within_cytoplasm_mask)>0: 
            #Finding the nuclei mask locations with the 
            count_nuclei = list(Counter(list_of_found_values).values())
            set_nuclei_loc = found_nuclei_within_cytoplasm_mask[np.argmax(count_nuclei)]
            x_nuclei_loc,y_nuclei_loc = np.where(NucleiMasks == set_nuclei_loc)
            
            #Enforcing a rule that the whole cell mask must be greater than 1.1 times that of the 
            #nuclei. This will prevent small cells with insufficent cytoplasms for estimation. 
            if len(x_loc)/len(y_nuclei_loc)>1.1: 
                counter_found_cells +=1 
                output_combined_mask[x_loc,y_loc] = counter_found_cells*2
                output_combined_mask[x_nuclei_loc,y_nuclei_loc] = counter_found_cells*2 - 1

    
    
    return output_combined_mask

def ConnectTrajectories(Masks,Threshold=5):
    #Maximum Number of Frames: 
    max_frames = Masks.shape[0]

    #Finding the centers and cells labels: 
    previous_centers,found_cells = FindCenters(Masks[0,:,:])

    #Stored Connections: 
    stored_values = np.ones((len(found_cells),max_frames))*-1
    
    #Storing the first set of found cells: 
    stored_values[:,0] = found_cells
    
    from scipy.spatial.distance import cdist
    for ith_frame in range(1,max_frames):
        next_centers,next_found_cells = FindCenters(Masks[ith_frame,:,:])

        dist_matrix = cdist(previous_centers,next_centers)

        #Finding the average minimum difference 
        mean_min = np.mean(np.min(dist_matrix,1))

        if mean_min>Threshold:
            dist_matrix =  dist_matrix - mean_min
            dist_matrix[dist_matrix<-Threshold] = 1000

        for ith,ith_traj in enumerate(found_cells): 
            min_value = dist_matrix[ith,:].min()
            min_index   = dist_matrix[ith,:].argmin()
            loc_of_previous = np.where(stored_values[:,ith_frame-1]==ith_traj)
            if min_value<Threshold: 
                loc_under = np.where(dist_matrix[ith,:] <Threshold)[0]
                if len(loc_under)>=2:
                    cell_candiates = next_found_cells[loc_under] 
                    
                    num_ponints_candiates= [] 
                    curr_frame = Masks[ith_frame,:,:].copy()
                    for ithtest in cell_candiates: 
                        num_ponints_candiates = num_ponints_candiates + [np.sum(np.where(curr_frame==ithtest))]

                    prev_frame = Masks[ith_frame-1,:,:].copy()
                    
                    num_points_prev = np.sum(np.where(prev_frame==ith_traj))
                    
                    abs_difference = abs(num_ponints_candiates-num_points_prev)/num_points_prev
                    
                    if np.min(abs_difference)<0.2:
                        recordthiscell = cell_candiates[np.argmin(abs_difference)]
                        stored_values[loc_of_previous,ith_frame] = recordthiscell
                else:
                    stored_values[loc_of_previous,ith_frame] = next_found_cells[min_index]
        
        found_cells = next_found_cells.copy()
        previous_centers = next_centers.copy()
    
    stored_values = stored_values[stored_values[:,-1]!=-1,:]
    return stored_values

def FindMinDistancesBetweenFrames(Frame1,Frame2,Threshold=5): 
    '''
    Function takes in two mask frames (output mask from cellpose) and connects cells between 
    frames 1 and 2 by finding the minimum distance between the centers of each frame. 
    
    Output is a numpy structured array: 
        - Column 1 = Identity of a cell in frame 1 
        - Column 2 = Identity of a cell in frame 2
        - Column 3 = Distance between the centers of column 1 and 2 
    '''
    from scipy.spatial.distance import cdist
    #Getting the centers and identities for each frame 
    center_1,cell_identity_1 = FindCenters(Frame1)
    center_2,cell_identity_2 = FindCenters(Frame2)
    
    
    #Calculating the distance matrix:
    #Each row is going to be a comparison between a cell in frame 2 to all cells in frame 1 
    distance_matrix = cdist(center_2,center_1)
    
    num_cells_2,num_cells_1 = distance_matrix.shape
    
    #Finding the min distance for each cell in frame 2 to frame 1 
    min_distance    = distance_matrix.min(axis=1)
    
    #Finding the average minimum difference 
    mean_min = np.median(min_distance)

    #If the mean of the minimum distances across each cell in frame 2 is 
    #greater than the threshold, this likely means that the plate shifted 
    #in a uniform way (most likely due to change in media conditions)
    if mean_min>Threshold:
        distance_matrix =  distance_matrix - mean_min
        distance_matrix[distance_matrix<-Threshold] = 1000
        #Need to update the min_distance matrix 
        min_distance = distance_matrix.min(axis=1)
    
    #The location of the minimum distance for each row corresponds to the 
    #index value for cell_indenitiy_1
    index_cell_identity_1 = distance_matrix.argmin(axis=1) 
    
    #Cell Identity 1 list in the same order as the connections to cell 
    #identity 2 
    cells_identity_1_ordered = cell_identity_1[index_cell_identity_1]
    
    #If the minimum distance is greater than the threshold, cells are assumed to be not 
    #connected: 
    cells_identity_1_ordered[min_distance>Threshold] = 0
    
    #Find cells in identity one with no found connection: 
    cells_identity_1_lost =  np.setdiff1d(cell_identity_1,cells_identity_1_ordered)

    #Formatting the data for output in a structure array 
    output = list(zip(cells_identity_1_ordered,cell_identity_2,min_distance))
    
    struct_output = np.array(output,
                             dtype=[('I1', 'i2'), ('I2', 'i2'), ('Distance', 'f4')])
    
    return struct_output

def ConnectTrajectoriesV2(MaskStack,Threshold=5,LengthThreshold=0): 
    '''
    LengthThreshold: The minimum number of percent frames that a cell 
                     trajectory must be tracked to be kept.  
    '''
    n_frames, width, height = MaskStack.shape 
    #n_frames = 10 #for speed and testing
    
    #Instantiating the ith-1, and ith-2 frames. 
    #In the first iteration ith-1 and ith-2 are the same 
    frame_i_minus_1 = MaskStack[0,:,:]
    frame_i_minus_2 = MaskStack[0,:,:]
    
    #PathConnectors: 
    paths = np.zeros((10000,n_frames))
    
    for ith_frame in range(1,n_frames):
        frame_i = MaskStack[ith_frame,:,:]
        
        neighbor_connections     = FindMinDistancesBetweenFrames(frame_i_minus_1,frame_i,Threshold)
        far_neighbor_connections = FindMinDistancesBetweenFrames(frame_i_minus_2,frame_i,Threshold)
        
        for ith,ith_connection in enumerate(neighbor_connections): 
            cell1,cell2,distance = ith_connection
            if cell1!=0: 
                paths = UpdatePaths(paths,cell1,cell2,ith_frame,1)
            else: 
                cell1_f,cell2_f,distance_f = far_neighbor_connections[ith]
                if cell1_f!=0: 
                    paths = UpdatePaths(paths,cell1_f,cell2_f,ith_frame,2)
            
        
        #Redefine frame i-2 
        frame_i_minus_2 = frame_i_minus_1
        
        #Redefine frame i-1
        frame_i_minus_1 = frame_i
      
    #Only return paths that have at least 20% points of the entire movie:
    threshold_of_points = int(n_frames*LengthThreshold/100) 
    number_of_found_cells_per_trajectory = np.sum(paths!=0,axis=1)
    keep_paths = number_of_found_cells_per_trajectory>threshold_of_points

    return paths[keep_paths,:]

def ConnectTrajectoriesV3(MaskStack,Threshold=5): 
    '''
    Assumes that the center of the nucleus will be within the same nucleus in the next frame. 
    This will not work with fast moving cells or cells that are very close together
    '''
    n_frames, width, height = MaskStack.shape 
    
    #Instantiating the ith-1, and ith-2 frames. 
    cellLoc,cellID = FindCenters(MaskStack[0,:,:])
    cellID2,cellTotalPixels = np.unique(MaskStack[0],return_counts=True)
    
    #Need to toss the first value since it has zero in it
    cellID2 = cellID2[1:]
    cellTotalPixels = cellTotalPixels[1:]
    
    
    #PathConnectors: 
    paths = np.zeros((int(1E6),n_frames),dtype=int)
    
    currentMaxIndex = len(cellID)
    paths[0:currentMaxIndex,0] = cellID
    
    
    for ith_frame in range(1,n_frames):
        currentFrame = MaskStack[ith_frame,:,:]
        NextcellLoc,NextcellID = FindCenters(currentFrame)
        NextcellID2,NextcellTotalPixels = np.unique(currentFrame,return_counts=True)
        
        #Remving First value because its background
        NextcellID2 = NextcellID2[1:]
        NextcellTotalPixels = NextcellTotalPixels[1:]
        
        for ithPath,ithCell in enumerate(paths[:,ith_frame-1]): 
            if ithCell == 0: 
                pass
            else:
                indexOfNextFrame = cellLoc[cellID==ithCell].flatten()
                maskValueNextFrame = currentFrame[int(indexOfNextFrame[1]),int(indexOfNextFrame[0])]
                paths[ithPath,ith_frame] = maskValueNextFrame
            if ithPath>currentMaxIndex: 
                break
        
        #Checking for any new cells: 
        newCells = np.setdiff1d(NextcellID,paths[:,ith_frame])
        nNewCells = len(newCells)
        paths[currentMaxIndex:currentMaxIndex+nNewCells,ith_frame] = newCells
        currentMaxIndex += nNewCells
        
        #UPdating previous cell
        cellLoc = NextcellLoc.copy()
        cellID = NextcellID.copy()
        cellID2= NextcellID2.copy()
        cellTotalPixels=NextcellTotalPixels.copy()
        
        
    return paths[0:currentMaxIndex,:].astype(int)
def ConnectTrajectoriesV4(MaskStack,Threshold=5): 
    '''
    Assumes that the center of the nucleus will be within the same nucleus in the next frame. 
    This will not work with fast moving cells or cells that are very close together
    '''
    n_frames, width, height = MaskStack.shape 
    
    #Instantiating the ith-1, and ith-2 frames. 
    cellLoc,cellID = FindCenters(MaskStack[0,:,:])
    cellID2,cellTotalPixels = np.unique(MaskStack[0],return_counts=True)
    
    #Need to toss the first value since it has zero in it
    cellID2 = cellID2[1:]
    cellTotalPixels = cellTotalPixels[1:]
    
    
    #PathConnectors: 
    paths = np.zeros((int(1E6),n_frames),dtype=int)
    
    currentMaxIndex = len(cellID)
    paths[0:currentMaxIndex,0] = cellID
    
    
    for ith_frame in range(1,n_frames):
        currentFrame = MaskStack[ith_frame,:,:]
        NextcellLoc,NextcellID = FindCenters(currentFrame)
        NextcellID2,NextcellTotalPixels = np.unique(currentFrame,return_counts=True)
        
        #Remving First value because its background
        NextcellID2 = NextcellID2[1:]
        NextcellTotalPixels = NextcellTotalPixels[1:]
        for ithPath,ithCell in enumerate(paths[:,ith_frame-1]): 
            if ithCell == 0: 
                pass
            else:
                indexOfNextFrame = cellLoc[cellID==ithCell].flatten()
                xloc = int(indexOfNextFrame[1])
                yloc = int(indexOfNextFrame[0])
                
                sizeOFBox = 15
                x1 = max(0,xloc-sizeOFBox)
                x2 = min(width,xloc+sizeOFBox)
                
                y1 = max(0,yloc-sizeOFBox)
                y2 = min(width,yloc+sizeOFBox)
                
                
                maskValueNextFrame = currentFrame[x1:x2,y1:y2]
                uniquevalues,counts = np.unique(maskValueNextFrame,return_counts=True)
                counts = counts[uniquevalues>0]
                uniquevalues = uniquevalues[uniquevalues>0]
                
                if np.any(counts):
                    paths[ithPath,ith_frame] = uniquevalues[np.argmax(counts)]
            if ithPath>currentMaxIndex: 
                break
        
        #Checking for any new cells: 
        newCells = np.setdiff1d(NextcellID,paths[:,ith_frame])
        nNewCells = len(newCells)
        paths[currentMaxIndex:currentMaxIndex+nNewCells,ith_frame] = newCells
        currentMaxIndex += nNewCells
        
        #UPdating previous cell
        cellLoc = NextcellLoc.copy()
        cellID = NextcellID.copy()
        cellID2= NextcellID2.copy()
        cellTotalPixels=NextcellTotalPixels.copy()
        
        
    return paths[0:currentMaxIndex,:].astype(int)

def ConnectTrajectoriesWithBtrack(MaskStack,Threshold=50): 
    import btrack
    from btrack import datasets

    #bTrack appears to be optimized for image sizes of 1024 X 1024. I am 
    #uncertain if its related to the orginal size of the image or something in the 
    #model that requires the image size. But I get accurate tracking and identification 
    #of dividing cells when I use 1024X1024 but not other sizes such as 512X512 
    #or 2048X2048. Going to hard code it for now. 
    MaskStack = RescaleStack(MaskStack,1024,1024).astype(int)

    
    FEATURES = [
    "area", 
    "major_axis_length", 
    "minor_axis_length", 
    "orientation", 
    "solidity"
    ]

    objects = btrack.utils.segmentation_to_objects(
        MaskStack, 
        properties=tuple(FEATURES)
    )
    #==================================================
    with btrack.BayesianTracker() as tracker:
        # configure the tracker using a config file
        CONFIG_FILE = datasets.cell_config()
        tracker.configure(CONFIG_FILE)
        tracker.max_search_radius = Threshold
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = FEATURES
        tracker.configuration.hypothesis_model.hypotheses = ['P_FP', 'P_init', 'P_term', 'P_link', 'P_dead']

        # append the objects to be tracked
        tracker.append(objects)

        # set the tracking volume
        tracker.volume=((0, MaskStack.shape[1]), (0, MaskStack.shape[2]))

        # track them (in interactive mode)
        tracker.track_interactive(step_size=100)

        # generate hypotheses and run the global optimizer
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari()

        # store the tracks
        tracks = tracker.tracks

        # store the configuration
        cfg = tracker.configuration
    #==================================================
    def ConvertbTrackResults(bTrackResults,MaskStack): 
        x = bTrackResults

        numFrames = int(x[:,1].max())+1
        numPaths  = int(x[:,0].max())+1

        paths = np.zeros((numPaths,numFrames))

        for ith in range(numFrames):     

            filterBy = x[:,1] == ith
            filteredX = x[filterBy]
            for jthRow in filteredX:
                pathIndex = int(jthRow[0])
                idCell = MaskStack[ith,int(jthRow[2]),int(jthRow[3])]
                paths[pathIndex,ith]= idCell



        return paths.astype(int)


    
    
    return ConvertbTrackResults(data,MaskStack),data,properties,graph,tracks

def UpdatePaths(Paths,Cell1,Cell2,FrameNumber,LookBackFrames): 
    found_cells_previous = Paths[:,FrameNumber-LookBackFrames]
    
    #Index of Cell 1 
    i_cell1,= np.where(found_cells_previous==Cell1)

    if i_cell1.size == 0:
        sum_columns = Paths.sum(axis=1)
        first_empty_row = np.min(np.where(sum_columns==0))
        Paths[first_empty_row,FrameNumber-LookBackFrames] = Cell1
        Paths[first_empty_row,FrameNumber] = Cell2
    else: 
        Paths[i_cell1,FrameNumber] = Cell2 
        
    return Paths

def _ConnectAndSaveTrajectories(WellLabel,LocOfOutput,MaxPixelDistance=8,EstimateCytoplasm=False): 
    #Loading the masks 
    if EstimateCytoplasm: 
        nucleus,cytoplasm = GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,20)
    else: 
        nucleus,cytoplasm = GetNucAndCytMasks(WellLabel,LocOfOutput)

    #Using the version 2 of connect trajectories where it allows for the imcomplete 
    #connection of the cells. 
    paths_found, napariout,properties,graphs,tracks = ConnectTrajectoriesWithBtrack(nucleus,MaxPixelDistance)

    colorPallet = GetColorPallet(paths_found)
    
    #Find number of complete paths by first counting the number of missing values in each trajectory and then 
    #finding the number of trajecotries with 0 missed points 
    count_zeros = np.sum(paths_found==0,1)
    complete_paths = np.sum(count_zeros==0)
    partial_paths = np.sum(count_zeros==1)

    print("Well Label: {},Total Paths: {}, Complete Number of Paths found:{} , Number of Paths with one missing connection: {}".format(WellLabel,paths_found.shape[0],complete_paths,partial_paths))

    #Saving the connection in the same location as the masks
    store_location = LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    store_location["Paths"] = paths_found
    store_location['NapariPaths'] = napariout
    store_location['ColorPallet'] = colorPallet
    store_location['Properties'] = properties
    store_location['Graph'] = graphs
    #store_location['Tracks'] = tracks #For some reason I stopped being able to pickle this. It contains a class unique to btrack. PRobably when I updated btrack.
    #therefore not going to include it when saving. 
    np.save(LocOfOutput+"/"+WellLabel+".npy",store_location)
    return paths_found

def ConnectAndSaveTrajectories(WellList,LocOfOutput,MaxPixelDistance=8,EstimateCytoplasm=False): 
    if not isinstance(WellList,list): 
        WellList = [WellList]

    for ith_well in WellList: 
        _ConnectAndSaveTrajectories(ith_well,LocOfOutput,MaxPixelDistance,EstimateCytoplasm)

    return None 

#===========================================================
#Functions associated decompressing and compressing masks
#===========================================================
def CompressMask(Image):
    width,height = Image.shape
    
    unique_pixel_values = np.unique(Image)
    
    #Background pixels are represented by 0 
    foundsegments = unique_pixel_values[unique_pixel_values!=0]

    found_contours = {}
    
    #Need to save the orginal shape of the image: 
    found_contours["OriginalShape"] = Image.shape
    found_contours["UniqueSegments"] = foundsegments

    for ith_contour in foundsegments:
        x_loc,y_loc = np.where(Image==ith_contour)
        
        item_to_contour = np.zeros((width,height),dtype=np.uint8)
        item_to_contour[x_loc,y_loc] = 255
        
        loc_of_borders,hierarchy = cv2.findContours(item_to_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        found_contours[ith_contour] = loc_of_borders
    return found_contours
    
def DecompressMask(ContourDictionary): 
    shape = ContourDictionary["OriginalShape"]
    foundsegments = ContourDictionary["UniqueSegments"]
    #First if statement is preventing against the condition where the size of found segments is zero 
    if foundsegments.size:
        if foundsegments.max() <=255:
            image = np.zeros(shape,dtype=np.uint8)
            for ith_segment in foundsegments: 
                contours = ContourDictionary[ith_segment] 
                image = cv2.fillPoly(image,contours,int(ith_segment))
        else: 
            image = np.zeros(shape,dtype=np.uint32) 
            for ith_segment in foundsegments:
                image_8bit = np.zeros(shape,dtype=np.uint8)
                contours = ContourDictionary[ith_segment] 
                image_8bit = cv2.fillPoly(image_8bit,contours,int(ith_segment))
                x,y = np.where(image_8bit>0)
                image[x,y] = ith_segment
    else: 
        image = np.zeros(shape,dtype=np.uint8)
    
    return image

def CompressImageStack(ImageStack): 
    StoredContours = []
    for ith_image in ImageStack: 
        StoredContours.append(CompressMask(ith_image))
    return StoredContours

def DecompressImageStack(CompressedImages):
    number_of_slides = len(CompressedImages)
    
    decompressed_images = []
    for ith_slide in range(number_of_slides): 
        decompressed_images.append(DecompressMask(CompressedImages[ith_slide]))
    return np.stack(decompressed_images)

def DecompressImageStackParallel(CompressedImages): 
    number_of_slides = len(CompressedImages)
    
    with multiprocess.Pool(5) as pool:
        decompressed_images = pool.map(DecompressMask,list(CompressedImages))
        pool.close()
        pool.join()
    
    return np.stack(decompressed_images)


#===========================================================
#Helper Functions
#===========================================================
def FindCenters(InputImage): 
    unique_values = np.unique(InputImage)
    unique_values = unique_values[unique_values!=0]
    n_unique = unique_values.shape[0]
    
    centers = np.zeros((n_unique,2))
    for nth,nth_traj in enumerate(unique_values): 
        y_loc,x_loc = np.where(InputImage==nth_traj)

        y_avg = int(np.mean(y_loc))
        x_avg = int(np.mean(x_loc))
        
        centers[nth,:] = [x_avg,y_avg]
    return centers,unique_values
    
def FindAllCenters(MaskStack):
    #Number of Frames 
    nFrames = MaskStack.shape[0]
    
    #Max Number of Cells per Frame 
    nMaxCells = int(MaskStack.max()+1)
    
    #Stored Centers: 
    centerStore = np.zeros((nMaxCells,nFrames,2))
    
    with multiprocess.Pool(5) as pool:
        output = pool.map(FindCenters,list(MaskStack))
        pool.close()
        pool.join()
        
    #Formatting Output
    for ithFrame in range(nFrames): 
        centers,index = output[ithFrame]
        index = index.astype(int)
        centerStore[index,ithFrame,:] = centers
        
    return centerStore.astype(int)

def CalculateStatisitic2(MaskStack,ImageStack,Function=np.sum):
    maxCells = int(MaskStack.max())

    
    #Number of Images: 
    nImage = MaskStack.shape[0]
    
    #Matrix where each column represents a single frame and each row represents the cell number 
    #as represented by the mask 
    dataStore = np.zeros((maxCells+1,nImage))

    def CalculateFrame(Image,Mask,Function,InputArray): 
        maxCell = int(Mask.max())+1

        for ithCell in range(1,maxCell): 
            x,y = np.where(Mask==ithCell)
            data = Image[x,y]
            InputArray[ithCell] = Function(data)

    for ithFrame in range(nImage):
        print(ithFrame)
        CalculateFrame(ImageStack[ithFrame],MaskStack[ithFrame],Function,dataStore[:,ithFrame])
    dataStore[dataStore==0]=np.nan
    
    return dataStore
def RescaleStack(Masks,Width,Height): 
    from skimage.transform import resize


    #Finding the number of frames. 
    n_frames= Masks.shape[0]

    #Rescaling masks to be orginal width and height. 
    masks_rescaled = np.zeros((n_frames,Width,Height))
    for ith_mask in range(n_frames):
        #Rescaling using internearest to preserve the values  
        masks_rescaled[ith_mask,:,:] = resize(Masks[ith_mask,:,:],[Width,Height],preserve_range=True,order=0)

    return masks_rescaled

#==========================================================
#Extracting Data from auto masks
#=========================================================
def LoadMasksAndPaths(WellLabel,LocOfOutput,SegmentBothCyAndNuc=True):
    '''
    Extract the nucleus, cytoplasm, and paths in preparation for data analysis. 
    '''
    #Load Nucleus mask and cytoplasm masks 
    if SegmentBothCyAndNuc: 
        nucleus_masks,cytoplasm_mask = GetNucAndCytMasks(WellLabel,LocOfOutput)
    else: 
        nucleus_masks,cytoplasm_mask = GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,20)
        
    #Loading paths: 
    paths = GetPaths(WellLabel,LocOfOutput)

    #Removing Nucleus that is not part of a path 
    nucleus_masks = RemoveNucleusMasks(nucleus_masks,paths)
    cytoplasm_mask = RemoveNucleusMasks(cytoplasm_mask,paths)
    
    return nucleus_masks,cytoplasm_mask,paths 

def GetSegmentStatistics(WellLabel,Channel,KeyInformation,LocOfData,LocOfOutput,BackgroundPercentile = 10,SegmentBothCyAndNuc=True,Point=0,AutomaticBackgroundCorrection = False,ZStack = False):
    nucleus_masks,cytoplasm_mask,paths =LoadMasksAndPaths(WellLabel,LocOfOutput,SegmentBothCyAndNuc)
    
    #Loading Stack: 
    image_stack = li.GetOnlyOneChannel(WellLabel,Channel,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,False)
    n_slides,width,height = image_stack.shape

    #Rescaling to fit orginal image 
    nucleus_masks = RescaleStack(nucleus_masks,width,height)
    cytoplasm_mask = RescaleStack(cytoplasm_mask,width,height)
    
    #Calculating the statsitics: 
    avg_nuc, avg_cyt, avg_backgrond = SegmentStatisticCalculation(image_stack,paths,nucleus_masks,cytoplasm_mask,BackgroundPercentile) 

    if BackgroundPercentile == -1: 
        avg_backgrond[0,:] = ed.GetBackground(WellLabel,image_stack,KeyInformation,LocOfOutput).transpose()
    return avg_nuc,avg_cyt,avg_backgrond

def SegmentStatisticCalculation(ImageStack,Paths,NucMasks,CytMasks,BackgroundPercentile=10):
    n_slides,width,height = ImageStack.shape
    #Finding Average Cytoplasm and Nucleus 
    avg_nuc = np.zeros(Paths.shape)
    avg_cyt = np.zeros(Paths.shape)
    avg_backgrond = np.zeros((1,n_slides))
    
    for ith_slide in range(n_slides): 
        cells_of_interest = Paths[:,ith_slide]
        slide_of_interest = ImageStack[ith_slide,:,:]
        for ith,ith_cell in enumerate(cells_of_interest): 
            if ith_cell == 0: 
                #Incomplete trajectory found 
                avg_nuc[ith,ith_slide] = np.nan
                avg_cyt[ith,ith_slide] = np.nan
            else: 
                x,y = np.where(NucMasks[ith_slide,:,:]==ith_cell)
                avg_nuc[ith,ith_slide] = slide_of_interest[x,y].mean()

                x,y = np.where(CytMasks[ith_slide,:,:]==ith_cell)
                avg_cyt[ith,ith_slide] = slide_of_interest[x,y].mean()
                if not np.any(x): 
                    print(np.unique(CytMasks[ith_slide,:,:]))
                    print(np.unique(NucMasks[ith_slide,:,:]))
                    print("Cell",ith_cell,"Slide",ith_slide,'Traj',ith) 
        
        if BackgroundPercentile != -1: 
            bottom = np.percentile(slide_of_interest,BackgroundPercentile)
            x,y = np.where(slide_of_interest<=bottom)
            avg_backgrond[0,ith_slide] = slide_of_interest[x,y].mean()

    return avg_nuc, avg_cyt, avg_backgrond
    
def GetRatioStatistics(WellLabel,Numerator,Denominator,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc= True,Point=0): 
    #Load Nucleus mask and cytoplasm masks 
    if SegmentBothCyAndNuc: 
        nucleus_masks,cytoplasm_mask = GetNucAndCytMasks(WellLabel,LocOfOutput)
    else: 
        nucleus_masks,cytoplasm_mask = GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,20)
        
    #Loading paths: 
    paths = GetPaths(WellLabel,LocOfOutput)

    #Removing Nucleus that is not part of a path 
    nucleus_masks = RemoveNucleusMasks(nucleus_masks,paths)
    cytoplasm_mask = RemoveNucleusMasks(cytoplasm_mask,paths)
    
    #Loading Stack: 
    manual_masks =  ed.GetPreviousMasks(LocOfOutput)
    ratio_stack = ed.CalculateRatio(WellLabel,Point,Numerator,Denominator,KeyInformation,LocOfData,manual_masks)
    n_slides,width,height = ratio_stack.shape

    #Rescaling to fit orginal image 
    nucleus_masks = RescaleStack(nucleus_masks,width,height)
    cytoplasm_mask = RescaleStack(cytoplasm_mask,width,height)

    avg_nuc, avg_cyt, avg_backgrond = SegmentStatisticCalculation(ratio_stack,paths,nucleus_masks,cytoplasm_mask,BackgroundPercentile=-1) 

    return avg_nuc, avg_cyt, avg_backgrond

def LoadMeanTrajectories(LocOfOutput): 
    try: 
        return np.load(os.path.join(LocOfOutput,"MeanTrajectoryies.npy"),allow_pickle=True).item()
    except: 
        return {}

def ExtractAllData(WellList,ChannelList,KeyInformation,LocOfData,LocOfOutput,BackgroundPercentile = 10,SegmentBothCyAndNuc=True,Point=0,AutomaticBackgroundCorrection = False):
    AllData = LoadMeanTrajectories(LocOfOutput)
    iter_count = -1 
    for ith_well in WellList: 
        iter_count+=1
        print(ith_well, "Percent Done: {}".format(iter_count/len(WellList)))
        for ith_channel in ChannelList: 
            #If the dictionary entry already exists, then we do not want to overwrite other data. 
            try: 
                AllData[ith_well][ith_channel] = {}
            except: 
                AllData[ith_well] = {}
                AllData[ith_well][ith_channel] = {}
            AllData[ith_well][ith_channel] = {}
            nuc,cyt,back = GetSegmentStatistics(ith_well,ith_channel,KeyInformation,LocOfData,LocOfOutput,BackgroundPercentile,SegmentBothCyAndNuc,Point,AutomaticBackgroundCorrection)
            AllData[ith_well][ith_channel]["Nucleus"] = nuc
            AllData[ith_well][ith_channel]["Cytoplasm"] = cyt
            AllData[ith_well][ith_channel]["Background"] = back
    np.save(os.path.join(LocOfOutput,"MeanTrajectoryies.npy"),AllData)
    print("Done Extracting Data")

def ExtractRatioData(WellList,Numerator,Denominator,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc= True,Point=0):
    AllData = LoadMeanTrajectories(LocOfOutput)
    iter_count = -1 
    
    DictionaryLabel = Numerator + "/" + Denominator

    for ith_well in WellList: 
        iter_count+=1
        print(ith_well, "Percent Done: {}".format(iter_count/len(WellList)))
        #If the dictionary entry already exists, then we do not want to overwrite other data. 
        try: 
            AllData[ith_well][DictionaryLabel] = {}
        except: 
            AllData[ith_well] = {}
            AllData[ith_well][DictionaryLabel] = {}
        nuc,cyt,back = GetRatioStatistics(ith_well,Numerator,Denominator,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc,Point)
        AllData[ith_well][DictionaryLabel]["Nucleus"] = nuc
        AllData[ith_well][DictionaryLabel]["Cytoplasm"] = cyt
        AllData[ith_well][DictionaryLabel]["Background"] = back
    np.save(os.path.join(LocOfOutput,"MeanTrajectoryies.npy"),AllData)

def CalculateBackgroundCorrectedRatio(Numerator, Denominator, Background): 
    return (Numerator - Background)/(Denominator - Background)

def ExtractInfoFromDictionary(InputDictionary,Well,Channel): 
        nuc  = InputDictionary[Well][Channel]["Nucleus"] 
        cyt  = InputDictionary[Well][Channel]["Cytoplasm"] 
        background = InputDictionary[Well][Channel]["Background"]
        return nuc,cyt,background 
    
def CalculateCytoplasmToNucleusRatio(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return CalculateBackgroundCorrectedRatio(cyt,nuc,background )

def GetBackground(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return background

def GetCytoplasmMean(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return cyt

def GetBackgroundCorrectedCytoplasmMean(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return cyt - background

def GetNucleusMean(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return nuc

def GetBackgroundCorrectedNucleusMean(InputDictionary,Well,Channel): 
    nuc,cyt,background = ExtractInfoFromDictionary(InputDictionary,Well,Channel)
    return nuc - background
    
def CombineReplicates(TargetFunction,WellLabels,Channel,LocOfOutput,CorrectLinearShift=False,ShowLinearShift=False): 
    if not isinstance(WellLabels,list): 
        WellLabels = [WellLabels]
    calculated_means = LoadMeanTrajectories(LocOfOutput)
    
    data_2_return = TargetFunction(calculated_means,WellLabels[0],Channel)
    
    for ith_replicate in range(1,len(WellLabels)): 
        next_data = TargetFunction(calculated_means,WellLabels[ith_replicate],Channel)
        data_2_return = np.append(data_2_return,next_data,axis=0)

    if CorrectLinearShift: 
        average_output,linear_fit_out,linear_fit_pars  = GetAverageAcrossPoints(TargetFunction,WellLabels,Channel,LocOfOutput,CorrectLinearShift,True)
        data_2_return = data_2_return - linear_fit_out + linear_fit_pars[1]
        if ShowLinearShift: 
            average_notcorrected = GetAverageAcrossPoints(TargetFunction,WellLabels,Channel,LocOfOutput)
            plt.figure(figsize=(15,5))
            plt.plot(average_notcorrected)
            plt.plot(linear_fit_out)
            plt.plot(average_output)
            plt.grid()
            plt.title("Display of how the Corrected Linear Shift Changed the Mean for {}\n Slope: {} ; Y-Intercept: {}".format(Channel,linear_fit_pars[0],linear_fit_pars[1]))
            plt.xlabel("Slide Number",size=15)
            plt.ylabel(pl.ReturnYAxisLabel(TargetFunction),size=15)
            plt.legend(["Uncorrected Average","Linear Fit Output","Corrected Average"])
    
    return data_2_return

def GetAverageAcrossPoints(FunctionType,WellList,Channel,LocOfOutput,CorrectLinearShift = False,ReturnLinearData=False):
    #Extract data that has already been saved: 
    extracted_data = CombineReplicates(FunctionType,WellList,Channel,LocOfOutput)

    #Any value that is 10X the median is assumed to be outliers 
    max_value = abs(np.nanmedian(extracted_data)*10)

    #Setting those values to nan
    extracted_data[extracted_data>max_value]= np.nan
    
    #Any value less than 10X the negative median is assumed to be an outlier 
    extracted_data[extracted_data<-max_value]= np.nan
    
    #Average across all cells excluding the nans 
    average_output = np.nanmean(extracted_data,axis=0)

    #Correcting linear shifting due to photobleaching or other reasons
    if CorrectLinearShift:
        average_output,linear_fit_out,linear_fit_pars = SetSlopeOfDataToZero(average_output)
        if ReturnLinearData: 
            return average_output,linear_fit_out,linear_fit_pars 

    return average_output

def SetSlopeOfDataToZero(TrajectoryData): 
    '''
    The input is a 1XD array that contains time trajectory data that has an apparant linear shift 
    not due to the mechanism of study. The function fits a best fit linear line and subtracts the 
    slope to make the output trajectory data have a slope of zero. 
    '''
    x_range = range(len(TrajectoryData))
    linear_fit,residuals, rank, singular_values, rcond = np.polyfit(x_range,TrajectoryData,1,full=True)

    #Results of the linear model 
    linear_fit_out = np.polyval(linear_fit,x_range)

    #Corrected mean Trajectory 
    corrected_mean_trajectory = TrajectoryData - linear_fit_out + linear_fit[1]
    
    #Returns the corrected trajectories, linear fit results, and the linear fit parameters. 
    return corrected_mean_trajectory, linear_fit_out, linear_fit

#==========================================================
#General Path Statistics: 
#=========================================================

def GetPathStatistics(WellLabel,LocOfOutput):
    found_paths = GetPaths(WellLabel,LocOfOutput)
    return GetPathStatisticsDirect(FoundPaths)
    

def GetPathStatisticsDirect(FoundPaths): 
    output_dictionary = {}
    output_dictionary["TotalTrajectoriesFound"] = FoundPaths.shape[0]
    
    #Paths that are complete from start to finish
    num_of_missing_values = np.sum(FoundPaths==0,axis=1)
    output_dictionary["CompletePaths"] = np.sum(num_of_missing_values==0)
    
    #Number of paths active in the end
    paths_still_active_at_end = FoundPaths[:,-1]!=0
    output_dictionary["ActivePathsEnd"]  = np.sum(paths_still_active_at_end)
    
    #Of the paths that are active in the end, report the maximum starting index
    active_paths = FoundPaths[paths_still_active_at_end,:]
    
    #Length of each trajectory still active in the end: 
    length_of_each_trajectory = np.sum(FoundPaths!=0,axis=1)
    output_dictionary["LengthOfEachTrajectory"] = length_of_each_trajectory
    
    #Minimum 
    return output_dictionary


def DisplayPathStatistics(KeyInformation,LocOfOutput):
    '''
    Get the statistics of the found paths from autodetection
    '''
    #Total Number of Points: 
    n_points = KeyInformation.shape[0]
    
    categories = ExtractSpecificConditions(KeyInformation)
    
    #Output DataFrame
    output = pd.DataFrame(np.zeros((n_points,6)),columns=["Title",
                                                          "WellLabel",
                                                          "Total Paths",
                                                          "Complete Paths",
                                                          "Active Paths In End",
                                                          "Minimum Path Length"])
    
    ith_entry = 0 
    for ith_category in categories.keys():
        total = 0 
        for ith_well in categories[ith_category]:
            output_dictionary=GetPathStatistics(ith_well,LocOfOutput)
            output.loc[ith_entry,"Title"] = ith_category
            output.loc[ith_entry,"WellLabel"] = ith_well
            output.loc[ith_entry,"Total Paths"] = output_dictionary["TotalTrajectoriesFound"]
            output.loc[ith_entry,"Complete Paths"] = output_dictionary["CompletePaths"]
            output.loc[ith_entry,"Active Paths In End"] = output_dictionary["ActivePathsEnd"]
            output.loc[ith_entry,"Minimum Path Length"] = output_dictionary["LengthOfEachTrajectory"].min()
            ith_entry+=1
            
    return output

def ExtractSpecificConditions(KeyInformation): 
    '''
    Returns a dictionary where each key represents a title and the corresponding well labels  
    '''
    output_dict = {}
    unique_titles = KeyInformation["Title"].unique()
    for ith_title in unique_titles:
        output_dict[ith_title] = KeyInformation[KeyInformation["Title"]==ith_title]["Well"].values
        
    return output_dict

def GetSpecificLocations(ExpCode,LocOfData,LocOfOutput,SpecificFolder,LocOfKey): 
    '''
    Useful for getting the path location and key specific information when anyalzing multiple experiments with different codes 
    '''
    full_path_data   = os.path.join(LocOfData,SpecificFolder)
    full_path_output =  os.path.join(LocOfOutput,SpecificFolder)
    key_specifics = pd.read_csv(LocOfKey)
    key_specifics = key_specifics[key_specifics["Expierment Code"]== ExpCode]
    return full_path_data,full_path_output, key_specifics


def ExtractCellInfo(WellLabel,LocOfOutput,Rescale = 1024,SegmentBothCyAndNuc=True): 
    '''
    Extracting and saving characteristics of cytoplasm and nuclear mask. Outputs 
    a dictionary containing general pixel characteristics of each cytoplasm and nucleus 

    The dictionary contains four keys
    "Average Nucleus Pixel Area Per Frame"-------->Pandas dataframe containing the mean and standard deviation of the cytoplasm pixel area 
    "Average Cytoplasm Area Per Frame"------------>Pandas dataframe containing the mean and standard deviation of the cytoplasm pixel area 
    "Nucleus Pixel Area Per Cell"------------------->Numpy list of individual cell nucleus pixel areas 
    "Cytoplasm Pixel Area Per Cell"----------------->Numpy list of individual cell cytoplasm pixel areas 
    
    '''
    #Loading nucleus, cytoplasm, and paths 
    nucleus_masks,cytoplasm_mask,paths  = LoadMasksAndPaths(WellLabel,LocOfOutput,SegmentBothCyAndNuc)
    
    #Rescaling the input images to match the Rescale value. 
    nucleus_masks = RescaleStack(nucleus_masks,Rescale,Rescale)
    cytoplasm_mask = RescaleStack(cytoplasm_mask,Rescale,Rescale)
    
    #Determining the number of 
    n_slides = nucleus_masks.shape[0]
    
    store_nucleus_pixel_area = np.zeros_like(paths)
    store_cytoplasm_pixel_area = np.zeros_like(paths)
    
    #====================================================================================
    #                          SubFunctions
    #====================================================================================
    def CreateDataFrame(): 
        return pd.DataFrame((np.zeros((n_slides,2))),columns=['Mean','STD'])

    def CalculateMaskStatistics(Mask,SlideNumber,PerCellStorage,MeanSTDStorage,Paths): 
        '''
        This function is used to calculate statistics on the mask which include the 
        sum, standard deviation, and mean of each found cell in cellpose. Note, zero 
        is assumed to be background: 
        '''
        (unique, counts) = np.unique(Mask[SlideNumber,:,:], return_counts=True)
        for i,label in enumerate(unique): 
            if i == 0: 
                #First label is assumed to be the background pixel 
                pass
            else: 
                loc_of_storage, = np.where(Paths[:,SlideNumber]==label)
                if len(loc_of_storage)==1: 
                    PerCellStorage[loc_of_storage[0],SlideNumber] = counts[i]
                else: 
                    print("WARNING: Mask and Paths are not matching")
                
        MeanSTDStorage.iloc[SlideNumber,0] = np.mean(counts[1::])
        MeanSTDStorage.iloc[SlideNumber,1] = np.std(counts[1::])

        return PerCellStorage,MeanSTDStorage

    
    #====================================================================================
    #                          Calculations
    #====================================================================================
    #----------------------------STORAGE UNITS-------------------------------------------
    avg_nucleus_pixel_area_per_frame   = CreateDataFrame()
    avg_cytoplasm_pixel_area_per_frame = CreateDataFrame()
    store_nucleus_pixel_area = np.zeros(paths.shape)
    store_cytoplasm_pixel_area = np.zeros(paths.shape)
    
    for ith_slide in range(n_slides): 
    
        store_nucleus_pixel_area,avg_nucleus_pixel_area_per_frame = CalculateMaskStatistics(nucleus_masks,
                                                                                            ith_slide,
                                                                                            store_nucleus_pixel_area,
                                                                                            avg_nucleus_pixel_area_per_frame,
                                                                                            paths)

        store_cytoplasm_pixel_area,avg_cytoplasm_pixel_area_per_frame = CalculateMaskStatistics(cytoplasm_mask,
                                                                                            ith_slide,
                                                                                            store_cytoplasm_pixel_area,
                                                                                            avg_cytoplasm_pixel_area_per_frame,
                                                                                            paths)
    
    
    #Set values equal to zero to np.nan . This is mainly to make plotting nicer 
    store_cytoplasm_pixel_area[store_cytoplasm_pixel_area==0] = np.nan
    store_nucleus_pixel_area[store_nucleus_pixel_area==0] = np.nan
    
    output_dict = {"Average Nucleus Area Per Frame":avg_nucleus_pixel_area_per_frame,
                   "Average Cytoplasm Area Per Frame":avg_cytoplasm_pixel_area_per_frame, 
                   "Nucleus Pixel Area Per Cell": store_nucleus_pixel_area, 
                   "Cytoplasm Pixel Area Per Cell": store_cytoplasm_pixel_area }
    
    return output_dict 


def ExtractCellInfoAndSave(Key,LocOfOutput,Rescale = 1024,SegmentBothCyAndNuc=True):
    '''
    Exact all data from an experiment given a key.
    '''
    #Unique Titles: 
    unique_titles = Key['Title'].unique()
    
    n_samples = Key.shape[0]
    
    output_dict = {}
    
    counter=1
    #Looping through each title and well/point 
    for ith_title in unique_titles: 
        locations = Key[Key["Title"]==ith_title]['Well']
        output_dict[ith_title] = {}
        for ith_location in locations:  
            output_dict[ith_title][ith_location] = ExtractCellInfo(ith_location,LocOfOutput,Rescale,SegmentBothCyAndNuc)
            print("Fraction Done: {}".format(counter/n_samples))
            counter+=1
    
    np.save(LocOfOutput+"/"+"MaskStatistics.npy",output_dict)
    return output_dict

def SaveMeanTrajectoriesToExcel(LocOfResults):
    '''
    This will convert a dictionary of results to a CSV file 
    so a user with minimal knowledge of python can utilize the
    results 
    '''

    X = LoadMeanTrajectories(LocOfResults)
    with pd.ExcelWriter(LocOfResults+'/Data.xlsx') as writer:  
        for ith in X.keys():
            for jth in X[ith].keys():
                for kth in X[ith][jth].keys():
                    t = X[ith][jth][kth]
                    dataframe = pd.DataFrame(t.transpose())
                    sheet = ith+"_"+jth+"_"+kth
                    sheet=sheet.replace("/","_")
                    dataframe.to_excel(writer, sheet_name=sheet)
    return None


def CalculateFrame(Args): 
    Image,Mask,maxCellsGlobal,Function = Args
    maxCell = int(Mask.max())+1
    array = np.zeros(maxCellsGlobal+1)
    for ithCell in range(1,maxCell): 
        x,y = np.where(Mask==ithCell)
        data = Image[x,y]
        array[ithCell] = Function(data)
    return array
def CalculateStatisitic(MaskStack,ImageStack,Function=np.sum):
    maxCellsGlobal = int(MaskStack.max())

    
    #Number of Images: 
    nImage = MaskStack.shape[0]
    

   
    output = []
    ImageStack = np.array(ImageStack)
    Args= [(ImageStack[i],MaskStack[i],maxCellsGlobal,Function) for i in range(nImage)]
    
    
        
    with multiprocess.Pool(5) as pool:

        output =  pool.map(CalculateFrame, Args)

        pool.close()
        pool.join()
    
    #for ithFrame in range(nImage):
    #    output.append(CalculateFrame(ImageStack[ithFrame],MaskStack[ithFrame]))

    output =np.stack(output).transpose()
    output[output==0] = np.nan
    return output
def CreateStatsTrack(UnConnectedData,Position,LocOfOutput): 
    '''
    The function take in a dicationary that contains unconnected cell data for multiple positions and 
    will connect the positions based off of the information in the .npy file in the LocOfOutput 
    '''

    data2Connect = UnConnectedData[Position]
    
    paths = np.load(os.path.join(LocOfOutput,Position+'.npy'),allow_pickle=True).item()['Paths']
    
    acceptedPaths = paths
    allPaths = []

    for ith in acceptedPaths:  
        singleTraj = np.zeros(ith.shape[0])
        for i,j in enumerate(ith): 
            singleTraj[i] = data2Connect[j,i] 
        allPaths.append(singleTraj)
    
    return np.stack(allPaths)

def CalculateListOfPositions(ListOfPositions,
                             Channel,
                             Key,
                             LocOfData,
                             LocOfOutput,
                             BackgroundPercentile=10,
                             Function=np.sum,
                             CalculateNucleus=True,
                             KernelSize=3,
                             Buffer=0,
                             BioFormats=False,
                             ZProject = False):
    output = {}
    for ithWell in ListOfPositions:
        
        image = li.GetOnlyOneChannel(ithWell,Channel,Key,LocOfData,BioFormats=BioFormats,ZProject = ZProject,CorrectShape=4)
        print(image.shape)

        #Subtracting Background 
        if BackgroundPercentile==-1:
            print("Manually Selected Background Used")
            background = ed.GetBackground(ithWell,image,Key,LocOfOutput)[:,:,np.newaxis]
        elif BackgroundPercentile==-2: 
            print("Mode of Each Frame Used as background")
            nFrames,width,height = image.shape
            background,_ = scipy.stats.mode(np.reshape(image,(nFrames,width*height)),1,keepdims=True)
            background  = background[:,:,np.newaxis].astype(np.float32)
        else:
            print("Bottom {} Percentile used as background".format(BackgroundPercentile))
            background = np.percentile(image,BackgroundPercentile,axis=(1,2),keepdims=True)
        print("Processing: {}".format(ithWell))
        image = image - background
        image[image<0] = 0

        width,height = image.shape[1:]

        nMask,cMask = GetNucleusAndEstimateCyto(ithWell,LocOfOutput,KernelSize,Buffer)
        if CalculateNucleus: 
            Mask = RescaleStack(nMask,width,height)
            print("Using the Nucleus Mask")
        else: 
            Mask = RescaleStack(cMask,width,height)
            print("Using the Cytoplasm Mask Estimated from the nucleus")

        output[ithWell] = CalculateStatisitic(Mask,image,Function)


    return output 

#===============================================================================
#AutoDetection Correction 
#===============================================================================

def MergePaths(Paths,PathIndex1,PathIndex2,MergeIndex): 
    '''
    This function is used to swap rows to connect trajectories. It will create a
    new trajectory if the current path continues past the swap point. This is
    where cell connection swaps paths 
    '''

    if PathIndex1==PathIndex2: 
        pass
    else:
        path1 = Paths[PathIndex1,:].copy()
        path2 = Paths[PathIndex2,:].copy()
        
        subPath1 = path1[0:MergeIndex]
        subPath2 = path2[MergeIndex:]
        
        newPath = np.append(subPath1,subPath2)
        
        
        Paths[PathIndex1,:] = newPath
        Paths[PathIndex2,MergeIndex:] = 0
        
        #Creating a new path if tracking continued on a different cell 
        subPath1PastMergeIndex = path1[MergeIndex:]
        
        #If the path continues past the merge point, then the tracking 
        #started to track a new cell. New path will be made
        if subPath1PastMergeIndex.sum()>0: 
            newRow = np.zeros_like(path1)
            newRow[MergeIndex:] = subPath1PastMergeIndex
            Paths = np.vstack([Paths,newRow])
        
        #If Path 2 only contains zeros then we can toss the row 
        if np.sum(Paths[PathIndex2,:])==0: 
            Paths = np.delete(Paths,PathIndex2,0)

    Paths = np.vstack([Paths[PathIndex1],Paths])
    Paths = np.delete(Paths,PathIndex1+1,0)
    
    
    return Paths