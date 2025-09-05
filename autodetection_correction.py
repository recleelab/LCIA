import numpy as np
from numpy.core.records import record
from LCIA import load_images as li
from LCIA import auto_detection as ad
from LCIA import process_images as pi


import cv2

import javabridge

def ManuallyLabelPaths(WellLabel,Channel,Channel2,Channel3,KeyInformation,LocOfData,LocOfOutput,SegmentBothCyAndNuc = True,Point=0,AutomaticBackgroundCorrection = False,ZStack = False): 
    #Loading the Image Stack
    image_stack = li.GetOnlyOneChannel(WellLabel,Channel,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,KillJavaBridge=True)
    image_stack2 = li.GetOnlyOneChannel(WellLabel,Channel2,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,KillJavaBridge=True)
    image_stack3 = li.GetOnlyOneChannel(WellLabel,Channel3,KeyInformation,LocOfData,Point,AutomaticBackgroundCorrection,ZStack,KillJavaBridge=True)
    n_frames,width,height = image_stack.shape

    #Loading the masks: 
    if SegmentBothCyAndNuc: 
        masks,cytoplasm_masks = ad.GetNucAndCytMasks(WellLabel,LocOfOutput)
    else: 
        masks,cytoplasm_masks = ad.GetNucleusAndEstimateCyto(WellLabel,LocOfOutput,20)


    #Creating the paths 
    store_location = ad.LoadPreviousAnnotatedMasks(LocOfOutput,WellLabel)
    previous_paths = store_location["Paths"]
    currentPath = 0 
    maxCells = 10000
    paths = np.zeros((maxCells,n_frames))
    if (previous_paths.shape[0]<100):
        paths[0:previous_paths.shape[0]] = previous_paths
        currentPath = previous_paths.shape[0]

    pathsPositive = np.sum(paths>0,axis=1)>0
    numPostivePaths = np.sum(pathsPositive>0)
    
    print("Starting at: {}".format(currentPath))

    recordPath = False

    masks =  ad.RescaleStack(masks,width,height)
    top_percentile_blue = 100
    bottom_perentile_blue = 0
    top_percentile_green  = 100
    bottom_perentile_green = 0

    top_percentile_red = 100
    bottom_perentile_red = 0

    def DisplayImage(ImageStack,CurrentImage,WindowName): 
        image2Show = ImageStack[CurrentImage,:,:]
        topPercentile,bottomPercentile = np.percentile(image2Show,[top_percentile_blue,bottom_perentile_blue])
        image2Show = pi.EnhanceContrast(image2Show,top=topPercentile,bottom=bottomPercentile)

        image2Show2 = image_stack2[CurrentImage,:,:]
        topPercentile,bottomPercentile = np.percentile(image2Show2,[top_percentile_green,bottom_perentile_green])
        image2Show2= pi.EnhanceContrast(image2Show2,top=topPercentile,bottom=bottomPercentile)

        image2Show3 = image_stack3[CurrentImage,:,:]
        topPercentile,bottomPercentile = np.percentile(image2Show3,[top_percentile_red,bottom_perentile_red])
        image2Show3= pi.EnhanceContrast(image2Show3,top=topPercentile,bottom=bottomPercentile)


        colorImage = np.zeros((width,height,3),dtype=np.uint8)
        
        if top_percentile_blue != 0 :
            colorImage[:,:,0] = image2Show

        if top_percentile_green !=0:
            colorImage[:,:,1] = image2Show2
        
        if top_percentile_red !=0:
            colorImage[:,:,2] = image2Show3

        for ithCell in range(maxCells): 
            cellLabel = paths[ithCell,current_image]
            if cellLabel>0: 
                xloc,yloc = np.where(masks[current_image]==cellLabel)
                if cellLabel == paths[currentPath,CurrentImage]:
                    colorImage[xloc,yloc,:] = (252, 157, 3)
                else:
                    colorImage[xloc,yloc,:] = (0,165,255)
            if not pathsPositive[ithCell]: 
                break 



        cv2.imshow(WindowName,colorImage)
        return None

    WindowName = WellLabel+"////"+Channel
    xSave,ySave = 0,0
    current_image = 0
    def MouseCommands(event,x,y,flags,param):
        nonlocal xSave, ySave
        if event == cv2.EVENT_MOUSEMOVE:
            maskID = masks[current_image,y,x]
            xSave,ySave = x,y
            maskID = masks[current_image,ySave,xSave]
            print("X: {}, Y: {}, Intensity: {}, Mask ID: {}".format(x,y,image_stack[current_image,y,x],maskID)) 
            if recordPath: 
                paths[currentPath,current_image] = maskID
                DisplayImage(image_stack,current_image,WindowName)

    
    DisplayImage(image_stack,current_image,WindowName)
    cv2.setMouseCallback(WindowName,MouseCommands)



    while True:
        store_waitkey = cv2.waitKey(1000)
        if store_waitkey & 0xFF == ord('q'):
            break
        elif (store_waitkey & 0xFF == ord('f')): #Right Arrow
            current_image += 1 
            
            if current_image >= n_frames: 
                current_image = 0 
                recordPath=False
            
            maskID = masks[current_image,ySave,xSave]
            if recordPath: 
                paths[currentPath,current_image] = maskID
                pathsPositive = np.sum(paths>0,axis=1)>0
            
            print("Current Image Number: {}".format(current_image))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('d')): #Left Arrow
            current_image -= 1
            if current_image<0:
                current_image = n_frames-1
            print("Current Image Number: {}".format(current_image))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('k')): 
            top_percentile_blue+=1 
            if top_percentile_blue>100: 
                top_percentile_blue=0
            print("Blue Channel Image Top Percentile:{}".format(top_percentile_blue))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('j')): 
            top_percentile_blue-=1 
            if top_percentile_blue<0: 
                top_percentile_blue=100
            print("Blue Channel Image Top Percentile:{}".format(top_percentile_blue))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('l')): 
            bottom_perentile_blue+=1 
            bottom_perentile_blue = min(bottom_perentile_blue,100)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_blue))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord(';')): 
            bottom_perentile_blue-=1 
            bottom_perentile_blue = max(bottom_perentile_blue,0)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_blue))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('i')): 
            top_percentile_green+=1 
            if top_percentile_green>100: 
                top_percentile_green=0
            print("Green Channel Image Top Percentile:{}".format(top_percentile_green))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('u')): 
            top_percentile_green-=1 
            if top_percentile_green<0: 
                top_percentile_green=100
            print("Green Channel Image Top Percentile:{}".format(top_percentile_green))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('p')): 
            bottom_perentile_green+=1 
            bottom_perentile_green = min(bottom_perentile_green,100)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_green))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('o')): 
            bottom_perentile_green-=1 
            bottom_perentile_green = max(bottom_perentile_green,0)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_green))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('m')): 
            top_percentile_red+=1 
            if top_percentile_red>100: 
                top_percentile_red=0
            print("Red Channel Image Top Percentile:{}".format(top_percentile_red))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('n')): 
            top_percentile_red-=1 
            if top_percentile_red<0: 
                top_percentile_red=100
            print("Red Channel Image Top Percentile:{}".format(top_percentile_red))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord('.')): 
            bottom_perentile_red+=1 
            bottom_perentile_red = min(bottom_perentile_red,100)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_red))
            DisplayImage(image_stack,current_image,WindowName)
        elif (store_waitkey & 0xFF == ord(',')): 
            bottom_perentile_red-=1 
            bottom_perentile_red = max(bottom_perentile_red,0)
            print("Blue Channel Image Bottom Percentile:{}".format(bottom_perentile_red))
            DisplayImage(image_stack,current_image,WindowName)       
        elif (store_waitkey) & 0xFF == ord('t'):
            recordPath = not recordPath
            if recordPath: 
                print("Starting to Record path: {}".format(currentPath))
                current_image = 0
                DisplayImage(image_stack,current_image,WindowName)
            else: 
                print("Stoping Recoring Path")
        elif (store_waitkey) & 0xFF == ord('r'):
            if pathsPositive[currentPath]:
                currentPath+=1 
            DisplayImage(image_stack,current_image,WindowName)
            print("The Current Path to Update is: {}".format(currentPath))
        elif (store_waitkey) & 0xFF == ord('e'):
            currentPath-=1 
            currentPath = max(0,currentPath)
            DisplayImage(image_stack,current_image,WindowName)
            print("The Current Path to Update is: {}".format(currentPath))
        
    
    print(paths[pathsPositive,:])
    cv2.destroyAllWindows()
    javabridge.kill_vm()

    store_location["Paths"] = paths[pathsPositive,:]
    np.save(LocOfOutput+"/"+WellLabel+".npy",store_location)