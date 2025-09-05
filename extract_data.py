import numpy as np
import pandas as pd
from LCIA import load_images as li
from LCIA import process_images as pi
import cv2
import time 
import matplotlib.pyplot as plt 
import os

def APIPF(InputData,ChannelToUse=None):
    """
    Calculates the "Average Pixel Intensity Per Frame". This can be useful as a crude measure to see if the cells are moving out of the frame or dying. This assumes that the cells tend to be brighter than the background. A decreasing average pixel intensities implies that the cells are either shrinking or moving out of the frame.
    Input:
        - Data in the form of (<Number of Frames>, <Number of Points>, <Number of Channels>, <Pixel Values Width>,<Pixel Values Height>) OR (<Number of Frames>, <Number of Points>, <Pixel Values Width>,<Pixel Values Height>)
        - If the channel has already been filtered, the channel to use input is not necessary
    Output:
        - A Pandas dataframe where the rows represent frame number and the columns represent the average pixel value of the frame
    """

    #Extracing Information from the shape of the input
    if ChannelToUse is not None:
        frames, points, channels, width,height = InputData.shape
    else:
        frames, points, width,height = InputData.shape

    #Reducing the data to only include the channel of interest as defined by ChannelToUse
    if ChannelToUse is not None:
        reduced_data = InputData[:,:,ChannelToUse,:,:]
    else:
        reduced_data = InputData

    #Calculating the average pixel intensity per frame and point
    mean_data = reduced_data.mean(axis=(2,3))

    max_data = np.percentile(reduced_data,95)
    min_data = np.percentile(reduced_data,5)

    #Creating the column labels:
    col_labels = ["Point: {}".format(ith+1) for ith in range(points)]



    return pd.DataFrame(mean_data,columns=col_labels),min_data,max_data

def AnnotateImages(WellLabel, KeyInformation, LocOfData,Channel,Point=0,PrevMasks=None,AutomaticBackgroundCorrection=False,ZProject = True,BioFormats=True,SkipFrames=1):
    
    
    #Determining the number of channels based off of the information in the key 
    channels = KeyInformation[KeyInformation["Well"]==WellLabel].squeeze().Channels
    nChannels = len(channels.split("//"))

    image_stack = li.GetOnlyOneChannel(WellLabel,Channel, KeyInformation, LocOfData, Point ,AutomaticBackgroundCorrection ,ZProject,False,CorrectShape=nChannels)
    image_stack = image_stack[::SkipFrames]
    window_name = KeyInformation[KeyInformation["Well"]==WellLabel].squeeze()['Title']
    window_name = window_name + " /// Well:{}".format(WellLabel) +" Point: {}".format(Point)
    return DisplayStack(image_stack,window_name,PrevMasks,ZProject)

def DisplayStack(ImageStack,WindowName = "Default",PrevMasks = None,ZProject=True):

    #This is important for generating the image for display with approiate header.
    header_size = 50 #px

    #Getting the height and width of the images
    _,width,height = ImageStack.shape

    #Clipping Values Min and Max Assumes 8-bit imaging'
    change_clippings = False
    min_clipping = 0
    max_clipping = 255

    #Switch between cmap and grayscale:
    cmapOption = False

    #Toggle the auto record:
    toggle_auto_record = False
    x_record, y_record = -1, -1

    #Display BackgroundCorrection
    backgroundCorrection = False

    #Toggle Find Length: 
    findlength= False
    toggle_i = 0 

    def GetDisplayImage(ImageStack,ImageToDisplay):
        nonlocal header_size,stored_info, background
        nonlocal current_cell,current_type
        nonlocal min_clipping, max_clipping
        nonlocal cmapOption,backgroundCorrection
        _,width,height = ImageStack.shape

        region_names = {1:'Cytoplasm', 0:'Nucleus',  -1: 'Background',-2: 'Set Standard Box Width',-3: 'Modify Image'}
        color_pallet = {1:(0,255,0),0: (0,0,255), -1: (255,0,0), -2: (255,255,0),-3: (164,67,138)}
        Image_I = ImageStack[ImageToDisplay,:,:].copy()

        #Subtract Background:
        currentBackgroundCoordinates = background.squeeze()[current_img,:]
        background_set = not np.any(currentBackgroundCoordinates<0) #Check if background has been set
        if backgroundCorrection:
            if background_set:
                mean_background,max_background = CalculateBackgroundInfo(Image_I,currentBackgroundCoordinates)
                Image_I = Image_I - mean_background
                Image_I[ Image_I < (max_background - mean_background)] = 0
            else:
                print("Background not set, cannot subtract background")

        #Enhance Contrast
        Image_I = pi.EnhanceContrast(Image_I,max_clipping,min_clipping)

        if cmapOption:
            Image_I = cv2.applyColorMap(Image_I,cv2.COLORMAP_JET)
        else:
            Image_I = np.dstack((Image_I,Image_I,Image_I))

        #Draw Nucleus Rectangles
        DrawRectangles(Image_I,ImageToDisplay,stored_info[0,:,:],color_pallet[0],DisplayCellNumber=True)

        #Draw Cytoplasm Rectangles
        DrawRectangles(Image_I,ImageToDisplay,stored_info[1,:,:],color_pallet[1])

        #Draw Background Rectangle:
        DrawRectangles(Image_I,ImageToDisplay,background,color_pallet[-1],Override=True)

        #The background color must be in uint8
        background_color = np.array(color_pallet[current_type],dtype=np.uint8)

        show_trajectories = False
        if show_trajectories:
            display_out = np.ones((height+header_size, width*2,3),dtype=np.uint8)
        else:
            display_out = np.ones((height+header_size, width,3),dtype=np.uint8)
        display_out[:,0:width,:] = background_color
        display_out[header_size:,0:width,:] = Image_I
        WriteHeaderInformation(display_out,ImageToDisplay,current_cell,region_names[current_type])
        return display_out

    def WriteHeaderInformation(DisplayOut,ImageToDisplay,CellNumber="N/A",RegionType="N/A"):
        nonlocal header_size,toggle_auto_record

        textToDisplay = 'Slide: {}, Cell Number: {}, Type: {}'.format(ImageToDisplay,CellNumber,RegionType)

        if toggle_auto_record:
            textToDisplay = textToDisplay + "----AutoRecording-----"


        cv2.putText(DisplayOut,text=textToDisplay,
                            org=(0,np.int32(header_size*0.7)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 0.7,color=(0,0,0),thickness=2,
                            lineType=cv2.LINE_AA)

    def DefineCellNumber(DisplayOut,X,Y,CellNumber): 
        coordinates = (X, Y)  # Replace X and Y with your coordinates

        # Set font type and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6

        # Set the color of the text and the line thickness
        color = (0,165,255)  
        thickness = 2

        # Apply the text on the image
        text = "Cell {}".format(CellNumber+1)
        cv2.putText(DisplayOut, text, coordinates, font, font_scale, color, thickness, cv2.LINE_AA)
    
    def DrawRectangles(DisplayOut,ImageToDisplay,RectangleInformation,ColorValue,Override = False,DisplayCellNumber=False):
        nonlocal current_cell, current_type,cell_type

        n_cells, n_frames, _ = RectangleInformation.shape
        for ith_cell in range(0,n_cells):
            xi,yi,xf,yf = RectangleInformation[ith_cell,ImageToDisplay,:]
            if DisplayCellNumber:
                DefineCellNumber(DisplayOut,xi,yi,ith_cell)
            if (xi>-1) and (yi>-1) and (xf>-1) and (yf>-1):

                #Settng Dividing cells to a different color:
                if cell_type[0,ith_cell] != 1 and Override == False :
                    ColorValueFinal = (0,128,255)
                else:
                    ColorValueFinal = ColorValue

                #Highlight the current cell with a shaded in region
                if (current_cell == ith_cell and current_type > -1):
                    overlay = DisplayOut.copy()
                    alpha = 0.3
                    cv2.rectangle(overlay,(xi,yi),(xf,yf),ColorValueFinal,-1)
                    cv2.addWeighted(overlay, alpha, DisplayOut, 1 - alpha,0, DisplayOut)
                else:
                    cv2.rectangle(DisplayOut,(xi,yi),(xf,yf),ColorValueFinal,1)


            else:
                break

    def SetXiYiToMin(xi,yi,xf,yf,standard_width=-1):
        '''
        The SetXiYiToMin function is to ensure that xi<xf and yi<yf. This can be broken if the square is  annotated backwards
        '''
        if standard_width > -1:
            xf = xi + standard_width
            yf = yi + standard_width
        else:
            xi,xf = np.sort([xi,xf])
            yi,yf = np.sort([yi,yf])
        return [xi,yi,xf,yf]

    def CalculateBackgroundInfo(Image,BackgroundCoordinates):
        xi,yi,xf,yf = BackgroundCoordinates
        background = Image[yi:yf,xi:xf]
        mean_background = background.mean()
        max_background = background.max()
        return mean_background,max_background

    #Record positions of mouse moves when clicked
    xi,yi,xf,yf = -1,-1,-1,-1
    mousedown = False
    mousemove = False
    def MouseCommands(event,x,y,flags,param):
        nonlocal xi,yi,xf,yf
        nonlocal x_record, y_record
        nonlocal current_img, ImageStack,header_size
        nonlocal mousedown,mousemove
        nonlocal stored_info, current_cell, current_type
        nonlocal WindowName, display_stack
        nonlocal background_set,standard_width
        nonlocal toggle_i
        if not findlength: 
            if event == cv2.EVENT_LBUTTONDOWN and not change_clippings:
                if y > header_size and x<width and not toggle_auto_record:
                    xi = x
                    yi = y - header_size
                    mousedown = True
                else:
                    xi,yi = -1,-1
            elif event == cv2.EVENT_MOUSEMOVE:
                if y > header_size and x<width:
                    print("X = {}, Y = {}, Intensity = {}".format(x,y-header_size,ImageStack[current_img,y-header_size,x]))
                    if not change_clippings:
                        if mousedown and not toggle_auto_record:
                            xf = x
                            yf = y - header_size
                            mousemove = True
                        elif toggle_auto_record: #Autorecord
                            if current_type >-1 and standard_width>-1:
                                x_record = x
                                y_record = y - header_size
                else:
                    if mousedown:
                        xf,yf = -1,-1
            elif event == cv2.EVENT_LBUTTONUP and not change_clippings:
                mousedown = False
                #If mose has moved or standard width is set, proceed to draw the square
                if (mousemove or (standard_width>-1) ) and y>header_size and x < width:
                    mousemove=False
                    if current_type >-1:
                        #If the cell annotation starts at a different location other than the first frame,
                        # this will fill in the other values
                        if stored_info[current_type,current_cell,0,:].sum() == -4:
                            stored_info[current_type,current_cell,:,:] = SetXiYiToMin(xi,yi,xf,yf,standard_width)
                        else:
                            stored_info[current_type,current_cell,current_img:,:] = SetXiYiToMin(xi,yi,xf,yf,standard_width)
                    elif current_type == -1:
                        #Same as above, this will ensure there are no incomplete sets of annotations.
                        background_set = True
                        if background[0,0,:].sum() == -4:
                            background[0,:,:] = SetXiYiToMin(xi,yi,xf,yf)
                        else:
                            background[0,current_img:,:] = SetXiYiToMin(xi,yi,xf,yf)
                    elif current_type == -2:
                        standard_width = int(np.mean([abs(yi-yf),abs(xi-xf)]))
                        print(standard_width,xi,xf,yi,yf)
                    else:
                        print("Something is wrong!!!!")

                    image_to_display = GetDisplayImage(display_stack,current_img)
                    cv2.imshow(WindowName,image_to_display)
                else:
                    xi,yi,xf,yf = -1,-1,-1,-1
        else: 
            if event == cv2.EVENT_LBUTTONDOWN: 
                if toggle_i == 0: 
                    xi = x
                    yi = y - header_size
                    toggle_i = 1 
                else: 
                    xf = x
                    yf = y - header_size
                    print(np.sqrt((xf-xi)**2 + (yf - yi)**2))
                    toggle_i = 0 


    #Processing the stack for displaying
    display_stack   = pi.ProcessingProcedure1(ImageStack)

    #Loading the display window and displaying the first image
    current_img = 0
    max_image = display_stack.shape[0]

    #Setting a standard width if necessary
    standard_width = -1

    #DataFrame of Points:
    #(Loc,Cells,Frame,[xi,xf,yi,yf])
    max_allowable_cells = 500
    stored_info = np.ones((2,max_allowable_cells,max_image,4),dtype=np.int32)*-1
    current_cell = 0
    current_type = -1 #0 = Nucleus/ 1 = Cytoplasm  / -1 = Background

    #Setting the background data storage
    background = np.ones((1,max_image,4),dtype=np.int32)*-1
    background_set = False

    #Creating a list that will represetnt the type of cell that is being tracked:
    # 1 - Non Dividing Cell throughout the imaging
    # 0 - Dividing Cell throughout the imaging
    cell_type = np.ones((1,max_allowable_cells),dtype = np.uint8)

    #Loading Previous masks if avaiable:
    if PrevMasks is not None and ZProject:
        background = PrevMasks['Background']
        background_set =  True

        NumberOfMasks = PrevMasks["Cells"].shape[1]
        stored_info[:,:NumberOfMasks,:,:] = PrevMasks["Cells"]
        cell_type[0,:NumberOfMasks] = PrevMasks["CellType"]

    #The window name is the description found in the key file
    cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL )

    image_to_display = GetDisplayImage(display_stack,current_img)
    cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
    cv2.imshow(WindowName,image_to_display)
    cv2.setMouseCallback(WindowName,MouseCommands)

    change_in_intensity = 5 #pixels

    #List of slides to pause 1 second on while recording is activated. 
    pause_on_slides = []
    pause_toggle = 0 #I need a pause toggle to get past the sleep function after the first time 
    tstart_pause = -1

    #Speed Annotate: 
    speed_annote = False

    playVideo = False
    timerIterations = 0
    while True:
        store_waitkey = cv2.waitKey(1)

        if store_waitkey == -1: 
            if playVideo: 
                if timerIterations <15: 
                    time.sleep(0.01)
                    timerIterations = timerIterations+1
                else:
                    timerIterations = 0; 
                    store_waitkey = ord('f')

        if store_waitkey & 0xFF == ord('q'):
            break
        elif (store_waitkey & 0xFF == ord('f')) and not change_clippings: #Right Arrow
            if toggle_auto_record and (current_img in pause_on_slides) and (pause_toggle == 0): 
                tstart_pause = time.time()
                pause_toggle = 1
            elif pause_toggle == 1: 
                dt = time.time() - tstart_pause 
                if dt > 1.0: 
                    pause_toggle = 2
            else: 
                pause_toggle = 0 
                if toggle_auto_record and x_record > -1 and y_record > -1:
                    stored_info[current_type,current_cell,current_img,:] = SetXiYiToMin(x_record,y_record,0,0,standard_width)

                current_img += 1
                if current_img>max_image-1:
                    if speed_annote: #Throw this function if you want to continue to annote one after another 
                        playVideo = False
                        time.sleep(1)
                        current_type = 1 - current_type
                        if current_type == 0: 
                            current_cell+=1 #Move to next cell after cyoplasm is set 
                    else: 
                        toggle_auto_record = False
                    x_record = -1
                    y_record = -1
                    current_img = 0

                image_to_display = GetDisplayImage(display_stack,current_img)
                cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('d')) and not change_clippings: #Left Arrow
            current_img -= 1
            if current_img<0:
                current_img = max_image-1
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('s')) and not change_clippings:
            #One can only go to the next cell if both the nucleus and cytoplasm regions were selected
            if not background_set:
                print("Background not set, cannot  continue")
            elif np.any(stored_info[:,current_cell,0,:]==-1):
                print("No Selection on current set. Please select both the nucleus and cytoplasm.")
            else:
                if current_cell == max_allowable_cells-1:
                    current_cell= 0
                else:
                    current_cell+=1

                if current_type == -1:
                    current_cell=0

                #Reset to first image and type when changing betwene cells
                current_type = 0
                current_img = 0

                image_to_display = GetDisplayImage(display_stack,current_img)
                cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('a')) and not change_clippings:
            if current_cell == 0:
                #Prevent returning to past cells if they are notdefined.
                if np.any(stored_info[:, max_allowable_cells-1,0,:]==-1):
                    current_cell = 0
                    print("Cannot go backwards because cell is not defined")
                else:
                    current_cell= max_allowable_cells-1
            else:
                current_cell-=1

            if current_type == -1:
                current_cell = 0

            #Only set current type to 0 if background has been set.
            if not background_set:
                print("Background not set, cannot  continue")
            else:
                current_type = 0
            current_img = 0
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif store_waitkey & 0xFF == ord(' ') and not change_clippings:
            if background_set:
                if current_type in [-1,-2] :
                    current_type = 0
                else:
                    current_type = 1 - current_type
            else:
                print("Background not set, cannot  continue")

            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif store_waitkey & 0xFF == ord('b') and not change_clippings:
            current_type = -1
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif store_waitkey & 0xFF == ord('w') and not change_clippings:
            #Set the standard width of squares
            current_type = -2
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('r')) and (current_type == -2) and not change_clippings:
            #Reset the standard width settign to manuel mode
            standard_width = -1
        elif (store_waitkey & 0xFF == ord('l')) and (current_type == -2) and not change_clippings:
            #Set standard width to size of first cell and current image
            xi,yi,xf,yf = stored_info[0,0,current_img,:]
            standard_width = int(np.mean([abs(yi-yf),abs(xi-xf)]))
        elif store_waitkey & 0xFF == ord('k') and not change_clippings:
            cell_type[0,current_cell] = 0 if cell_type[0,current_cell]==1 else 1
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif store_waitkey & 0xFF == ord('r') and not change_clippings: #Start Recording location with mouseloc
            toggle_auto_record = not toggle_auto_record
            if toggle_auto_record and standard_width >-1:
                print("Starting Record With mouse location.")
                print("Must have standard width set")
            elif standard_width == -1:
                print("Set Width of box")
                toggle_auto_record = False
            else:
                print("AutoRecord Stopped")
                speed_annote = False 
            current_img = 0
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif store_waitkey & 0xFF == ord('u') and toggle_auto_record: 
            speed_annote = not speed_annote
            print("Speed Annotation Starting")
        elif store_waitkey & 0xFF == ord('t'):
            '''
            If the change clippings toggle is on, then intensity thresholding
            changes take place instead of the usual arrow keys
            '''
            change_clippings = not change_clippings
            current_type = -3 if change_clippings else -1
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('f')) and change_clippings:
            max_clipping += change_in_intensity
            max_clipping = np.clip(max_clipping,0,255)
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('d')) and change_clippings:
            max_clipping -= change_in_intensity
            max_clipping = np.clip(max_clipping,0,255)
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('s')) and change_clippings:
            min_clipping += change_in_intensity
            min_clipping = np.clip(min_clipping,0,255)
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('a')) and change_clippings:
            min_clipping -= change_in_intensity
            min_clipping = np.clip(min_clipping,0,255)
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('r')) and change_clippings:
            '''
            Reset pixel range
            '''
            min_clipping = 0
            max_clipping = 255
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('e')) and change_clippings:
            change_in_intensity = 5 if change_in_intensity != 5 else 1
        elif (store_waitkey & 0xFF == ord('w')) and change_clippings:
            min_clipping = 0
            max_clipping = change_in_intensity
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
            print("Pixel Display Range",min_clipping,max_clipping)
        elif (store_waitkey & 0xFF == ord('c')) and change_clippings:
            #Toggle between cmap and grayscale
            cmapOption = not cmapOption
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('b')) and change_clippings:
            #Toggle the background correction display
            backgroundCorrection = not backgroundCorrection
            image_to_display = GetDisplayImage(display_stack,current_img)
            cv2.imshow(WindowName,image_to_display)
        elif (store_waitkey & 0xFF == ord('l')):
            findlength = not findlength
        if (store_waitkey & 0xFF == ord('p')):
            #pause_on_slides.append(current_img)
            #print("Slide Slected to be Paused On")
            playVideo = not playVideo
            print("Value Of Play Video",playVideo)



    cv2.destroyAllWindows()

    #Returning the annotations for it to be saved for later.
    if background.sum() < 0:
        return None
    else:
        Data2Return = {}
        Data2Return['Background'] = background

        #Only going to return a matrix where data was stored;
        filledIndex = np.all((stored_info[1,:,0,:]>-1),axis=1)

        Data2Return['Cells'] = stored_info[:,filledIndex,:,:]
        Data2Return['CellType'] = cell_type[:,filledIndex]

        return Data2Return

def AnnoteImagesAndSave(KeyInformation, LocOfOutput, LocOfData,Channel,WellLabel='',Point=0,AutomaticBackgroundCorrection=False,ZProject = True,SkipFrames=1):
    '''
    This is going to be a method to annote and save the images. Currently, one needs a location of the output however I plan to move this over to a sql or mongodb database.
    '''

    #Attempt to load an intial mask file. If it does not exists, an empty dictionary is created.
    masks = GetPreviousMasks(LocOfOutput)

    #Annote Masks:
    #Extract List of wells to annotate. If No well specified, all wells within the expierment will be succesffully displayed to be annotated
    if WellLabel == '':
        WellList = KeyInformation["Well"]
    else:
        if isinstance(WellLabel, list):
            WellList = WellLabel
        else:
            WellList = [WellLabel]

    #Looping through each well selected:
    for Loc in WellList:
        #If no specific points are selcted, all avaiable poitns are chosen.
        if Point == -1:
            num_of_points = KeyInformation[KeyInformation["Well"]==Loc]["NumberOfPoints"]
            point_list = range(num_of_points.values[0])
        else:
            if isinstance(Point,list):
                point_list = Point
            else:
                point_list = [Point]

        for ith_point in point_list:
            if Loc not in masks:
                masks[Loc] = dict()
            if ith_point not in masks[Loc]:
                masks[Loc][ith_point] = None

            #Using the AnnoteImage function to create and return the mask.
            masks[Loc][ith_point] = AnnotateImages(Loc,KeyInformation,LocOfData,Channel,ith_point, PrevMasks=masks[Loc][ith_point],AutomaticBackgroundCorrection=AutomaticBackgroundCorrection,ZProject = ZProject,SkipFrames=SkipFrames )
            locToSave = os.path.join(LocOfOutput,"Masks.npy")
            np.save(locToSave,masks)
    return masks

def GetPreviousMasks(LocOfOutput):
    #Check if a Masks.npy file exists in the output directory. If it exists then load, otherwise return an empty dictionary.
    try:
        locOfMasks = os.path.join(LocOfOutput,"Masks.npy")
        masks = np.load(locOfMasks,allow_pickle=True)
        return masks.item()
    except:
        return dict()

def CalculateRatio(Well,Point,Filter1,Filter2,Key,LocOfData,Mask,R_Correction=1,AutomaticBackgroundCorrection = False):

    #Loading Data From Source. Must know the key and location of data
    Data = li.ExtractByWell(Well,Key,LocOfData)

    #Extracting the specific channels
    Numerator   = li.GetImagesFromExtracted(Data,Filter1,Point,AutomaticBackgroundCorrection).copy().astype('float')
    Denominator = li.GetImagesFromExtracted(Data,Filter2,Point,AutomaticBackgroundCorrection).copy().astype('float')

    #Extracting Information about Masks:
    Background = Mask[Well][Point]["Background"].squeeze()

    #Standard processing procdure:
    def StandardRatioPreProcessing(SlideInput,Xi,Yi,Xf,Yf):
        backgroundregion = SlideInput[Yi:Yf,Xi:Xf]
        mean_background = backgroundregion.mean()
        max_background  = backgroundregion.max()

        clipping_value  = int(max_background - mean_background) + 1

        OutSlide  = SlideInput.copy()
        OutSlide = OutSlide - mean_background
        OutSlide[OutSlide<clipping_value] = 0
        return OutSlide


    #For each slide and channel, must process the slide before the ratio is taken, this includes:
    #   1.) Backgroun Subtraction based off of the mask (User Defined)
    #   2.) Clipping the values where any values less than the max background - mean background are set to zero
    for slide in range(Background.shape[0]):
        xi,yi,xf,yf = Background[slide,:]
        Numerator[slide,:,:] = StandardRatioPreProcessing(Numerator[slide,:,:],xi,yi,xf,yf)
        Denominator[slide,:,:] = StandardRatioPreProcessing(Denominator[slide,:,:],xi,yi,xf,yf)


    #If Zero/Zero division occros, the ratio is set ot 0:
    #MaskZeros = (Numerator ==0 ) & (Denominator ==0)
    MaskZeros = Denominator ==0

    Numerator[MaskZeros] = 0
    Denominator[MaskZeros] = 1

    Ratio = Numerator/Denominator
    MaxValue = np.max(Ratio[Ratio!=np.inf])
    Ratio[Ratio==np.inf] = 0
    Ratio = np.nan_to_num(Ratio)
    Ratio[Ratio>1E10] = 0

    #Line flips the signal so that an increase in signal corresponds to increasing fret. 
    Ratio = 1 - Ratio/R_Correction

    return Ratio

def PlotRatio(Well,Point,Filter1,Filter2,Key,LocOfData,Mask,R_Correction=1):
    '''
    Plotting the ratio from GetRatio
    '''
    Ratio = CalculateRatio(Well,Point,Filter1,Filter2,Key,LocOfData,Mask,R_Correction)
    window_name = Key[Key["Well"]==Well]["Title"].values[0] + "-----Ratio: {}/{}".format(Filter1,Filter2)+" /// Well:{} /// Point:{}".format(Well,Point)
    DisplayStack(Ratio,window_name,Mask[Well][Point])

#====================================================================
#                  Post Annotation Proccessing (PAP)
#====================================================================

def ExtractBackgroundInfo(Well,Key,LocOfOutput): 
    masks = GetPreviousMasks(LocOfOutput)[Well][0]
    return masks["Background"]

def ExtractMaskInfo(Well,Key,LocOfOutput): 
    masks = GetPreviousMasks(LocOfOutput)[Well][0]
    
    #Format as made in DisplayStack
    background = masks["Background"]
    nucleus    = masks["Cells"][0,:,:,:]
    cytoplasm  = masks["Cells"][1,:,:,:]
    cell_type  = masks["CellType"]
    
    return background, nucleus, cytoplasm, cell_type 

def GetBackground(Well,ImageStack,Key,LocOfOutput): 
    background_masks = ExtractBackgroundInfo(Well,Key,LocOfOutput)
    return CalculateBoxStatistics(ImageStack,background_masks)

def CalculateBoxStatistics(ImageStack,Masks): 
    n_frames          = ImageStack.shape[0]
    n_boxes_per_frame = Masks.shape[0]
    data_mean         = np.zeros((n_frames,n_boxes_per_frame))
    
    for nth_box in range(n_boxes_per_frame):
        box_n = Masks[nth_box] #Getting the list of boxes per frames
        for ith_frame in range(n_frames):
            xi,yi,xf,yf = box_n[ith_frame,:]
            frame_i = ImageStack[ith_frame,yi:yf,xi:xf]
            data_mean[ith_frame,nth_box] = frame_i.mean()
    return data_mean

def GetdtAndTimeOfTreatments(Well,Key): 
    dt = Key[Key["Well"]==Well]['dt'].values[0]
    time_of_treatment = Key[Key["Well"]==Well]["TimeOfTreatment"].values
    title = Key[Key["Well"]==Well]["Title"].values[0]

    return dt, time_of_treatment, title

def GetCytoplasmToNucleus(Well,Channel,Key,LocOfData,LocOfOutput,ShiftTrajectories = 0,Point=0,ZStack = False): 
    
    #Need to Load the masks already made 
    background, nucleus, cytoplasm, cell_type =  ExtractMaskInfo(Well,Key,LocOfOutput)
    
    #Need Load Image Stack and only the channel needed: 
    image_stack = li.GetOnlyOneChannel(Well,Channel, Key, LocOfData, Point = Point,ZProject = ZStack)
    
    #Calculate Background: 
    background_avg = CalculateBoxStatistics(image_stack,background)
    cytoplasm_avg   = CalculateBoxStatistics(image_stack,cytoplasm)
    nucleus_avg   = CalculateBoxStatistics(image_stack,nucleus)
    
    ratio = (cytoplasm_avg-background_avg)/(nucleus_avg-background_avg)
    
    #Uncomment if one wants to remove tossed trajectories by setting it a max 
    #ratio[:,cell_type[0,:]!=1] = ratio[:,cell_type[0,:]==1].max()

    ratio = ratio[:,cell_type[0,:]==1]

    #Shifting Trajectories by a constant amount 
    ratio = ratio + ShiftTrajectories

    return ratio

def PlotCytoplasmToNucleusTrajectories(Well,Channel,Key,LocOfData,LocOfOutput,ShiftTrajectories = 0,CorrectionType=-1,Point=0):
    #Get the Trajectories 
    trajectories = GetCytoplasmToNucleus(Well,Channel,Key,LocOfData,LocOfOutput,ShiftTrajectories=ShiftTrajectories,Point=Point)

    y_label = Channel+ ": Cytoplasm to Nucleus"

    #Call custom plotting script: 
    CustomPlotting_V1(trajectories,Well,Key,y_label,CorrectionType=CorrectionType)

def CustomPlotting_V1(X,Well,Key,y_label,CorrectionType=-1): 
    
    #Get the information to label axis: 
    dt, time_of_treatment,title = GetdtAndTimeOfTreatments(Well,Key)
    
    #Extract shape of object 
    [n_slides,n_cells]= X.shape

    time = range(n_slides)*dt

    plt.figure(figsize=(15,5))
    if CorrectionType == 1: 
        #Set baseline value equal to zero 
        X = X - X[0,:]

    
    plt.plot(time,X)
    plt.xlabel("Time [Minutes]",size=15)
    plt.ylabel(y_label,size=15)
    for ith_treatment_time in time_of_treatment:
        plt.axvline(x=ith_treatment_time-dt,color='k')
        
    plt.plot(time,np.mean(X,axis=1),color='b',linewidth=5,linestyle='--')
    plt.title(title,size=15)
    plt.grid()

def PlotFretTrajectories(Well,DonorChannel,RecieverChannel,Key,LocOfData,LocOfOutput,ShiftTrajectories = 0,R_Correction = 1,CorrectionType = -1,Point=0):
    #Get the Trajectories 
    trajectories = GetRatioTrajectories(Well,DonorChannel,RecieverChannel,Key,LocOfData,LocOfOutput,
                                        ShiftTrajectories = ShiftTrajectories,R_Correction=R_Correction,Point=Point)

    y_label = "Cytoplasm: {}/{}".format(DonorChannel,RecieverChannel)

    #Call custom plotting script: 
    CustomPlotting_V1(trajectories,Well,Key,y_label,CorrectionType=CorrectionType)
def GetRatioTrajectories(Well,DonorChannel,RecieverChannel,Key,LocOfData,LocOfOutput,ShiftTrajectories = 0,R_Correction = 1,Point=0): 
    mask = GetPreviousMasks(LocOfOutput)
    
    image_stack = CalculateRatio(Well,Point,DonorChannel,RecieverChannel,Key,LocOfData,mask,R_Correction)
    
    background, nucleus, cytoplasm, cell_type = ExtractMaskInfo(Well,Key,LocOfOutput)
    
    cytoplasm_avg   = CalculateBoxStatistics(image_stack,cytoplasm)

    
    
    #Uncomment if one wants to remove tossed trajectories by setting it a max 
    #cytoplasm_avg[:,cell_type[0,:]!=1] = cytoplasm_avg.max()
    
    ratio = cytoplasm_avg[:,cell_type[0,:]==1]
    
    #Shifting the trajectories by some specified amount given by user: 
    ratio =  ratio + ShiftTrajectories

    return ratio

def GetNucleusandCytoplasm(Well,Channel,Key,LocOfData,LocOfOutput,ShiftTrajectories = 0,Point=0,ZStack = False,BackgroundCorrectFirst=False,SkipFrames=1): 
    
    #Need to Load the masks already made 
    background, nucleus, cytoplasm, cell_type =  ExtractMaskInfo(Well,Key,LocOfOutput)
    
    #Determining the number of channels based off of the information in the key 
    channels = Key[Key["Well"]==Well].squeeze().Channels
    nChannels = len(channels.split("//"))

    #Need Load Image Stack and only the channel needed: 
    image_stack = li.GetOnlyOneChannel(Well,Channel, Key, LocOfData, Point = Point,ZProject = ZStack,CorrectShape=nChannels)
    image_stack = image_stack[::SkipFrames]

    
    #Calculate Background: 
    background_avg = CalculateBoxStatistics(image_stack,background)

    if BackgroundCorrectFirst: 
        image_stack = image_stack-background_avg[:,:,np.newaxis]
        print(image_stack.min())
        image_stack[image_stack<0] = 0 

    cytoplasm_avg   = CalculateBoxStatistics(image_stack,cytoplasm)
    nucleus_avg   = CalculateBoxStatistics(image_stack,nucleus)
    
    

    return nucleus_avg,cytoplasm_avg,background_avg