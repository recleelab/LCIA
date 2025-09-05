import pandas  as pd
import numpy as np
#import mrc
#from PIL import Image
from LCIA import process_images as pi
import os

def LoadImages(LocOfDVFile,NumOfChannels=None,NumOfPoints = None,ZStack = False):
    """
    This function will load a set of images from the .dv format when only one point per files is chosen. If more than one point was used using plate scan, then the number of channels and points must be inputted in the optional arguments.
    Input:
        - LocOfDVFile = <Location of .dv file needed to be loaded>
        - NumOfChannels = Number of channels performed in expierments
        - NumOfPoints = <int> Expected number of points when multiple locations per .dv file are recorded

    Returns an mrc object that can be sliced in a simialar manner as a numpy array. The format of the output is:
        - (<Number of Frames>, <Number of Points>, <Number of Channels>, <Pixel Values Width>,<Pixel Values Height>)
    """
    try: 
        loaded_images = mrc.imread(LocOfDVFile)
        if len(loaded_images.shape) == 3: 
            x,y,z = loaded_images.shape 
            loaded_images = loaded_images.reshape(x//NumOfChannels,NumOfChannels,y,z)
        if len(loaded_images.shape) == 5:
            x,y,z,w,h = loaded_images.shape
            shift = 1 
        else: 
            y = 1
            shift = 0

        if ((y>1) and (NumOfPoints<2)) | ZStack: 
            if not ZStack: 
                print("Image not specified for Z-stack but detected!")
                
            print("Max Projection of Z-Stack Taken")
            loaded_images = loaded_images.max(1,keepdims=True)
    except: 
        if NumOfPoints == 1: 
            '''
            This only works if one point is observed
            '''
            tiff_images = ExtractTiffStack(LocOfDVFile)
            loaded_images = np.zeros((tiff_images.shape[0]//NumOfChannels,NumOfPoints,NumOfChannels,tiff_images.shape[1],tiff_images.shape[2]))
            time = 0 
            channel = 0 
            for i in range(tiff_images.shape[0]): 
                loaded_images[time,0,channel,:,:] = tiff_images[i,:,:]
                channel +=1 
                if channel >= NumOfChannels: 
                    channel = 0 
                    time +=1 
                
            print("----------")
            print("Loaded data through untested code. Use results with caution!")
            print("----------")
        else: 
            print("Failure to load data")

    if (NumOfChannels and NumOfPoints) is not None:
        #If the number of channels and number of points is specified, reshaping the output so the
        #matrix has the shape of [<Slide Number>, <PointLoc>,<Channel>,<WidthPixel>,<HeightPixels> ]
        num_of_frames       = loaded_images.shape[0]
        width_pixel_length  = loaded_images.shape[-1]
        height_pixel_length = loaded_images.shape[-2]
        loaded_images = loaded_images.reshape(num_of_frames,int(NumOfPoints),int(NumOfChannels), width_pixel_length,height_pixel_length)
        return loaded_images
    elif loaded_images.shape[1] > 1:
        #If the loaded images  contains more than one point, then LoadImagesWithMultiplePoints
        #function should be used instead where an additional input of the number of channels is
        #required
        raise ValueError('Error: Multiple Points Detected, must set expected number of channels and points')
    else:
        return loaded_images

def OpenTiffStack(LocOfTiff,Filters=None): 
    opened_image = Image.open(LocOfTiff)
    
    opened_image.seek(0)
    ith_frame = np.array(opened_image)
    
    #Getting Dimensions of Stack
    x,y = opened_image.size
    n_frames = opened_image.n_frames
    
    
    output = np.zeros((n_frames,x,y),dtype=ith_frame.dtype)
    output[0,:,:] = ith_frame
    for i in range(1,n_frames): 
        opened_image.seek(i)
        output[i,:,:] = np.array(opened_image)
    
    
    if Filters != None: 
        list_of_filters = Filters.split("//")
        n_filters = len(list_of_filters)
    else: 
        n_filters = 1
    
    output = output.reshape((int(n_frames/n_filters),1,n_filters,x,y),order='f')
    output = output.astype('float64')
    return output
def ExtractByWell(WellLabel, KeyInformation, LocOfData,ZStack = False):
    """
    Extracting the data by well label. This will require a KeyInformation csv file to be inputted that contains key information on the location of the files and the name of the file.
    Input:
        - WellLabel = Formatted as a string and must match exactly as the well is defined in the KeyInformation section
        - KeyInformation = CSV loaded into python that contains necessary information to load the files
        - LocOfData = Location of the folder where the data is located
    """
    CurrentRow = KeyInformation[KeyInformation["Well"] == WellLabel]

    if CurrentRow.shape[0 ]>1:
        raise ValueError("Multiple inputs with label found. Recheck KeyInformation for incorrect inputs")
    elif CurrentRow.shape[0] <1:
        print("No Wells with name {} found".format(WellLabel))
        return None

    CurrentRow = CurrentRow.squeeze()
    #Get Data
    data = ExtractDataByKeyRow(CurrentRow,LocOfData,ZStack)

    out = {}
    out["Data"] = data
    out["Information"] =  CurrentRow
    return out



def ExtractDataByKeyRow(CurrentRow,LocOfData,ZStack = False):
    """
    Extract the data by inputting a single row as a pandas data series with the required columns.
    """
    #Setting current row as a series
    CurrentRow = CurrentRow.squeeze()

    #Extracting the filter set and the number of filters used
    filter_set = CurrentRow.Channels.split("//")
    num_of_channels = len(filter_set)

    #Extracting the Number of points per file location
    num_of_points = CurrentRow.NumberOfPoints

    #Full Path to the location of the image
    loc_of_images = LocOfData+ CurrentRow.File

    #Extraction of the data
    data_type = CurrentRow.File.split(".")[-1]
    if data_type =="dv":
        data = LoadImages(loc_of_images,num_of_channels,num_of_points,ZStack)
    elif (data_type=="tif") | (data_type=="tiff") :
        data = OpenTiffStack(loc_of_images,CurrentRow.Channels)
    else: 
        print("ERROR, undefined image type")
        return 0 


    return data

def GetImagesFromExtracted(Data,Channel,Point=0,AutomaticBackgroundCorrection = False):
    '''
    Once  the images have been extracted, further slicing may occur to get a specific channel and point set. Default value for the point is 1.
    '''
    image_stack = Data["Data"]
    key_information = Data["Information"]

    filter_set = key_information.Channels.split("//")
    index_of_fiter = filter_set.index(Channel)
    x = image_stack[:,Point,index_of_fiter,:,:]
    return x

def GetOnlyOneChannel(WellLabel,Channel, KeyInformation, LocOfData, Point = 0,AutomaticBackgroundCorrection = False,ZProject = True,BioFormats=True,KillJavaBridge=False,CorrectShape=0):
    '''
    When only one channel is needed to be extracted 
    '''
    
    return LoadByAICSImageIO(WellLabel=WellLabel,
                      Channel=Channel,
                      KeyInformation=KeyInformation,
                      LocOfData=LocOfData,CorrectShape=CorrectShape)
    
    if BioFormats: 
        fileName,indexOfChannel = ExtractImportantInfo(WellLabel,Channel, KeyInformation)
        fullFilePath = os.path.join(LocOfData,fileName)
        image_stack = LoadBioFormats(fullFilePath,indexOfChannel,ZProject,KillJavaBridge)

    else: 
        Data = ExtractByWell(WellLabel,KeyInformation,LocOfData,ZProject)
        image_stack   = GetImagesFromExtracted(Data,Channel,Point)
    
    if AutomaticBackgroundCorrection:
        image_stack,background = pi.ProcessStack4Segmenation(image_stack,0,95)
        print("Automatic Background Correction Applied!")

    return image_stack

def LoadZStackImage(LocOfDVFile,CombineZStack = None):
    '''
    Load a *.dv file that contains a single channel with z-stack images

    Parameters
    ----------
    LocOfDVFile : str
        Relative or Absolute path to a *.dv file 
    
    CombineZStack: str, default = None
        Specifies how to return the images in a z-stack. The default behavior is 
        to return the z-stack however the shape of the stack can be reduced by 
        summing, averaging, taking the min or max of the z-stack. 

    Returns
    -------
    imagestack: a numpy array, shape (n_frames,n_z-stacks,n_width, n_height) 
        Returns images in a ndarray_inMrcFile format which can be treated as 
        a numpy array 
    
    Notes:
    ------
    This is a standard processing procedure only used for user visualization 
    '''
    
    imagestack = mrc.imread(LocOfDVFile)
    
    if CombineZStack==None: 
        return imagestack
    elif CombineZStack.lower() == "sum": 
        return imagestack.sum(axis=1)
    elif CombineZStack.lower() == "avg": 
        return imagestack.mean(axis=1)
    elif CombineZStack.lower() == "min": 
        return imagestack.min(axis=1)
    elif CombineZStack.lower() == "max": 
        return imagestack.max(axis=1)
    else: 
        raise NameError("ERROR: CombineZStack option unknown. \n The input can be one of the following: \n sum  \n avg \n min \n max \n None")

def ExtractTiffStack(LocTiffStack): 
    '''
    Load a tiff file with multiple images. 
    
    Parameters: 
    -----------
    LocTiffStack : str 
        Relative or absolute path to the tiff file 
        
    Returns: 
    --------
    imagestack: a numpy array, shape (n_frames, n_width, n_height )
        Images in a numpy array. 
    '''
    I = Image.open(LocTiffStack) 
    
    ith_image = 0 
    
    #Looping through all files. Unclear how many frames 
    data_2_return = []
    while ith_image >=0: 
        
        try: 
            I.seek(ith_image)
            data_2_return.append(np.array(I))
        except: 
            ith_image = -1000
        ith_image+=1
    return np.array(data_2_return)

def LoadBioFormats(LocOfData,IndexOfChannel,ZProject=False,KillJavaBridge=False,KeepDims=False):
    import javabridge
    import bioformats

    javabridge.start_vm(class_path=bioformats.JARS,run_headless=True)


    fileType = os.path.basename(LocOfData).split('.')[-1]
    if fileType == 'vsi': 
        seriesNumber = IndexOfChannel
        channel = 0 
    else: 
        seriesNumber = 0 
        channel = IndexOfChannel

    #Had to look at the following YouTube Video to get the 
    #format to load images: 
    # https://www.youtube.com/watch?v=DY3Hbd6Os4c
    reader = bioformats.get_image_reader(bioformats.ImageReader,path=LocOfData)

    #Extracting Information on Images 
    imageCount = reader.rdr.getImageCount()
    nChannels  = reader.rdr.getSizeC()
    nWidth     = reader.rdr.getSizeX()
    nHeight    = reader.rdr.getSizeY()
    nDepth     = reader.rdr.getSizeZ()
    nTime      = reader.rdr.getSizeT()
    
    load_image = np.zeros((imageCount,nWidth,nHeight),dtype=np.uint16)

    for ithPlane in range(imageCount): 
        load_image[ithPlane,:,:] = reader.read(series=seriesNumber,index=ithPlane,rescale=False)
    
    if KillJavaBridge:
        #The java bridge must  be killed manually when viewing stacks using openCV
        #Otherwise when loading stacks in Jupter  notebook, javabridge must remain 
        #open. Its a known bug that developers know about but have not fixed. 
        #javabridge.kill_vm()
        pass

    #Reshaping format 
    load_image = load_image.reshape((nTime,nDepth,nChannels,nWidth,nHeight))

    #Extracting Single Channel: 
    load_image = load_image[:,:,channel,:,:]

    #Take a projection over the z-axis if necessary: 
    if ZProject: 
        #Taking the max value from the projection 
        load_image = load_image.max(1)
    elif KeepDims: 
        pass 
    else: 
        #Stacking the z-projection so they are right after one another
        load_image = load_image.reshape((nTime*nDepth,nWidth,nHeight))

    load_image = np.flip(load_image,axis=1)

    return load_image

def ExtractImportantInfo(WellLabel,Channel, KeyInformation): 
    '''
    Function that extracts important information from the key
    '''

    CurrentRow = KeyInformation[KeyInformation["Well"] == WellLabel]

    if CurrentRow.shape[0 ]>1:
        raise ValueError("Multiple inputs with label found. Recheck KeyInformation for incorrect inputs")
    elif CurrentRow.shape[0] <1:
        print("No Wells with name {} found".format(WellLabel))
        return None

    CurrentRow = CurrentRow.squeeze()
    
    
    #List of Channels
    channelList = np.array(CurrentRow['Channels'].split("//"))
    
    #Find Index of Channel: 
    indexLoc, = np.where(channelList==Channel)
    print(indexLoc)
    if indexLoc.size==0: 
        print("=================================")
        print("Index Not found")
        print("Avaiable Channels")
        print(channelList)
        print("=================================")
        
        return None
    
    
    indexLoc = indexLoc[0]
    
    return CurrentRow['File'],indexLoc


def ReshapeResults(InputStack,NumberOfChannels): 
    t,c,z,x,y = InputStack.shape
    return InputStack.reshape(t//NumberOfChannels,NumberOfChannels,z,x,y)


def LoadByAICSImageIO(WellLabel,Channel,KeyInformation,LocOfData,CorrectShape=0): 
    from aicsimageio import AICSImage
    
    #Loading Full File name and index 
    fileName,indexOfChannel = ExtractImportantInfo(WellLabel,Channel, KeyInformation)
    
    fullFilePath = os.path.join(LocOfData,fileName)
    #Loading Image to memory
    img = AICSImage(fullFilePath).data
    
    if( CorrectShape !=img.shape[1]) and CorrectShape>0 : 
        img = ReshapeResults(img,CorrectShape)
        
    return img[:,indexOfChannel,:,:,:].squeeze()


def LoadByAICSImageIOALL(WellLabel,KeyInformation,LocOfData,CorrectShape=0): 
    from aicsimageio import AICSImage
    
    #Loading Full File name and index 
    CurrentRow = KeyInformation[KeyInformation["Well"] == WellLabel]
    CurrentRow = CurrentRow.squeeze()
    fileName = CurrentRow['File']
    
    fullFilePath = os.path.join(LocOfData,fileName)
    #Loading Image to memory
    img = AICSImage(fullFilePath).data
    
    if( CorrectShape !=img.shape[1]) and CorrectShape>0 : 
        img = ReshapeResults(img,CorrectShape)
        
    return img.squeeze(),CurrentRow['Channels']