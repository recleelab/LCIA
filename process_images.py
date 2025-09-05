import numpy as np
import matplotlib.pyplot as plt

def EnhanceContrast(X,top = None, bottom=None):
    '''
    Enhance Contrast of an single image 

    EnhanceContrast by default only rescales the image to an 8 bit. Optional
    arguments can clip the image. 
    
    Parameters
    ----------
    X : a numpy array, shape (n_width, n_height)
        Represents an image taken at any scale intensities 
    
    top : int, default=None
        Represents the maximum pixel intensity of the image that should serve as 
        the maximum. 

    bottom : int, default = None 
        Represents the minimum pixel intensity of the image that should serve as 
        the minimum. 

    Returns
    -------
    normalized_image : numpy array, shape (n_width, n_height)
        The processed image where the imaged rescaled such that the minimum 
        value is 0 and maximum value is 255. 

    Notes:
    ------
    Enhanced contrast is only used for visualization purposes and not used for
    any trajectory analysis where raw values are used. 
    '''

    #Normalize range:
    X = X.astype('float')

    top = np.percentile(X,99)
    bottom = np.percentile(X,1)

    Normalized_Range = (X - bottom)/(top - bottom)
    ClipAndMake8Bit = np.uint8(np.clip(Normalized_Range,0,1)*255)
    return ClipAndMake8Bit

def ProcessingProcedure1(DataToProcess,top=None):
    '''
    Processing Procedure 1 will accept a stack of images as an numpy array. For  and 
    each image, the EnhanceContrast function will be used to remove the top and 
    bottom 5% of the pixels and to rescale the data in 8bit images. 

    Parameters
    ----------
    X : a numpy array, shape (n_frames,n_width, n_height) or (n_width,n_height)
        Represents an set of images taken at any scale intensities. A single 
        image or a stack of images can be inputted. 

    Returns
    -------
    normalized_stack: a numpy array, shape (n_frames,n_width, n_height) or 
                      (n_width,n_height)
        Returns the normalized image stack based on a per frame basis. The 
        output shape will match the same shape as X. 
    
    Notes:
    ------
    This is a standard processing procedure only used for user visualization 
    '''

    #DataToProcess is expected to be in the form of n_timepoints X n_width X  
    #n_height , where n_timepoints is the number of images needed to be 
    #processed. If (n_width X n_height) array is given, dimensions are expanded 
    #in order for the code to function. 
    if len(DataToProcess.shape)==2: 
        DataToProcess = np.expand_dims(DataToProcess,0)

    #Standard Processing Pipeline
    processed_data = np.zeros(DataToProcess.shape,dtype=np.uint8)

    for ith_slice  in range(DataToProcess.shape[0]):
        slice_i = DataToProcess[ith_slice,:,:].copy()
        processed_data[ith_slice,:,:] = EnhanceContrast(slice_i,top)
    return processed_data.squeeze()

def StandardProcessingProcedure(Data,Point=None,Channel = None):
    '''
    Function is not used. Will eventually be deleted if no use is found in the future. 
    Date of Comment: 02/22/2021
    
    The code will process all of the data. This can take time, therefore a Point and channel can also  be provided.
    '''
    print("=========================================================")
    print("=========================================================")
    print("=========================================================")
    print("\n\n\n\n Function will be deleted in a future version. \n\n\n")
    print("=========================================================")
    print("=========================================================")
    print("=========================================================")
    Images = Data["Data"]
    RowInformation = Data["Information"]

    if (Point and Channel) is not None:
        filter_set = RowInformation.Channels.split("//")
        Channel_Index = filter_set.index(Channel)
        if Point == -1:
            Points = range(Images.shape[1])
            DataToProcess = Images[:,Points,Channel_Index,:,:]
            ProcessedData = np.zeros(DataToProcess.shape,dtype=np.uint8)
            for ith_point in Points:
                ProcessedData[:,ith_point,:,:] = ProcessingProcedure1(DataToProcess[:,ith_point,:,:])
            return ProcessedData
        else:
            DataToProcess = Images[:,Point,Channel_Index,:,:]
            return ProcessingProcedure1(DataToProcess)
    else:
        print("No Specific points and/or channel to plot given. Processing entire file.")
        print("Be patient! This will take computer memory and time. Kill process if limited on memory")
        #Code Useful for extracting all of the data at once.
        #Make sure this is uint8,  it is important for memory managment
        ProcessedOut = np.zeros(Images.shape,dtype=np.uint8)
        #Number of Channels:
        NumChannels = len(RowInformation.Channels.split("//"))
        for ith_point in range(RowInformation.NumberOfPoints):
            for ith_filter in range(NumChannels):
                print(Images[:,ith_point,ith_filter,:,:].shape)
                ProcessedOut[:,ith_point,ith_filter,:,:] = ProcessingProcedure1(Images[:,ith_point,ith_filter,:,:])
        return ProcessedOut

def ClipImage(InputSlice,LowerPercentile,UpperPercentile):
    x = InputSlice.copy()
    
    bottom,top = np.percentile(x,[LowerPercentile,UpperPercentile])
    
    x = np.clip(x,bottom,top)
    
    return x

def FindMean(InputSlice): 
    x = np.array((InputSlice!=InputSlice.max()) & (InputSlice!=InputSlice.min()))
    
    return np.mean(InputSlice)

def SubtractBackground(InputSlice,Background=None):
    if Background==None: 
        Background = np.percentile(InputSlice,10)
    x = InputSlice - Background
    x[x<=0] = 1
    return x
    
def Enhance4Segmentation(InputSlice,LowerPercentile,UpperPercentile): 
    clippedImage =  ClipImage(InputSlice,LowerPercentile,UpperPercentile)
    mean = FindMean(clippedImage)
    
    subtractedBackground = SubtractBackground(clippedImage,mean)
    
    return subtractedBackground,mean


def ProcessStack4Segmenation(Stack,LowerPercentile,UpperPercentile): 
    storedImages = np.zeros_like(Stack)
    storedMeans = np.zeros(Stack.shape[0])
    
    for ith,ithImage in enumerate(Stack): 
        storedImages[ith],storedMeans[ith] =  Enhance4Segmentation(ithImage,LowerPercentile,UpperPercentile)
    
    return storedImages,storedMeans
