import os
from LCIA import load_images as li
from LCIA import extract_data as ed
from LCIA import plotting as pl
from LCIA import process_images as pi
from LCIA import auto_detection as ad
import pandas as pd
import numpy as np


def CheckExpectedVsFoundMasks(MasksDirContents,ExpectedMaskFiles,MovieFileNames): 
    '''
    Function to confirm that all of the expected masks are found. If they are
    not found then print statements will be produced saying which mask failed 
    to find its correspoinding file. 
    
    Parameters:
    ==========
        

    Returns None if no masks found else returns a list of found masks 
    '''
    #Check to ensure each movie has a corresponding mask: 
    check_masks = np.array([False if x in MasksDirContents else True for x in ExpectedMaskFiles ])

    if np.any(check_masks): 
        print("\n===============================================================")
        print("The Following Files did not find a corresponding Mask:")
        print("===============================================================")
        print("\nFiles with No Masks:")
        print(MovieFileNames[check_masks])
        print("\nExpected Mask File Names")
        print(ExpectedMaskFiles[check_masks])
        print("")
        print("===============================================================")
    else: 
        print("\n===============================================================")
        print("All Files found corresponding masks")
        print("===============================================================")

    if np.all(check_masks): 
        print("\n===============================================================")
        print("No files with corresponding masks found. Program terminating")
        print("===============================================================")
        return None

    return zip(ExpectedMaskFiles[~check_masks],MovieFileNames[~check_masks])

def AutoOptimizePaths(MasksStack,MinThreshold,MaxThreshold):
    best_score = 0 
    best_traj_found  = 0
    
    for ith_dist in range(MinThreshold,MaxThreshold): 
        traj = ad.ConnectTrajectoriesV2(MasksStack,ith_dist)
        statstra = ad.GetPathStatisticsDirect(traj)
        
        total_found    = statstra["TotalTrajectoriesFound"]
        complete_found = statstra["CompletePaths"]
        active_found   = statstra["ActivePathsEnd"]
        
        length      = statstra["LengthOfEachTrajectory"]
        avglength = 1 if length.size==0 else round(length.mean())
        
        cost_function = complete_found + total_found/avglength
        
        if cost_function>best_score: 
            best_score = cost_function
            best_traj_found = traj.copy()
            
            if False: 
                if statstra["TotalTrajectoriesFound"] == 0: 
                    print("Threshold: {}\n".format(ith_dist),"No Traj Found")
                else:
                    print("Threshold: {}\n".format(ith_dist),
                          "Total Trajecotries Found {} ///".format(total_found),
                          "Complete Paths Found: {} ///".format(complete_found),
                          "Active Paths in End: {} ///".format(active_found),
                          "Average Length: {}".format(avglength))
    return best_traj_found 

def ExtractStatisticsFromSingleMasks(Masks,ImageStack,Paths):
    n_slides,width,height = ImageStack.shape
    
    #Rescale Masks to ensure they are the same as input images
    rescaled_masks = ad.RescaleStack(Masks,width,height)
    
    #Connected Trajectories 
    avg_pixel_connected = np.zeros(Paths.shape)
    sum_of_region_connected = np.zeros(Paths.shape)
    
    #Unconnected Trajectories. 
    max_number_of_cells = Masks.max()
    avg_pixel_unconnected = np.zeros((n_slides,max_number_of_cells))
    sum_of_region_unconnected = np.zeros((n_slides,max_number_of_cells))
    
    for ith_slide in range(n_slides): 
        cells_of_interest = Paths[:,ith_slide]
        slide_of_interest = ImageStack[ith_slide,:,:]
        for ith,ith_cell in enumerate(cells_of_interest): 
            if ith_cell == 0: 
                #Incomplete trajectory found 
                avg_pixel_connected[ith,ith_slide] = np.nan
                sum_of_region_connected[ith,ith_slide] = np.nan
            else: 
                x,y = np.where(rescaled_masks[ith_slide,:,:]==ith_cell)
                avg_pixel_connected[ith,ith_slide] = slide_of_interest[x,y].mean()
                sum_of_region_connected[ith,ith_slide] = slide_of_interest[x,y].sum()
                if not np.any(x): 
                    print(np.unique(rescaled_masks[ith_slide,:,:]))
                    print("Cell",ith_cell,"Slide",ith_slide,'Traj',ith)
        
        #Getting the pixel intensity for every mask available 
        for ith_mask in range(int(rescaled_masks[ith_slide,:,:].max())): 
            x,y = np.where(rescaled_masks[ith_slide,:,:]==ith_mask+1)
            avg_pixel_unconnected[ith_slide,ith_mask] = slide_of_interest[x,y].mean()
            sum_of_region_unconnected[ith_slide,ith_mask] = slide_of_interest[x,y].sum()
        
    avg_pixel_unconnected[avg_pixel_unconnected == 0] = np.nan
    sum_of_region_unconnected[sum_of_region_unconnected == 0] = np.nan
    return {"API Connected": avg_pixel_connected.transpose(),
            "SP Connected": sum_of_region_connected.transpose(),
            "API UnConnected": avg_pixel_unconnected,
            "SP UnConnected": sum_of_region_unconnected}

def AddHeaderAndSave(Data2Save,Background,BackgroundLoc,Paths,ZStackCombineMethod,NameOfCSV,NameOfEachColumn):    
    with pd.ExcelWriter(NameOfCSV) as writer:  
        temp = pd.DataFrame([ZStackCombineMethod],index=["Method To Combine ZStack"])
        temp.to_excel(writer,"Information")
        for ith_statistic in Data2Save.keys():
            ith_data = Data2Save[ith_statistic]
            num_columns = ith_data.shape[1]

            header = np.array([NameOfEachColumn+" " + str(ith) for ith in range(num_columns)],dtype='str')
            temp= pd.DataFrame(ith_data,columns=header)

            temp.to_excel(writer,ith_statistic)
        
        for values,tabname,colname in zip([Background,Paths],["Background","Paths"],["Background",NameOfEachColumn]):
            num_columns = values.shape[1]
            header = np.array([colname+" " + str(ith) for ith in range(num_columns)],dtype='str')
            temp= pd.DataFrame(values,columns=header)

            temp.to_excel(writer,tabname)
        

       
        temp= pd.DataFrame(BackgroundLoc.squeeze(),columns=["X_i","Y_i","X_f","X_I"])

        temp.to_excel(writer,"Background Locations")
        

def ConnectAndEstimateData(MaskStack,ImageStack,OutputSaveLoc,BackgroundLoc,ZStackCombineMethod,MinThreshold=20,MaxThreshold=25): 
    paths = AutoOptimizePaths(MaskStack,MinThreshold,MaxThreshold)

    background = ed.CalculateBoxStatistics(ImageStack,BackgroundLoc)
    data = ExtractStatisticsFromSingleMasks(MaskStack,ImageStack,paths)
    
    AddHeaderAndSave(data,background,BackgroundLoc,paths.transpose(),ZStackCombineMethod,OutputSaveLoc,"Cell")


def ExtractDataFromDirectory(LocOfData,MaskExtension,LocOfMasks=None):

    #If Mask Loction not set, assume in same directory as LocOfData
    if LocOfMasks == None:  
        LocOfMasks = LocOfData
    
    #List of Movies to Analyze: 
    data_dir = os.listdir(LocOfData)
    masks_dir = os.listdir(LocOfMasks)

    #Load Background location for all movies and timepoints: 
    background_loc = LoadPrevMasksDir(LocOfData)
    
    #All movies that are within the directory with a .dv extension and does not 
    #contain the mask extension are assumed to movies to be extracted
    movie_filenames = [x for x in data_dir if ((".dv" in x) and MaskExtension not in x)]
    movie_filenames = np.array(movie_filenames)

    #Assume Masks are already calculated and are within the LocOfMask directory 
    #with the same names except the user defined function: 
    mask_filenames = [x.split('.')[0]+MaskExtension for x in movie_filenames]
    mask_filenames = np.array(mask_filenames)

    #Check expected and found mask files and only return files with both mask 
    #and image 
    files_2_process = CheckExpectedVsFoundMasks(masks_dir,mask_filenames,movie_filenames)
    #Terminate program early if no mask file found: 
    if files_2_process is None: 
        return 

    ZStackCombineMethod = "sum"
    for ith_maskstack,ith_datastack in files_2_process: 
        datastack = li.LoadZStackImage(os.path.join(LocOfData,ith_datastack),ZStackCombineMethod )
        maskstack = li.ExtractTiffStack(os.path.join(LocOfMasks,ith_maskstack))

        datastackname = ith_datastack.split(".")[0]
        loc_of_output = os.path.join(LocOfData,datastackname+".xlsx")
        background_loc_specific = background_loc[datastackname]["Background"]
        ConnectAndEstimateData(maskstack,datastack,loc_of_output,background_loc_specific,ZStackCombineMethod,MinThreshold=20,MaxThreshold=21)

    
def LoadPrevMasksDir(Directory):
    full_path_prev_mask = os.path.join(Directory,"BackgroundMasks.npy")
    try:
        backgroundmasks = np.load(full_path_prev_mask,allow_pickle=True).item()
    except: 
        backgroundmasks = {}
    return backgroundmasks
    
def DefineAndSaveBackground(Directory,Save=False): 
    #All movies that are within the directory with a .dv extension and does not 
    #contain the mask extension are assumed to movies to be extracted
    movie_filenames = [x for x in os.listdir(Directory) if (".dv" in x)]
    movie_filenames = np.array(movie_filenames)

    backgroundmasks = LoadPrevMasksDir(Directory)

    for ith_movie in movie_filenames: 
        full_path = os.path.join(Directory,ith_movie)
        name  = ith_movie.split(".")[0]

        try: 
            prev_mask = backgroundmasks[name]
        except: 
            prev_mask = None

        background = ed.DisplayStack(li.LoadZStackImage(full_path,"sum"),WindowName = name,PrevMasks=prev_mask)
        backgroundmasks[name] = background
    
    if Save: 
        np.save(os.path.join(Directory,"BackgroundMasks.npy"),backgroundmasks)
    return None

