import pandas as pd 
import matplotlib.pyplot as plt 
import os 

from LCIA import load_images as li
from LCIA import extract_data as ed 
from LCIA import auto_detection as ad
from LCIA import ml_tools as ml

import numpy as np


def Plot_APIPF(DataToPlot,ShowTitle=False,ShowImage=True,SetYAxis=[np.nan,np.nan]): 
    """
    Provide the standard plotting for the average pixel intensity per frame. 
    Input: 
        - Output from the APIPF function 
    Output: 
        - Average pixel intensity per frame per point 
    """

    DataToPlot.plot(figsize=(10,10))
    plt.grid()
    plt.xlabel("Frame Number",fontsize=20)
    plt.ylabel("Average Pixel Intensity",fontsize=20)
    if not np.any(np.isnan(SetYAxis)):
        plt.ylim(SetYAxis)
    if ShowTitle: 
        plt.title(ShowTitle,fontsize=20)
    if ShowImage:
        plt.show()


def plotAllAvgPixelIntensities(filter_to_display,key,folder_name,LocOfOutput,LocOfData):
    '''
    Plotting and saving all average pixel intensities per frame. The y-axis is set to the 
    bottom 5 percent and top 95 percent of the pixel intensities to get a good idea on how
    much flucuation is occuring.
    '''
    #Number of Wells: 
    max_files = key.shape[0]

    try: 
        os.makedirs(os.path.join(LocOfOutput,folder_name))
    except: 
        print("Folder Already Exists")

    for ith_file in range(max_files): 
        current_information = key.iloc[ith_file,:]

        filter_set = current_information.Channels.split("//")
        index_of_fiter = filter_set.index(filter_to_display)

        WellNumber = current_information["Well"]
        #Extract Data By Well Number: 
        data = li.ExtractByWell(WellNumber,key,LocOfData)

        average_pixel_value_per_frame,min_data,max_data = ed.APIPF(data["Data"],index_of_fiter)

        title = data["Information"].Title + "\n Filter: " + filter_to_display 
        Plot_APIPF(average_pixel_value_per_frame,ShowTitle=title,ShowImage=False,SetYAxis=[min_data,max_data])
        save_prefix = current_information.SavePrefix + "_"+filter_to_display+".png"
        figure_save_loc = os.path.join(LocOfOutput,folder_name,save_prefix)
        plt.savefig(figure_save_loc)
        plt.close()

def PlotAvgAcrossAllConditions(FunctionType,Key,Channel,LocOfOutput,Offset=-1,CorrectLinearShift = False):
    unique_conditions = Key["Title"].unique()
    
    #Finding the time unit: 
    time_unit = list(Key["Time Unit"].unique())
    if len(time_unit)==1: 
        time_unit = time_unit[0]
    else: 
        time_unit="Ambiguous"
    
    plt.figure(figsize=(15,5))
    
    
    for ith_condition in unique_conditions: 
        well_label_list = list(Key[Key["Title"]==ith_condition]["Well"])
        dt, time_of_treatment, title = ed.GetdtAndTimeOfTreatments(well_label_list[0],Key)
        average=ad.GetAverageAcrossPoints(FunctionType,well_label_list,Channel,LocOfOutput,CorrectLinearShift)
        #Offsetting the averages to start from same loc: 
        if Offset!=-1:
            average = average - average[int(Offset/dt)]

        x_axis = np.arange(0,dt*average.shape[0],dt)
        plt.plot(x_axis,average,label=ith_condition)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("Time [{}]".format(time_unit),size=15)
    #plt.ylabel(,size=15)
    plt.ylabel(ReturnYAxisLabel(FunctionType),size=15)
    plt.grid()
    
    title = "Average Values Across All Points in Condition// Channel:{}".format(Channel)
    if CorrectLinearShift: 
        title+="   ///   Linearly Corrected"
    plt.title(title)
    return plt.gca

def GetPlottingInformationFromKey(KeyInformation,WellList):
    '''
    Function that takes in the key information file and a list of wells to return 
    the common title, dt, and time unit. If more than one title, dt, and time unit is 
    found, the function will return None.
    '''
    focused_points = KeyInformation[KeyInformation["Well"].isin(WellList)]
    
    title = focused_points["Title"].unique()
    dt = focused_points["dt"].unique()
    timeunit = focused_points["Time Unit"].unique()
    
    def ReturnAlertIfMoreThanOneUniqueVariableFound(Input): 
        if len(Input)>1:  
            print("More than one input found for list of wells. Therefore, not consistent conditions")
            return 1  
        elif len(Input)<1: 
            print("Something went wrong, no input found. Check key information file")
            return 1  
        return 0
    
    #Counting errors if there is more than one unique title, dt, and timeunit for a set of wells 
    error_count = 0 
    error_count+=ReturnAlertIfMoreThanOneUniqueVariableFound(title)
    error_count+=ReturnAlertIfMoreThanOneUniqueVariableFound(dt)
    error_count+=ReturnAlertIfMoreThanOneUniqueVariableFound(timeunit)
    
    if error_count == 0: 
        return title[0],dt[0],timeunit[0]
    else: 
        return None
    
def AddLabelsToGraph(Title,Channel,FunctionType,TimeUnit,LinearlyCorrected): 
    Title = Title+"          Channel:"+Channel
    if LinearlyCorrected: 
        Title = Title+"          Linearly Corrected"
    plt.title(Title,size=15)
    plt.xlabel("Time [{}]".format(TimeUnit),size=15)
    plt.grid()
    plt.ylabel(ReturnYAxisLabel(FunctionType),size=15)

def PlotAllTrajectories(FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift = False): 
    #Getting the trajectories: 
    traj = ad.CombineReplicates(FunctionType,WellList,Channel,LocOfOutput,CorrectLinearShift,
                                ShowLinearShift=False)
    
    average_traj = ad.GetAverageAcrossPoints(FunctionType,WellList,Channel,LocOfOutput,CorrectLinearShift)
    
    #Extracting information for the figure axis 
    title, dt, time_unit = GetPlottingInformationFromKey(KeyInformation,WellList)
    
    #Finding the number of cells and timepoints 
    n_cells,n_timepoints = traj.shape

    #X-axis 
    x_range = np.arange(n_timepoints)*dt
    
    plt.figure(figsize=(15,5))
    plt.plot(x_range,traj.T,c=(0.859375,0.859375,0.859375))
    plt.plot(x_range,average_traj,linewidth=5,c='k',linestyle="--")

    #Adding color to 10 random trajectories: 
    if n_cells > 10: 
        chosen_traj = np.random.randint(0,n_cells,10) 
    else: 
        chosen_traj = np.arange(n_cells)

    for ith_cell in chosen_traj: 
        plt.plot(x_range,traj[ith_cell,:],linewidth=5)
        

    
    #Set Y-axis labels so outliers do not distort the scale 
    lower_percentile = np.nanpercentile(traj,1)
    upper_percentile = np.nanpercentile(traj,99)

    distance = upper_percentile - lower_percentile 

    #Extending the bounds by 50 percent 
    extended_distance =  distance*1.5

    #Mean distance between the percentiles 
    mean_distance = (lower_percentile+upper_percentile)/2

    #Calculating the distances 
    lower_limit = mean_distance - extended_distance/2
    upper_limit = mean_distance + extended_distance/2

    plt.ylim([lower_limit,upper_limit])
    
    AddLabelsToGraph(title,Channel,FunctionType,time_unit,CorrectLinearShift)    
    
    return title

def NormalizeTrajectory(Traj): 
    '''
    Input must be a TXN value where T is the number of timepoints and M is 
    the number of samples. The function returns a normalized trajectory for 
    each sample
    '''
    return (Traj-np.nanmin(Traj,axis=0))/(np.nanmax(Traj,axis=0)-np.nanmin(Traj,axis=0))

def DirectlyCompareChannels(Function1,Channel1,Function2,Channel2,Key,WellList,LocOfOutput,
                            LinearShift1=False,LinearShift2=False,NormalizeTraj=False,KnownTreatmentTimes=[]): 
    '''
    Function to directly compare channels by plotting all trajectories found in a well list on 
    individual plots. 
    '''
    
    #Getting the Trajectory 1: 
    traj1 = ad.CombineReplicates(Function1,WellList,Channel1,LocOfOutput,LinearShift1,
                                ShowLinearShift=False).T
    
    #Getting the Trajectory 2: 
    traj2 = ad.CombineReplicates(Function2,WellList,Channel2,LocOfOutput,LinearShift2,
                                ShowLinearShift=False).T
    
    if NormalizeTraj: 
        traj1 = NormalizeTrajectory(traj1)
        traj2 = NormalizeTrajectory(traj2)
        
    def AddLinearShiftNotation(Label,LinearShift,FunctionType):
        if LinearShift: 
            return Label+" Linearly Shifted Results "+ReturnYAxisLabel(FunctionType)
        else: 
            return Label + " "+ReturnYAxisLabel(FunctionType)
    
    traj1label = AddLinearShiftNotation(Channel1,LinearShift1,Function1)
    traj2label = AddLinearShiftNotation(Channel2,LinearShift2,Function2)
        
    n_timepoints,n_traj = traj1.shape
    
    #Extracting information for the figure axis 
    title, dt, time_unit = GetPlottingInformationFromKey(Key,WellList)
    x_range = np.arange(n_timepoints)*dt
    for ith_traj in np.arange(n_traj):
        plt.figure(figsize=(15,5))
        plt.plot(x_range,traj1[:,ith_traj],label=traj1label)
        plt.plot(x_range,traj2[:,ith_traj],label=traj2label)
        
        plt.grid()
        plt.xlim([x_range.min(),x_range.max()])
        plt.xlabel("Time: [{}]".format(time_unit),size=15)
        plt.ylabel("Normalized Output",size=15)
        plt.title(title+"   ///  Cell Label: {}".format(ith_traj),size=15)
        plt.legend(fontsize=10)
        
        for ith_treatment in KnownTreatmentTimes: 
            plt.axvline(ith_treatment,color='k')
    
    
    return plt.gca

def GetUniqueWellListCombos(LocOfKey,ExperimentNumber):  
    #Load KeyInformation File
    df = pd.read_csv(LocOfKey)
    
    #Reduce to experiment code: 
    df = df[df["Expierment Code"]==ExperimentNumber]
    
    #Unique Experimental Conditions: 
    unique_conditions = df["Title"].unique()
    
    output_dictionary = {}
    for ith_unique_condition in unique_conditions: 
        output_dictionary[ith_unique_condition] = df[df["Title"]==ith_unique_condition]["Well"].values
    
    return output_dictionary

def PlotByKmeanClustering(NumClusters,NumComponents,FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift = False,NormalizeTraj=True,StartValue=0): 
    k_means = ml.KMeanTrajectories(NumClusters,NumComponents,FunctionType,WellList,KeyInformation,Channel,LocOfOutput,CorrectLinearShift,NormalizeTraj,StartValue)

    #Extracting information for the figure axis 
    title, dt, time_unit = GetPlottingInformationFromKey(KeyInformation,WellList)

    #Getting the trajectories: 
    traj = ad.CombineReplicates(FunctionType,WellList,Channel,LocOfOutput,CorrectLinearShift,
                                ShowLinearShift=False)

    #Finding the number of cells and timepoints 
    n_cells_total,n_timepoints = traj.shape

    x_range = np.arange(n_timepoints)*dt

    #Normalize Trajectory 
    if NormalizeTraj:
        #The input of normalize trajectory must be Time X Cells 
        traj = NormalizeTrajectory(traj.T)
    else:
        #Trajectory must be flipped to be Time X Cells 
        traj = traj.T

    for ith_plot in range(NumClusters): 
        plt.figure(figsize=(15,5))

        conditional = ith_plot==k_means.labels_
        traj_to_plot = traj[:,ith_plot==k_means.labels_].copy()
        plt.plot(x_range,traj_to_plot,c=(0.859375,0.859375,0.859375))
        mean_of_selected = np.nanmean(traj_to_plot,axis=1)
        
        #Number of chosen cells in the ith cluster 
        n_cells = np.sum(conditional)
        loc_of_hits = np.where(conditional)[0]

        #Adding color to 10 random trajectories: 
        if n_cells > 10: 
            chosen_indices = np.random.randint(0,n_cells,10) 
            chosen_traj = loc_of_hits[chosen_indices]
        else: 
            chosen_indices = np.arange(n_cells)
            chosen_traj = loc_of_hits[chosen_indices]

        for ith_cell in chosen_traj: 
            plt.plot(x_range,traj[:,ith_cell],linewidth=5)

        #Plotting the mean last so it stand out on top 
        plt.plot(x_range,mean_of_selected,linewidth=5,c='k',linestyle='--')
        AddLabelsToGraph(title,Channel,FunctionType,time_unit,CorrectLinearShift)   

def ReturnYAxisLabel(Function): 
    if Function == ad.CalculateCytoplasmToNucleusRatio:
        return "Cytoplasm to Nucleus Ratio"
    elif Function == ad.GetCytoplasmMean: 
        return "Cytoplasm Mean"
    elif Function == ad.GetBackgroundCorrectedCytoplasmMean: 
        return "Background Corrected Cytoplasm Mean"
    elif Function == ad.GetBackground:
        return "Average Background"
    else: 
        return "Unkown Function"


def PlotAvgPixelArea(LocOfOutput): 
    '''
    After the ExtractCellInfo was run, this function can be used to visualize 
    the results. This is best if done in Jupyter notebook. 
    '''
    try: 
        data = np.load(LocOfOutput+"/"+"MaskStatistics.npy",allow_pickle=True)
        data = data.item()
    except: 
        print("ALERT: Need to run ExtractCellInfo function first!")
        return 
    
    titles = data.keys()
    for ith_title in titles: 
        wells = data[ith_title].keys()
        plt.figure(figsize=(10,10))
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)
        
        min_value = 1E10
        max_value = 0 
        for ith_well in wells:
            data2plot = data[ith_title][ith_well]["Average Nucleus Area Per Frame"]['Mean']
            ax1.plot(data2plot)
            max_value = max(data2plot.max(),max_value)
            min_value = min(data2plot.min(),min_value)

            data2plot = data[ith_title][ith_well]["Average Cytoplasm Area Per Frame"]['Mean']
            ax2.plot(data2plot)
            max_value = max(data2plot.max(),max_value)
            min_value = min(data2plot.min(),min_value)
            
            data2plot = data[ith_title][ith_well]["Nucleus Pixel Area Per Cell"]
            ax3.hist(data2plot.flatten())
            
            data2plot = data[ith_title][ith_well]["Cytoplasm Pixel Area Per Cell"]
            ax4.hist(data2plot.flatten(),)
        
        
        ax1.set_xlabel("Slide Number",size=15)
        ax1.set_ylabel("Sum of Pixels in Area",size=15)
        ax1.set_ylim([min_value, max_value])
        
        ax2.set_xlabel("Slide Number",size=15)
        ax2.set_ylabel("Sum of Pixels in Area",size=15)
        ax2.set_ylim([min_value, max_value])
        
        
        ax3.set_xlabel("Number of Pixels in Nuclei",size=15)
        ax3.set_ylabel("Count",size=15)
        
        ax4.set_xlabel("Number of Pixels in Cytoplasm",size=15)
        ax4.set_ylabel("Count",size=15)
        
        ax1.set_title("Average Nucleus Pixel Area")
        ax2.set_title("Average Cytoplasm Pixel Area")
        plt.suptitle(ith_title,size=20)
        plt.tight_layout()
    
def PlotTrajectory(Figure2Update,Trajectories,PathNumber,CurrentSlide,UpdateSlideOnly):
    import plotly.graph_objects as go
    import plotly.express as px
    filters = list(Trajectories.keys())
    plt.figure(figsize=(15,5))
    for ith,fileterID in enumerate(filters): 
        data2Plot = Trajectories[fileterID]["Nucleus"][PathNumber,:]-Trajectories[fileterID]["Background"].squeeze()
        if UpdateSlideOnly: 
            with Figure2Update.batch_update():
                Figure2Update.data[ith*2+1].x = [CurrentSlide]
                Figure2Update.data[ith*2+1].y = [data2Plot[CurrentSlide]]
                Figure2Update.data[ith*2].y = data2Plot
        else:
            Figure2Update.add_scatter(name=fileterID)
            Figure2Update.data[ith*2].y = data2Plot

            Figure2Update.add_trace(go.Scatter(mode='markers',y=[data2Plot[CurrentSlide]],x=[CurrentSlide],
                marker=dict(color='LightSkyBlue',
                            size=10,
                            line=dict(color='MediumPurple',width=2)),
                            showlegend=False))     
    if True:
        Figure2Update.show()
    return True
