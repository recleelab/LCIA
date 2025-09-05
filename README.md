# LCIA 
**L**ive **C**ell **I**maging **A**nalysis 

## Summary 
A set of tools developed for analysis of live cell imaging data where: 
1. Manually define regions for the nucleus and cytoplasm acrossed an entire movie and for a set of cells. The process has been optimized to reduce the number of keystrokes and mouse clicks for speed. 
2. Utilize CellPose to segment a set of expierments and store the masks. 
3. Tools to extract the data from the masks and automatically plot the results. 

## Setup 
1. Download and install the Individual Edition of Anaconda 
2. Create a new enviroment, go to the command line and type the following: 

conda env create -f environment.yml

3. Activate enviroment by either typeing: 
conda activate LCIA   
or 
source activate LCIA


## KeyInformation 
The KeyInformation input needs to be a pandas dataframe with the following columns: 
1. **Date** : The date in which the experiment was ran and can be in any format as specified by the user. 
2. **Experiment Code** : The experiment code is for easy filtering of the dataframe. The KeyInformation document has all expierments that were run and this code is added to be able to filter out only the experiments of interest during analysis 
3. **File** : File name of the images. 
4. **Well** : Short label given to the image for reference. It usually references the well location on a 96-well plate and the point number in which the image stack corresponds to. 
5. **Title** : Text that will display on a figure title when graphing using LCIA tools. 
6. **Description** : Additoinal text that may appear on graphs when using LCIA tools. 
7. **SavePrefix** : Save prefix when saving analysis from LCIA tools 
8. **NumberOfPoints** : Number of points per image stack. This is an unlikely scenario and the LCIA tools may not be able to actually support this. Kept here for legacy purposes 
9. **Channels** : The channels imaged in the image stack. Multiple filters are separated by double dashes "//". Therefore, CFP//YFP//mCherry//Pol, indicates 4 filters were used. This list must be in the same order in which the images were taken. 
10. **dt** : Frequency of time the images were taken 
11. **Time Unit** : Time unit in which dt was reported 
12. **TimeOfTreatment**: Time of treatment (if given). Currently, can only support a single treatment time. 

Below is an example of a single row for an experiment 
|Date|Experiment Code|File|Well|Title|Description|SavePrefix|NumberOfPoints|Channels|dt|Time Unit|TimeOfTreatment|
|---|---|---|---|---|---|---|---|---|---|---|---|
|12/11/20|1001000|NameOfFile.dv|A1_P01|Control|No Changes To Media|Control_A01| 1|CFP//YFP//mCherry//Pol|2|minute|14|
