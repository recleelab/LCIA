import sys
import numpy as np 
sys.path.append("../../")
from LCIA import process_images as pi

#===============================================================================
#==================Testing EnhanceContrast======================================
#===============================================================================

#Creating an example with obvious solution 
test_image = np.arange(0,100,1).reshape(10,10)

#Normalizing test matrix to ensure functionality 
normalized_image = pi.EnhanceContrast(test_image)

#Normalized test matrix should be an 8-bit 
test_8bit = normalized_image.dtype==np.uint8

#Test Max value: 
max_value = normalized_image.max()==255
min_value = normalized_image.min()==0

#Test Default Values: First 6 and last 5 values in the matrix should be zero and
#255 respectively. test cases based off of removing the bottom and top 5 percent 
percentile_bottom = np.all(normalized_image[0,0:6]==0)
percentile_top = np.all(normalized_image[-1,-5:]==255)

#Change top and bottom percentiles: 
normalized_image = pi.EnhanceContrast(test_image,75,25)

#In the test case, the number of zeros in the normalized image must be 26 and 
#the number of 255's must be 25
num_zeros = np.sum(normalized_image==0) ==26
num_255 = np.sum(normalized_image==255) ==25


Total_EnhanceContrastTest = np.all([test_8bit,max_value,min_value,
                        percentile_bottom, percentile_top,num_zeros,num_255])

if Total_EnhanceContrastTest: 
    print("===========================================")
    print("EnhanceContrast passed all tests!!!!")
    print("===========================================")
else:
    print("===========================================")
    print("Something is wrong, here are the details:")
    print("EnhanceContrast")
    print("8-Bit output test: ",test_8bit)
    print("Max Value 255: ",max_value)
    print("Min Value 0: ",min_value)
    print("Default Bottom Percentile Working: ",percentile_bottom)
    print("Default Top Percentile Working: ",percentile_top)
    print("Number of Minimum Values in Custom Percentile Range: ",num_zeros)
    print("Number of Maximum Values in Custom Percentile Range: ",num_255)  


#===============================================================================
#==================Testing ProcessingProcedure1=================================
#===============================================================================

#The output of ProcessingProcedure1 and EnhanceContrast should be the same for 
#the same image: 
test_image = np.arange(0,100,1).reshape(10,10)
normalized_image = pi.EnhanceContrast(test_image)
normalized_image_stack = pi.ProcessingProcedure1(test_image)

are_functions_same = np.all(normalized_image==normalized_image_stack) 

#Test Stack 
test_stack = np.arange(0,200,1).reshape(2,10,10)
test_normalized_stack = pi.ProcessingProcedure1(test_stack)

test_output_shape= test_stack.shape==(2,10,10)

test_individual_1 = np.all(test_normalized_stack[0,:,:]==pi.EnhanceContrast(test_stack[0,:,:]))
test_individual_2 = np.all(test_normalized_stack[1,:,:]==pi.EnhanceContrast(test_stack[0,:,:]))

Total_ProcessingProcedure1 = [are_functions_same,test_output_shape,test_individual_1,test_individual_2]

if np.all(Total_ProcessingProcedure1): 
    print("===========================================")
    print("ProcessingProcedure1 passed all tests!!!!")
    print("===========================================")
else: 
    print("===========================================")
    print("Something is wrong, here are the details:")
    print("ProcessingProcedure1")
    print("Single Image Input: ",are_functions_same)
    print("Output Shape Test: ",test_output_shape)
    print("Individual Test 1: ",test_individual_1)
    print("Individual Test 2: ",test_individual_2)