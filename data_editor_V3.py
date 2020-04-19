# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image
import os

#%% Functions
def datagenerator(data_directory,blur_directory,pixel_number):
    for data_file in os.listdir(data_directory):
        img = Image.open(data_directory+'\\'+data_file)
        # Resize smoothly down to 16x16 pixels
        imgSmall = img.resize((pixel_number,pixel_number),resample=Image.BILINEAR)
        
        # Scale back up using NEAREST to original size
        result = imgSmall.resize(img.size,Image.NEAREST)
        result.save(blur_directory+'\\'+data_file)
    return

def image_combiner(dir_A,dir_B):
    
    return

#%% What you want it to do

old_bin = 'C:\\Users\\piete\\OneDrive\\Documents\\Studies\\Space Engineering\\Third Period 2019-2020\\Deep Learning\\Reproducibility Project\\Group 28\\Data_Model_V2\\Original_Data'
new_bin = 'C:\\Users\\piete\\OneDrive\\Documents\\Studies\\Space Engineering\\Third Period 2019-2020\\Deep Learning\\Reproducibility Project\\Group 28\\Data_Model_V2\\New_Data'

datagenerator(old_bin,new_bin,64)