"""
Configuration of sharpness computation 
Sharpness (the degree of focus) Index defined exactly the same as
ISO/IEC 29794-6:2015(E)
Information technology  Biometric sample quality 
Part 6: Iris image data
6.2.10 Sharpness
Written by Yunlong Wang 2020.09.10
Rewritten in Python by Hongda Liu 2020.04.13
"""

import os
import numpy as np 

# database name
database_name = 'blurring'

# Iris image folders and list
root_path = "D:\\Experiment\\BlurringIRISdatabase\\image\\"
ext = '.jpg'
mask_path = 'D:\\Experiment\\BlurringIRISdatabase\\iris_mask\\'
mask_ext = '.png'
img_list_txt = 'D:\\Experiment\\BlurringIRISdatabase\\imgList.txt'

# 2021.01.08
# add the path and extensions of iris label (circle params)
iris_label_path = 'D:\\Experiment\\BlurringIRISdatabase\\circle_params\\'
iris_label_ext = '.ini'

# border (5 pixels by default)
B = 5;

res_path = 'out/';
if not os.path.exists(res_path):
        os.mkdir(res_path)

saveFile = res_path+database_name+'_sharpness_stat.mat'
overwrite = 0

# convolution kernel
F = np.array([
    [0,1,1,2,2,2,1,1,0],
    [1,2,4,5,5,5,4,2,1],
    [1,4,5,3,0,3,5,4,1],
    [2,5,3,-12,-24,-12,3,5,2],
    [2,5,0,-24,-40,-24,0,5,2],
    [2,5,3,-12,-24,-12,3,5,2],
    [1,4,5,3,0,3,5,4,1],
    [1,2,4,5,5,5,4,2,1],
    [0,1,1,2,2,2,1,1,0]
              ]) 

# Constant empirically to be 1 800 000 as the ISO/IEC 29794-6:2015(E)
C = 1.8E6