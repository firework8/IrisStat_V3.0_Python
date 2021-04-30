"""
Configuration of motion blur computation 
Referred to the lab-produced iris SDK, the part of image quality attached in the bottom
Created by Yunlong Wang 2021.01.08
Rewritten in Python by Hongda Liu 2021.04.13
"""

import os

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
iris_label_path = 'D:\\Experiment\\BlurringIRISdatabase\\circle_params\\';
iris_label_ext = '.ini';

# border (5 pixels by default)
B = 5;

res_path = 'out/';
if not os.path.exists(res_path):
        os.mkdir(res_path)

saveFile = res_path+database_name+'_motionblur_stat.mat'
overwrite = 0

