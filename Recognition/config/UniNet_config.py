"""
Configurations TO accommodate iris features stored in 'iriscode'
"""
import os

global algo_name
global norm_case
global output_path

algo_name = 'UniNet_NoMaskNet'
norm_case = 'rubbersheet'
output_path = 'out/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
save_matfile = output_path + algo_name + '_stat.mat'

root_iris_path = 'D:\\Experiment\\UniNet\\out\\%s\\'%norm_case
path_to_code = root_iris_path + 'iris\\'

root_mask_path = 'D:\\Experiment\\BlurringIRISdatabase\\normalized_%s\\'%norm_case
path_to_mask = root_iris_path + 'mask\\'
path_to_mask2 = root_iris_path + 'mask\\'

shift_bit = range(-15,16)

# H = 64
# W = 512

# resolution of DET curve
det_resolution = 100000

# if or not overwrite some stored variables like similarity matrix
overwrite = 1

# extensions of images
ext = '*.png'

# feature mode, 'iriscode' or 'vector' 
feat_mode = 'iriscode'

# select valid points on the iris template
points = []

# List of Valid Images
valid_image_list = 'D:\\Experiment\\BlurringIRISdatabase\\imgList_%s.txt'%norm_case

