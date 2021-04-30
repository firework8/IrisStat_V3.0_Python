"""
Configurations TO accommodate iris features stored in 'iriscode'
"""
import os

global algo_name
global norm_case
global output_path

algo_name = 'OM'
norm_case = 'circularHough_GT'
output_path = 'out/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
save_matfile = output_path + algo_name +'_' + norm_case + '_stat.mat'

root_iris_path = 'D:\\Experiment\\BlurringIRISdatabase\\OM\\out\\%s\\'%norm_case
path_to_code = root_iris_path + 'iris\\'
# root_mask_path = root_iris_path
# path_to_mask1 = root_iris_path + 'mask/'
# path_to_mask2 = root_iris_path + 'mask/'

root_mask_path = 'D:\\Experiment\\BlurringIRISdatabase\\OM\\out\\%s\\'%norm_case
path_to_mask = root_iris_path + 'mask\\'
path_to_mask2 = root_iris_path + 'mask\\'

# lobeDis=7:3:7
lobeDis=7

rotation_angle_num = 11


# resolution of DET curve
det_resolution = 100000

# if or not overwrite some stored variables like similarity matrix
overwrite = 1

# extensions of images
ext = '*.mat'

feat_mode = 'simpleOM'

# select valid points on the iris template
points = []
