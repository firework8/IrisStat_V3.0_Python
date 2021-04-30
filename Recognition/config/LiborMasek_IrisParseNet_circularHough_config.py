"""
Configurations TO accommodate iris features stored in 'iriscode'
"""

import os

global algo_name
global norm_case
global output_path

algo_name = 'Libor Masek'
norm_case = 'circularHough_IrisParseNet'
output_path = 'out/'

if not os.path.exists(output_path):
        os.mkdir(output_path)
save_matfile = output_path + algo_name +'_' + norm_case + '_stat.mat'

root_iris_path = '/home/yunlong.wang/OpenSourceIris/Libor Masek-iriscode/templates/%s/'%(norm_case)
path_to_code = root_iris_path + 'iris/'
# root_mask_path = root_iris_path
# path_to_mask1 = root_iris_path + 'mask/'
# path_to_mask2 = root_iris_path + 'mask/'

root_mask_path = '/home/yunlong.wang/OpenSourceIris/Libor Masek-iriscode/templates/%s/'%(norm_case)
path_to_mask = root_iris_path + 'mask/'
path_to_mask2 = root_iris_path + 'mask/'

shift_bit = range(-15,16)

H = 64
W = 1024

# resolution of DET curve
det_resolution = 100000

# if or not overwrite some stored variables like similarity matrix
overwrite = 1

# extensions of images
ext = '*.png'

# feature mode, 'iriscode' or 'vector' 
feat_mode = 'iriscode'

# Select certain points on iriscodes if variable 'points' is not empty
points = []

# if or not use valid image list
valid_image_list = []
