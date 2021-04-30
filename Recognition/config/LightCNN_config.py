"""
Configurations TO accommodate iris features stored in 'iriscode'
"""
import os

global algo_name
global output_path

algo_name = 'LightCNN'
output_path = 'out/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
save_matfile = output_path + algo_name + '_stat.mat'

# .mat file where features and labels are saved
code_label_matfile = '/home/yunlong.wang/OpenSourceIris/NDLightCNN9/lightCNN.mat'

# Length of Feature
feat_length = 256

# resolution of DET curve
det_resolution=100000

# overwrite or not already stored variables like similarity matrix
overwrite = 1

# extensions of images
ext = '*.png'

# feature mode, 'iriscode' or 'vector' 
feat_mode = 'vector'
