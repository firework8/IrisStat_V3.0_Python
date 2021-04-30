"""
Configurations TO accommodate iris features stored in 'iriscode'
"""

import os
import numpy as np

global algo_name
global norm_case
global output_path

algo_name = 'OSIris'
norm_case = 'rubbersheet'
output_path = 'out/'

if not os.path.exists(output_path):
    os.mkdir(output_path)
save_matfile = output_path + algo_name +'_' + norm_case + '_stat.mat'

path_to_code = 'D:\\Experiment\\BlurringIRISdatabase\\templates\\OSIris\\%s\\'%(norm_case)
path_to_mask = 'D:\\Experiment\\BlurringIRISdatabase\\normalized_%s\\mask\\' %(norm_case)

shift_bit = range(-15,16)

# resolution of DET curve
det_resolution=100000

# if or not overwrite some stored variables like similarity matrix
overwrite = 1

# extensions of images
ext = '*.png'

# feature mode, 'iriscode' or 'vector' 
feat_mode = 'iriscode'

# select points from the iris code when calculating similarities
points_file = 'D:\\Experiment\\BlurringIRISdatabase\\templates\\points.txt'

if not os.path.exists(points_file):
    print('NO such file storing the points: ' + points_file)
    points = []

if not os.path.exists(points_file):
    print('NO such file storing the points: ' + points_file)
    points = []

with open(points_file, 'r') as f:
    points_in = f.readlines()
    for i in range(0, len(points_in)):
        points_in[i] = points_in[i].strip()
H = []
W = []
for i in range(0, len(points_in)):
    thenames = points_in[i].split('\t',1)
    H.append(thenames[0]) 
    if i != 0:
        W.append(thenames[1]) 
    else :
        W.append(0) 
points = np.zeros([2,len(H)-1])
points[0,:] = np.array(H[1:])
points[1,:] = np.array(W[1:])

valid_image_list = 'D:\\Experiment\\BlurringIRISdatabase\\imgList_%s.txt'%norm_case