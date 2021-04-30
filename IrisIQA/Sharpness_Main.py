"""
Compute IQA (Image Quality Assessment) of iris image stored in a folder
and listed by a correlated text file
Sharpness (the degree of focus) Index defined exactly the same as
ISO/IEC 29794-6:2015(E)
Information technology  Biometric sample quality
Part 6: Iris image data
6.2.10 Sharpness

Written by Yunlong Wang 2020.09.10
Revised by Yunlong Wang 2021.01.08
Modified by Yunlong Wang 2021.01.13
Rewritten in Python by Hongda Liu 2021.04.13
--- the titles and legends of figures are changed, 
'ocluar Region'->'iris region'
'0~25' -> '[0,25)'
the calculated score is round to integers
"""

from config.SharpnessConfig import *
from functions import computeSharpness
import os
import sys
import math
import numpy as np
import scipy.io as scio
from PIL import Image 
import time
from tqdm import tqdm
import configparser
import matplotlib.pyplot as plt 

# Read images
if os.path.exists(res_path):
    with open(img_list_txt, 'r') as f:
        img_names = f.readlines()
        for i in range(0, len(img_names)):
            img_names[i] = img_names[i].strip()
else :
    print('Error:No such image list text file, '+img_list_txt)
    sys.exit()

N = len(img_names)
if N == 0 :
    print('Error:No such image list text file, '+img_list_txt)
    sys.exit()

# Split the correlated IDs frome the filenames
id = []
for i in range(0, len(img_names)):
    thenames = img_names[i].split('_')
    id.append(thenames[0]) 
classes = list(set(id))
classes.sort()
labels = []
for i in range(0, len(id)):
    labels.append(classes.index(id[i]))

# Check if out/xxx_sharpness_stat.mat already exists, load in the variables
if os.path.exists(saveFile) and not overwrite :
    print('********************************')
    print('Results of Sharpness Calcuation already exists, load in...')
    data = scio.loadmat(saveFile)
    sharpnessVal_Img = data['sharpnessVal_Img'].reshape(N,1)
    sharpnessVal_Iris = data['sharpnessVal_Iris'].reshape(N,1)
    print('********************************')
else :
    sharpnessVal_Img = np.zeros(N)
    sharpnessVal_Iris = np.zeros(N)

    for i in tqdm(range(N), desc='Computing Motionblur:'):
        thenames = img_names[i].split('.')
        imgName = thenames[0]
        imgFile = root_path + imgName + ext
        irislabelFile = iris_label_path + imgName + iris_label_ext
        if not os.path.exists(imgFile):
            print('NO such iris code file, ' + imgFile)
        else:
            I = Image.open(imgFile)   
            img = np.array(I)
            # img = img.astype('float16')
            if os.path.exists(irislabelFile):
                conf = configparser.ConfigParser()
                conf.read(irislabelFile, encoding="utf-8")  # python3
                sections = conf.sections()
                gtLabel = conf.items('iris')
                irisLabel = {}
                for thepair in gtLabel:
                    irisLabel[thepair[0]] = thepair[1]
                if irisLabel['exist'] == 'false' :
                    print('No iris region labeled in this image, so choose full image to compute motion blur')
                    cropped_img = img
                else :
                    irisCx = math.floor(float(irisLabel['center_x']))
                    irisCy = math.floor(float(irisLabel['center_y']))
                    irisR = math.floor(float(irisLabel['radius']))
                cropped_img = img[irisCy-irisR-B-1:irisCy+irisR+B][:,irisCx-irisR-B-1:irisCx+irisR+B]
                
                sharpnessVal_Img[i] = computeSharpness(img,[],F,C)
                sharpnessVal_Iris[i] = computeSharpness(cropped_img,[],F,C)

plt.figure(1)
plt.hist(sharpnessVal_Img,range(1,101),alpha=0.5)
plt.hist(sharpnessVal_Iris,range(1,101),alpha=0.5)
plt.legend(['Full Image','Iris Region'],prop = {'size':18})
plt.xlabel('Sharpness',{'size':18})
plt.ylabel('Frequency',{'size':18})
plt.savefig(res_path+database_name+'_sharpness_histogram.png')

meanSharpness_Img = np.mean(sharpnessVal_Img)
stdSharpness_Img = np.std(sharpnessVal_Img, ddof=1)

meanSharpness_Iris = np.mean(sharpnessVal_Iris)
stdSharpness_Iris = np.std(sharpnessVal_Iris, ddof=1)

# print out the statistics of performance indices
print('********************************')
print('Sharpness Statistics');
print('--------------------------------')
print('Full Image: mean %.4f std %.4f' %(meanSharpness_Img, stdSharpness_Img))
print('Iris Region: mean %.4f std %.4f' %(meanSharpness_Iris, stdSharpness_Iris))
print('--------------------------------')

# Calculate and Plot the distributions of sharpness,range 1-100                
numLevels = 4;
interval = 25;

plt.figure(2)
sharpnessVal_distribution = np.zeros((numLevels,max(labels)))
for n in range(N):
    score = sharpnessVal_Img[n]
    class1 = round(labels[n])
    level = min(math.floor(score/interval)+1,numLevels)
    sharpnessVal_distribution[level-1,class1-1] = sharpnessVal_distribution[level-1,class1-1] + 1;
for l in range(1,numLevels+1):
    plt.subplot(2,2,l)
    levelDistribution = sharpnessVal_distribution[l-1,:]
    levelRatio = 100.0*np.sum(levelDistribution)/N
    plt.bar(range(max(labels)),levelDistribution,color='g')
    plt.tight_layout()
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    title_str = '[%d,%d) %.2f%%'%(math.floor((l-1)*interval),math.floor(l*interval),levelRatio)
    plt.title(title_str) 
plt.savefig(res_path+ database_name+ '_sharpness_distribution_fullImg.png')

plt.figure(3)
sharpnessVal_distribution_Iris = np.zeros((numLevels,max(labels)))
for n in range(N):
    score = sharpnessVal_Iris[n]
    class1 = round(labels[n])
    level = min(math.floor(score/interval)+1,numLevels)
    sharpnessVal_distribution_Iris[level-1,class1-1] = sharpnessVal_distribution_Iris[level-1,class1-1] + 1;
for l in range(1,numLevels+1):
    plt.subplot(2,2,l)
    levelDistribution = sharpnessVal_distribution_Iris[l-1,:]
    levelRatio = 100.0*np.sum(levelDistribution)/N
    plt.bar(range(max(labels)),levelDistribution,color='g')
    plt.tight_layout()
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    title_str = '[%d,%d) %.2f%%'%(math.floor((l-1)*interval),math.floor(l*interval),levelRatio)
    plt.title(title_str) 
plt.savefig(res_path+ database_name+ '_sharpness_distribution_Iris.png')
plt.show()

# save results to .mat file
scio.savemat(saveFile, {'sharpnessVal_Img': sharpnessVal_Img,
                        'sharpnessVal_Iris': sharpnessVal_Iris,
                        'sharpnessVal_distribution': sharpnessVal_distribution,
                        'sharpnessVal_distribution_Iris': sharpnessVal_distribution_Iris})
