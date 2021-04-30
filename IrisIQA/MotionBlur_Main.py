"""
Compute IQA (Image Quality Assessment) of iris image stored in a folder
and listed by a correlated text file
Motionblur 
Written by Yunlong Wang 2020.09.10
Modified by Yunlong Wang 2021.01.13
Rewritten in Python by Hongda Liu 2021.04.13
--- the titles and legends of figures are changed, 
'ocluar regions'->'iris region'
'0~25' -> '[0,25)'
the calculated score is round to integers
"""

from config.MotionBlurConfig import *
from functions import computeMotionblur
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

# Check if out/xxx_motionblur_stat.mat already exists, load in the variables
if os.path.exists(saveFile) and not overwrite :
    print('********************************')
    print('Results of Motion Blur already exists, load in...')
    data = scio.loadmat(saveFile)
    motionblurVal_Img = data['motionblurVal_Img'].reshape(N,1)
    motionblurVal_Iris = data['motionblurVal_Iris'].reshape(N,1)
    print('********************************')
else :
    motionblurVal_Img = np.zeros(N)
    motionblurVal_Iris = np.zeros(N)

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
                
                motionblurVal_Img[i] = computeMotionblur(img)
                motionblurVal_Iris[i] = computeMotionblur(cropped_img)

plt.figure(1)
plt.hist(motionblurVal_Img,range(1,101),alpha=0.5)
plt.hist(motionblurVal_Iris,range(1,101),alpha=0.5)
plt.legend(['Full Image','Iris Region'],prop = {'size':18})
plt.xlabel('Motion blur',{'size':18})
plt.ylabel('Frequency',{'size':18})
plt.savefig(res_path+database_name+'_motionblur_histogram.png')

meanMotionblur_Img = np.mean(motionblurVal_Img)
stdMotionblur_Img = np.std(motionblurVal_Img, ddof=1)

meanMotionblur_Iris = np.mean(motionblurVal_Iris)
stdMotionblur_Iris = np.std(motionblurVal_Iris, ddof=1)

# print out the statistics of performance indices
print('********************************')
print('Motion blur Statistics');
print('--------------------------------')
print('Full Image: mean %.4f std %.4f' %(meanMotionblur_Img, stdMotionblur_Img))
print('Iris Region: mean %.4f std %.4f' %(meanMotionblur_Iris, stdMotionblur_Iris))
print('--------------------------------')

# Calculate and Plot the distributions of motionblur,range 1-100
numLevels = 4;
interval = 25;

plt.figure(2)
motionblurVal_distribution = np.zeros((numLevels,max(labels)))
for n in range(N):
    score = motionblurVal_Img[n]
    class1 = round(labels[n])
    level = min(math.floor(score/interval)+1,numLevels)
    motionblurVal_distribution[level-1,class1-1] = motionblurVal_distribution[level-1,class1-1] + 1;
for l in range(1,numLevels+1):
    plt.subplot(2,2,l)
    levelDistribution = motionblurVal_distribution[l-1,:]
    levelRatio = 100.0*np.sum(levelDistribution)/N
    plt.bar(range(max(labels)),levelDistribution,color='g')
    plt.tight_layout()
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    title_str = '[%d,%d) %.2f%%'%(math.floor((l-1)*interval),math.floor(l*interval),levelRatio)
    plt.title(title_str) 
plt.savefig(res_path+ database_name+ '_motionblur_distribution_fullImg.png')

plt.figure(3)
motionblurVal_distribution_Iris = np.zeros((numLevels,max(labels)))
for n in range(N):
    score = motionblurVal_Iris[n]
    class1 = round(labels[n])
    level = min(math.floor(score/interval)+1,numLevels)
    motionblurVal_distribution_Iris[level-1,class1-1] = motionblurVal_distribution_Iris[level-1,class1-1] + 1;
for l in range(1,numLevels+1):
    plt.subplot(2,2,l)
    levelDistribution = motionblurVal_distribution_Iris[l-1,:]
    levelRatio = 100.0*np.sum(levelDistribution)/N
    plt.bar(range(max(labels)),levelDistribution,color='g')
    plt.tight_layout()
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    title_str = '[%d,%d) %.2f%%'%(math.floor((l-1)*interval),math.floor(l*interval),levelRatio)
    plt.title(title_str) 
plt.savefig(res_path+ database_name+ '_motionblur_distribution_IrisRegion.png')
plt.show()

# save results to .mat file
scio.savemat(saveFile, {'motionblurVal_Img': motionblurVal_Img,
                        'motionblurVal_Iris': motionblurVal_Iris,
                        'motionblurVal_distribution': motionblurVal_distribution,
                        'motionblurVal_distribution_Iris': motionblurVal_distribution_Iris})
