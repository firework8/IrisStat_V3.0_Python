"""
iris_integrodifferntial_operator 2020.05.25
Yunlong Wang
Rewritten in Python by Hongda Liu 2021.04.20
"""
import os
import sys
import configparser
import numpy as np
import matplotlib
import time
from functions import *

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
type = sys.getfilesystemencoding()
sys.stdout = Logger('evalSegRes_TVMIris.txt')


# Set variables
image_folder='/home/yunlong.wang/BlurringIRISdatabase/image/'
circleGT_folder='/home/yunlong.wang/BlurringIRISdatabase/circle_params/'
circleGT_ext = '.ini'
segGT_folder= '/home/yunlong.wang/BlurringIRISdatabase/iris_mask/'
segGT_ext = '.png'
segRes_folder='/home/yunlong.wang/OpenSourceIris/TVMIris/output/mask/'
segRes_ext = '.png'

# Read images
images = []
for root, dirs, files in os.walk(image_folder):
    for name in files:
        images.append(name)

nrof_images = len(images)

failedSegNo = 0

allP = []
allR = []
allF = []
allE1 = []
allE2 = []
allHdist = []

for i in range(nrof_images) :

    image_name = images[i]
    the_image_name = image_name.split('.')[0]
    print('%d/%d %s'%(i+1,nrof_images,image_name))

    # Name sytle for set Gt
    segGT_file = segGT_folder + the_image_name + segGT_ext
    
    if not os.path.exists(segGT_file) :
        print('no such GT iris mask, %s'%segGT_file)
        sys.exit()

    gtIrisMask = matplotlib.image.imread(segGT_file)

    # Name sytle for circle parpam .ini file
    circleGT_file = circleGT_folder + the_image_name + circleGT_ext
    if not os.path.exists(circleGT_file) :
        print('no such GT cricle params, %s'%circleGT_file)
        sys.exit()
    conf = configparser.ConfigParser()
    conf.read(circleGT_file, encoding="utf-8")  # python3
    sections = conf.sections()
    gtLabel = conf.items('iris')
    irisLabel = {}
    for thepair in gtLabel:
        irisLabel[thepair[0]] = thepair[1]

    # Name style for seg results
    segRes_file = segRes_folder + 'irismask_' + image_name[0:-4] + segRes_ext
    
    if not os.path.exists(segRes_file) :
        print('no such seg result, %s'%segRes_file)
        failedSegNo = failedSegNo + 1
        # set performance index to NaN
        the_allP = np.nan
        the_allR = np.nan
        the_allF = np.nan
        the_allE1 = np.nan
        the_allE2 = np.nan
        the_allHdist = np.nan
        allP.append(the_allP)
        allR.append(the_allR)
        allF.append(the_allF)
        allE1.append(the_allE1)
        allE2.append(the_allE2)
        allHdist.append(the_allHdist)
        continue
    
    segResIrisMask1 = matplotlib.image.imread(segRes_file)
    segResIrisMask = np.zeros([segResIrisMask1.shape[0],segResIrisMask1.shape[1]])
    for i in range(segResIrisMask.shape[0]):
        for j in range(segResIrisMask.shape[1]):
            if segResIrisMask1[i,j] > 127:
                segResIrisMask[i,j] = 1
            else:
                segResIrisMask[i,j] = 0

    
    # 20200525 Functions for Encode Normalized Iris
    P, R, F, E1, E2, Hdist , IoU, errImg= evalSeg(gtIrisMask, segResIrisMask)
    # Normalize the Hausordoff distance by the GT iris diameter
    if np.isinf(Hdist) :
        Hdist = 2
    else :
        iris_diamteter  = float(irisLabel['radius']) * 2
        Hdist = Hdist / iris_diamteter
    print('Precison:%.4f, Recall:%.4f, F-score:%.4f, E1:%.4f, E2:%.4f, Hdist:%.4f'
        %(P, R, F, E1, E2, Hdist))
    
    the_allP = P
    the_allR = R
    the_allF = F
    the_allE1 = E1
    the_allE2 = E2
    the_allHdist = Hdist
    allP.append(the_allP)
    allR.append(the_allR)
    allF.append(the_allF)
    allE1.append(the_allE1)
    allE2.append(the_allE2)
    allHdist.append(the_allHdist)
    

validP = []
validR = []
validF = []
validE1 = []
validE2 = []
validHdist = []

for i in range(len(allP)):
    if not np.isnan(allP[i]) :
        validP.append(allP[i])
        validR.append(allR[i])
        validF.append(allF[i])
        validE1.append(allE1[i])
        validE2.append(allE2[i])
        validHdist.append(allHdist[i])

uP = np.mean(validP)
uR = np.mean(validR)
uF = np.mean(validF)
uE1 = np.mean(validE1)
uE2 = np.mean(validE2)
uHdist = np.mean(validHdist)

stdP = np.std(validP,ddof=1)
stdR = np.std(validR,ddof=1)
stdF = np.std(validF,ddof=1)
stdE1 = np.std(validE1,ddof=1)
stdE2 = np.std(validE2,ddof=1)
stdHdist = np.std(validHdist,ddof=1)

print('--------------------------------------------------')
now = time.localtime()
nowt = time.strftime("%Y-%m-%d %H:%M:%S", now)  
print(nowt)
print('--------------------------------------------------')

print('--------------------------------------------------')
print('%d/%d failed'%(failedSegNo, nrof_images))
print('Precsion mean: %.4f, std: %.4f'%(uP, stdP))
print('Recall mean: %.4f, std: %.4f'%(uR, stdR))
print('F-score mean: %.4f, std: %.4f'%(uF, stdF))
print('E1 mean: %.4f, std: %.4f'%(uE1, stdE1))
print('E2 mean: %.4f, std: %.4f'%(uE2, stdE2))
print('Hausdorff distance mean: %.4f, std: %.4f'%(uHdist, stdHdist))
print('--------------------------------------------------')
