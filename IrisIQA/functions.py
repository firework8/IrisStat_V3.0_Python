import numpy as np
import math
import cv2
import scipy.signal

def computeMotionblur( img ):
    """
    Compute IQA (Image Quality Assessment) of iris image stored in a folder
    and listed by a correlated text file
    Motion Blur Degree
    Written by Yunlong Wang 2020.09.15
    Rewritten in Python by Hongda Liu 2021.04.13
    """
    integralImg = cv2.integral(img)
    [row,col] = img.shape
    countPtNum = 0
    blurscoreSum = 0
    for r in range(2,row-6) :
        for c in range (2,col-6,4):
            down = integralImg[r-1,c+6]+integralImg[r-2,c-2]-integralImg[r-1,c-2]-integralImg[r-2,c+6];
            up = integralImg[r,c+6]+integralImg[r-1,c-2]-integralImg[r,c-2]-integralImg[r-1,c+6];
            blurscoreSum = blurscoreSum + (up-down)*(up-down);
            countPtNum = countPtNum + 1;
    if blurscoreSum < 0 :
        BlurScore = 100
    else :
        # BlurScore = math.floor(math.sqrt(blurscoreSum/countPtNum)/8);
        BlurScore = math.floor(math.sqrt(blurscoreSum/countPtNum));
    if BlurScore >= 100:
        BlurScore = 99
    return BlurScore

def computeSharpness(img, mask, F, C ):
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
    """
    convImg = scipy.signal.correlate2d(img, F, 'same')
    s = (int)((F.shape[0]-1)/2)
    convImg = convImg[ ::s,::s]
    ss = np.int64(convImg**2).sum()
    power = ss / (convImg.shape[0]*convImg.shape[1])
    sharpness_val_full = 100.0*power*power /(power*power+C*C)
    
    if sharpness_val_full < 0 :
        sharpness_val_full = 0
    else :
        sharpness_val_full = math.floor(sharpness_val_full)
    if sharpness_val_full >= 100 :
        sharpness_val_full = 99
    if len(mask) == 0 or convImg.shape[0] != mask.shape[0] or convImg.shape[1] != mask.shape[1] :
        sharpness_val_mask = sharpness_val_full
    elif len(mask.nonzero()[0]) != 0:
        convImg_masked = convImg*mask
        ss = np.sum(np.sum(convImg_masked**2))
        power = ss / len(mask.nonzero()[0])
        C = C * len(mask.nonzero()[0]) / img.shape[0] * img.shape[1]
        sharpness_val_mask = 100.0*power*power /(power*power+C*C)
    else :
        sharpness_val_mask = 0
    if sharpness_val_mask < 0 :
        sharpness_val_mask = 0
    else :
        sharpness_val_mask = math.floor(sharpness_val_mask)
    if sharpness_val_mask >= 100 :
        sharpness_val_mask = 99

    return sharpness_val_full
        
        