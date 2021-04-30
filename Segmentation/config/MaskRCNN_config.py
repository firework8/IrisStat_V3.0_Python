"""
Configurations to evaluate the segmentation method in MaskRCNN
"""
import os

global algo_name
algo_name = 'MaskRCNN'
outpath = 'out/'

# Set variables
image_folder='/home/yunlong.wang/BlurringIRISdatabase/image/'
circleGT_folder='/home/yunlong.wang/BlurringIRISdatabase/circle_params/'
circleGT_ext = '.ini'
segGT_folder= '/home/yunlong.wang/BlurringIRISdatabase/iris_mask_new/'
segGT_ext = '.png'
segRes_folder='/home/yunlong.wang/OpenSourceIris/MaskRCNN/masks_ori/'
segRes_prefix = []
segRes_suffix = '.png'

errImgPath = outpath + algo_name + '_errImgs/'
if not os.path.exists(errImgPath):
    os.mkdir(errImgPath)
errImgExt = '.png'

save_matfile = outpath + algo_name + '_stat.mat'
diary_file = outpath + algo_name + '_evalSegRes.txt'
# if not os.path.exists(diary_file):
#     os.system(r"touch {}".format(diary_file))

# select images from this list file
img_list_txt = '/home/yunlong.wang/BlurringIRISdatabase/imgList.txt'
