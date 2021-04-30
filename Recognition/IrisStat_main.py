"""
Compute identification and verification performance of iris feature
extraction algorithms

Written by RenMin 20191016
Modified by Yunlogn Wang 20200617
Again updated by Jianze Wei 20200722
Re-organized by Yunlong Wang 20200724
further optimized by Yunlong Wang 20200727
Extended by Yunlong Wang 20200817
Rewritten in Python by Hongda Liu 2021.04.16
Add Decidability Index to verification accuracy

i. Configure the setting in a xxx_config.m script file, for instance, 'UniNet_config.m'
ii. Support two mode, i.e. iris features stored in 'vecotr' or 'iriscode'
iii. Classes and labels can be automatically split and extracted from
list of filenames or string arrays, define the manner you get the
arropriate label from the corresponding filename just witn one line in
the function *compute_iriscode_sim*, 
whereas getting the lables from string arrays are without this
defination
iv. The resolution of FMR and FNMR is properly set to save memory space
v. A complete set of performance indices including 
['acc_rank1','acc_rank5','acc_rank10','d_indx','eer', 'fnmr_fmr', 'fmr', 'fnmr']
are calculated and saved in a .mat file with its filename indicating the test algorthm
"""
# # Setting in a script .m file 
# from config.LiborMasek_config import *
# from config.LiborMasek_rubbersheet_config import *
# from config.LiborMasek_GT_circularHough_config import *
# from config.LiborMasek_IrisParseNet_circularHough_config import *

# # USIT V3.0.0 QSW
# from config.USIT_qsw_config import *
# from config.USIT_qsw_rubbersheet_config import *
# from config.USIT_qsw_GT_circularHough_config import *
# from config.USIT_qsw_IrisParseNet_circularHough_config import *

# # USIT V3.0.0 Lg
# from config.USIT_lg_config import *
# from config.USIT_lg_rubbersheet_config import *
# from config.USIT_lg_GT_circularHough_config import *
# from config.USIT_lg_IrisParseNet_circularHough_config import *

# # UniNet
# from config.UniNet_config import *
# from config.UniNet_rubbersheet_config import *
# from config.UniNet_GT_circularHough_config import *
# from config.UniNet_IrisParseNet_circularHough_config import *

# # UniNet_WithMaskNet
# from config.UniNet_WithMaskNet_config import *
# from config.UniNet_rubbersheet_config import *
# from config.UniNet_GT_circularHough_config import *
# from config.UniNet_IrisParseNet_circularHough_config import *

# # Maxout CNN
# from config.MaxoutCNN_config import *
# from config.MaxoutCNN_rubbersheet_config import *
# from config.MaxoutCNN_GT_circularHough_config import *
# from config.MaxoutCNN_IrisParseNet_circularHough_config import *

# # GraphNet
# from config.GraphNet_config import *
# from config.GraphNet_rubbersheet_config import *
# from config.GraphNet_GT_circularHough_config import *
# from config.GraphNet_IrisParseNet_circularHough_config import *

# # OSIris
# from config.OSIris_config import *
# from config.OSIris_rubbersheet_config import *
# from config.OSIris_GT_circularHough_config import *
# from config.OSIris_IrisParseNet_circularHough_config import *

# # AFNet
# from config.AFNet_config import *
# from config.AFNet_rubbersheet_config import *
# from config.AFNet_GT_circularHough_config import *
# from config.AFNet_IrisParseNet_circularHough_config import *

# # OM
# from config.simpleOM_config import *
# from config.simpleOM_rubbersheet_config import *
# from config.simpleOM_GT_circularHough_config import *
# from config.simpleOM_IrisParseNet_circularHough_config import *

# # LightCNN
# from config.LightCNN_config import *
               
from config.USIT_qsw_config import *
# from config.OSIris_config import *
# from config.USIT_lg_config import *
from functions import *
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np

# Check if out/xxx_stat.mat already exists, load in the variables
if os.path.exists(save_matfile) and not overwrite :
    print('********************************')
    print('Results already exists, load in...')
    data = scio.loadmat(save_matfile)
    acc_rank1 = data['acc_rank1']
    acc_rank5 = data['acc_rank5']
    acc_rank10 = data['acc_rank10']
    cmc_curve_point = data['cmc_curve_point']
    d_indx = data['d_indx']
    eer = data['eer']
    fnmr_fmr_list = data['fnmr_fmr']
    fnmr_fmr = fnmr_fmr_list[0]
    fmr = np.array(data['fmr'])
    fnmr = np.array(data['fnmr'])
    print('********************************')
else :
    if feat_mode == 'iriscode' :
        classes, sim, labels = compute_iriscode_sim(path_to_code, path_to_mask, ext, shift_bit, points, valid_image_list)
    elif feat_mode == 'vector' :
        classes, sim, labels = compute_vector_sim(code_label_matfile)
    elif feat_mode == 'simpleOM' :
        classes, sim, labels = compute_om_sim(path_to_code, path_to_mask, ext, rotation_angle_num, lobeDis)
    else :
        print('NOT support feature mode.');
        sys.exit(0);
    
    cut_off_rule = '-'
    for i in range(40):
        cut_off_rule += '-'
    print(cut_off_rule)
    print('%d classes, %d samples in total.\n'%(len(classes), sim.shape[0]))
    print(cut_off_rule)

    # Calculate Identification Accuracy, rank 1, 5, 10
    acc_rank1,acc_rank5,acc_rank10, cmc_curve_point = IdentiACC(sim, labels)
    # Calculate Verificaiton Accuracy, EER, FNMR_FMR, FNR, FPR
    d_indx, eer,fnmr_fmr,fnmr,fmr = VerfiACC(sim, labels, output_path ,algo_name ,det_resolution )

# Draw DET curve 
draw_DET_curve(fnmr, fmr, output_path ,algo_name)
# Draw CMC curve, 2021.01.18 by Yunlong Wang
draw_CMC_curve(cmc_curve_point, output_path ,algo_name)

# print out the statistics of performance indices
print('********************************')
print('Identification Accuracy')
print('--------------------------------')
print('Rank 1: %.4f' % acc_rank1)
print('Rank 5: %.4f' % acc_rank5)
print('Rank 10: %.4f' % acc_rank10)
print('--------------------------------')
print('Verification Accuracy')
print('Decidability index: %.4f' % d_indx)
print('EER: %.4f'%eer)
print('FNMR@FMR=10^-1: %.4f'%fnmr_fmr[0])
print('FNMR@FMR=10^-2: %.4f'%fnmr_fmr[1])
print('FNMR@FMR=10^-3: %.4f'%fnmr_fmr[2])
print('FNMR@FMR=10^-4: %.4f'%fnmr_fmr[3])
print('FNMR@FMR=10^-5: %.4f'%fnmr_fmr[4])
print('********************************')

# save the statistics of performance indices
scio.savemat(save_matfile,{'acc_rank1': acc_rank1,'acc_rank5': acc_rank5,'acc_rank10': acc_rank10,
                        'cmc_curve_point': cmc_curve_point,'d_indx': d_indx, 'eer': eer ,
                        'fnmr_fmr': fnmr_fmr,'fmr': fmr,'fnmr': fnmr})
