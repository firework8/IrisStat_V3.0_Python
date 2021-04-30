import os
import sys
import numpy as np
from PIL import Image 
import time
from tqdm import tqdm
import scipy.io as scio
import math
import matplotlib
import matplotlib.pyplot as plt 

def sub2ind(array_shape, rows, cols):
    return (cols-1)*array_shape[0] + rows

def BitShift(code, bit):
    A = []
    for i in range(code.shape[2]):
        A.append(code[:,:,i])
    if bit < 0:
        # right shift
        bit = -bit
        for i in range(bit):
            A.insert(0,A.pop())
    else :
        # left shift
        for i in range(bit):
            A.insert(len(A),A[0])
            A.remove(A[0])
    code_shifted = np.zeros([code.shape[0],code.shape[1],code.shape[2]])
    for j in range(code.shape[2]):
        code_shifted[:,:,j]=A[j]
    return code_shifted

def compute_iriscode_sim( path_to_code, path_to_mask, 
    ext='*.png', shift_bit = range(-15,16), points=[], valid_img_list='') :
       
    # Compute similarity matrix from iriscodes
    # input
    # path_to_code: folders where iriscodes are stored, N samples
    # path_to_mask: folders where irismasks are stored,
    # N masks correlated to iris, the filenames should be the same
    # ext: extensions of iris codes and masks, should be the same, as '*.png'
    # shitf_bit: left and right shitf bits to alleviate eye rotation
    # sH: times of rows of the iris code to the iris mask, mostly set to 1 by default
    # sW: times of columns of the iris code to the iris mask, mostly set to 1 by default
    # output
    # classes: C*S, classes vector in string (or char rows, maximum length S )
    # sim: N*N similarity matrix
    # label: N*1 vector,
    # each entry means the class index correlated to *classes* that each sample belongs to.
    # Authors: Yunlong Wang, 2020.07.24
    # further optimized by Yunlong Wang, 2020.07.27
    # 1. split the label from filename, first filled into a cell array,
    # then transferred to a string array A
    # 2. Use built-in 'unique' function to return the unique rows of the string
    # array as *classes*. Index vector are just the prescribed *labels*, so
    # that A = classes(labels)
    # 3. support the size of the iris code is multiple times of the iris
    # mask,repeat the mask to match the size of the iris code
    # 4. support selection of points from on the iriscode
    
    # read data
    if len(valid_img_list) == 0 :
        fileNames_code = []
        for root, dirs, files in os.walk(path_to_code):
            for name in files:
                fileNames_code.append(name)
    else :
        with open(valid_img_list, 'r') as f:
            fileNames_code = f.readlines()
            for i in range(0, len(fileNames_code)):
                fileNames_code[i] = fileNames_code[i].strip()
    n = len(fileNames_code)
    if n == 0 :
        print('NO images in the path.')
        sys.exit()
    id = []
    for i in range(0, len(fileNames_code)):
        thenames = fileNames_code[i].split('_')
        id.append(thenames[0]) 
    classes = list(set(id))
    classes.sort()
    labels = []
    for i in range(0, len(id)):
        labels.append(classes.index(id[i]))

    I = Image.open(path_to_code+fileNames_code[0])   
    first_img = np.array(I)
    H = first_img.shape[0]
    W = first_img.shape[1]
    del first_img
    del I
    codes = np.zeros([n,H,W]).astype(np.float16)
    masks = np.zeros([n,H,W]).astype(np.float16)
    the_points_mul = []
    
    # Transfer Codes and Masks
    for i in tqdm(range(n), desc='Transfer Codes and Masks'):
        codeFile = path_to_code + fileNames_code[i]
        if not os.path.exists(codeFile):
            print('NO such iris code file '+codeFile)
            sys.exit()
        else: 
            if len(points) == 0 : 
                img1 = Image.open(codeFile)
                code = np.array(img1)/255
            else:
                img1 = matplotlib.image.imread(codeFile)
                code = np.array(img1)
            cH = code.shape[0]
            cW = code.shape[1]

        maskFile = path_to_mask + fileNames_code[i]
        if not os.path.exists(maskFile):
            print('NO such iris mask file '+codeFile)
            mask = np.ones([cH,cW])
            mH = cH
            mW = cW
        else:
            if len(points) == 0 : 
                img2 = Image.open(maskFile)
                mask = np.array(img2)/255
            else:
                img2 = matplotlib.image.imread(maskFile)
                mask = np.array(img2)
            mH = mask.shape[0]
            mW = mask.shape[1]
        sH = int(cH / mH)
        sW = int(cW / mW)
        if sH != int(sH) or sW != int(sW) or H != cH or W!= cW :
            print('Size not match!')
            sys.exit()
        
        codes[i,:,:] = code
        masks[i,:,:] = np.tile(mask, (sH,sW)) 
        # Generate the multpile times of points according to the size of
        # iriscode
        if i == n-1:
            if len(points) != 0 :
                points_indx = np.zeros([1,points.shape[1]])
                for i in range(points.shape[1]):
                    points_indx[0,i] = sub2ind([mH, mW],points[0,i],points[1,i])
                # number of points
                nump = points_indx.shape[1]
                # stride of one batch
                stride = mH * mW 
                points_mul = np.zeros([1,points_indx.shape[1]*sH*sW])
                points_mul = np.tile(points_indx, (1, sH*sW))
                for h in range(1,sH+1):
                    for w in range(1,sW+1):
                        batch = (h-1)*sW + w
                        start_p = (batch-1)*nump + 1
                        stop_p = batch*nump
                        batch_stride = stride * ((w-1)*sH+h-1)
                        for ki in range(points_indx.shape[1]):
                            points_mul[0,start_p-1+ki] = points_indx[0][ki] + batch_stride
                the_points_mul.append(points_mul)

    codes[codes>0.5] = 1
    codes[codes<0.5] = -1
    masks[masks>0.5] = 1
    masks[masks<0.5] = 0
    
    # distribution of labels
    high = max(labels)+1
    hist = np.zeros(high)
    for i in range(len(labels)):
        hist[labels[i]] = hist[labels[i]] + 1
    
    # get similarities
    sims_shift = np.zeros([len(shift_bit), n, n])
    codes = codes.astype(np.int16)
    masks = masks.astype(np.int16)

    codes_masked = (codes*masks).reshape(n, H*W)
    masks_reshaped = masks.reshape(n, H*W)

    if len(points) != 0 :
        del codes_masked
        del masks_reshaped
        codes_masked1 = (codes*masks).reshape(n, H*W, order="F")
        masks_reshaped1 = masks.reshape(n, H*W, order="F")
        points_mul = the_points_mul[0]
        codes_masked = np.zeros([n,points_mul.shape[1]])
        masks_reshaped = np.zeros([n,points_mul.shape[1]])
        for i in range(n):
            for j in range(points_mul.shape[1]):
                codes_masked[i,j] = codes_masked1[i,int(points_mul[0,j]-1)]
                masks_reshaped[i,j] = masks_reshaped1[i,int(points_mul[0,j]-1)]
        del codes_masked1
        del masks_reshaped1
    

    for i in tqdm(range(len(shift_bit)), desc='Get Similarities'):
    
        bit = shift_bit[i]
        if len(points) == 0 :
            codes_shifted = BitShift(codes, bit).reshape([n,H*W])
            masks_shifted = BitShift(masks, bit).reshape([n,H*W])
        else :
            codes_shifted = BitShift(codes, bit).astype(np.int16).reshape([n,H*W], order="F")
            masks_shifted = BitShift(masks, bit).astype(np.int16).reshape([n,H*W], order="F")
            codes_shifted1 = np.zeros([n,points_mul.shape[1]])
            masks_shifted1 = np.zeros([n,points_mul.shape[1]]) 
            for ki in range(n):
                for kj in range(points_mul.shape[1]):
                    codes_shifted1[ki,kj] = codes_shifted[ki,int(points_mul[0,kj]-1)]
                    masks_shifted1[ki,kj] = masks_shifted[ki,int(points_mul[0,kj]-1)]
            codes_shifted = codes_shifted1
            masks_shifted = masks_shifted1

        codes_shifted_masked = codes_shifted * masks_shifted
        
        mul_codes = np.dot(codes_masked,codes_shifted_masked.conj().T)
        mul_masks = np.dot(masks_reshaped,masks_shifted.conj().T) 

        # modified by Yunlong Wang, elminate bugs caused by 0-divided, add a
        # small epsilon 1e-6
        sim_shift = (mul_codes + mul_masks) / (2*mul_masks+1e-6);
        sims_shift[i,:,:] = sim_shift

    sim = np.max(sims_shift, axis=0).reshape(n, n).astype(np.float32)

    return classes, sim, labels

def compute_vector_sim(code_label_matfile):

    classes = []
    sim = []
    labels = []
    print("This function ws not implemented")
    sys.exit()

    # data = scio.loadmat(code_label_matfile)
    # if (not data.has_key('features')) or (not data.has_key('labels')):
    #     print('No *features* or *labels* loaded in, incorrect file or no such variables.')
    #     sys.exit()
    # features = data['features'] 
    # labels = data['label'] 
    # if features.shape[0] != labels.shape[0]:
    #     print('Size mismatch between *features* and *lables*')
    #     sys.exit()
    # n = labels.shape[0]
    # if n == 0 :
    #     print('NO images in the path.')
    #     sys.exit()
    # labels = np.squeeze(labels)

    # # get classes and labels assigned to each sample
    # id = labels.tolist()
    # classes = list(set(id))
    # classes.sort()
    # out_labels = []
    # for i in range(0, len(id)):
    #     out_labels.append(classes.index(id[i]))

    # # normalize the feature matrix in row 
    # for i in range(features.shape[0]):
    #     features[i,:] = features[i,:] / np.linalg.norm( x = features[i,:] )
    
    # # cosine similarity, ranges [-1,1]
    # sim = np.dot(features,features.conj().T)

    # # rescale to [0,1]
    # sim = (sim+1) / 2

    return classes, sim, labels

def compute_om_sim(path_to_code, path_to_mask, ext, rotation_angle_num, lobeDis):
    
    classes = []
    sim = []
    labels = []
    print("This function ws not implemented")
    sys.exit()

    return classes, sim, labels 

def IdentiACC(sim, labels):
    # updated by Yunlong Wang, 2021.01.18
    # added CMC curve

    # check dimensions
    print('Start to check dimensions')
    if not (sim.shape[0] == sim.shape[1] and sim.shape[0] == len(labels)):
        print('Dimension not in accord with each other. ')
        sys.exit()
    row, col = np.diag_indices_from(sim)
    sim[row,col] = -1

    N = sim.shape[0]
    index = np.zeros([sim.shape[0],sim.shape[1]])
    sim_sorted = np.zeros([sim.shape[0],sim.shape[1]])
    for i in range(sim.shape[0]):
        A = sim[i,:].tolist()
        sim_sorted[i,:] = np.array([v for i,v in sorted(enumerate(A), key=lambda x:(x[1],-x[0]), reverse=True)])
        index[i,:] = np.array([i for i,v in sorted(enumerate(A), key=lambda x:(x[1],-x[0]), reverse=True)])

    thelabels = np.array(labels).reshape(N,1)
    sorted_labels = np.zeros([sim.shape[0],sim.shape[1]])
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            sorted_labels[i,j] = labels[int(index[i,j])]
    sub_labels = sorted_labels - np.tile(thelabels, (1, N))
    # Calculate rank 1 accuracy
    acc_rank1 = 1 - len(sub_labels[:,0].nonzero()[0])/N
    
    # Calculate rank 5 accuracy, former vesion before 2020.07.22
    nrof_rank5 = 0
    for n in range(N) :
        if not len(sub_labels[n,0:5].nonzero()[0]) == 5 :
            nrof_rank5 = nrof_rank5 +1
    acc_rank5  = nrof_rank5 / N
    
    # Calculate rank 10 accuracy
    nrof_rank10 = 0
    for n in range(N) :
        if not len(sub_labels[n,0:10].nonzero()[0]) == 10 :
            nrof_rank10 = nrof_rank10 +1
    acc_rank10  = nrof_rank10 / N

    # Plot the e Cumulative Match Characteristic (CMC) curve
    # of Rank-N accuracy, edited by Yunlong Wang, 2021.01.18
    # 50 points of CMC curve by default
    cmc_curve = np.zeros([50,2])
    for p in range(cmc_curve.shape[0]) :
        cmc_curve[p,0] = p+1
        for n in range(N) :
            if not len(sub_labels[n,0:p+1].nonzero()[0]) == p+1 :
                cmc_curve[p,1] = cmc_curve[p,1] +1
        cmc_curve[p,1] = cmc_curve[p,1] / N
    
    return acc_rank1, acc_rank5, acc_rank10, cmc_curve

def VerfiACC(sim, labels, output_path , algo_name, det_resolution=100000) :
       
    # Compute verification accuracy of iris feature extraction methods
    # input
    # sim: N*N similarity matrix
    # labels: N*1, each entry means the class index that the sample belongs to.
    # det_resolution: the number of sampling points of FNMR and FMR
    # output
    # eer: Equal Error Rates
    # fnmr_fmr: pairs of FNMR @ FMR = 10^(-1*A), A= 1,2,3,4,5...
    # fnmr_o: resolution-restirted False Non-Match Rate
    # fmr_o: resolution-restirted False Match Rate
    # Authors: Yunlong Wang, 2020.07.24
    
    print('Compute verification accuracy of iris feature extraction methods')
    # check dimensions
    if not (sim.shape[0] == sim.shape[1] and sim.shape[0] == len(labels)):
        print('Dimension not in accord with each other. ')
        sys.exit()
    thelabels = np.array(labels).reshape(len(labels),1)
    label_value = np.tile(thelabels, (1, len(labels)))-np.tile(thelabels.conj().T, (len(labels), 1))
    label_value[label_value==0] = 0
    label_value[label_value!=0] = 1
    newshape = label_value.shape[0]
    label_value = np.ones([newshape,newshape])-label_value
    sample_matx = (np.ones([newshape,newshape]) - np.tril(np.ones([newshape,newshape]))).astype(np.bool_)
        
    # Python indexes by line
    # Matlab indexes by row        
    # sim_list = np.array(sim[sample_matx]).conj().T
    # labels_list = np.array(label_value[sample_matx]).conj().T.astype(np.bool_)
    sim_list = []
    labels_list = []
    for j in range(sample_matx.shape[1]):
        for i in range(sample_matx.shape[0]):
            if sample_matx[i,j] :
                sim_list.append(sim[i,j])
                labels_list.append(label_value[i,j])
            else :
                i = sample_matx.shape[0]-1
    
    d_indx =[]
    eer=[]
    fnmr_fmr =[]
    fmr = []
    print("Calculate EER, fnmr_fmr, fnr, fpr")
    # Calculate EER, fnmr_fmr, fnr, fpr
    d_indx, eer,fnmr_fmr,fnmr,fmr = EER(sim_list,labels_list,output_path,algo_name)

    # To save storage space, restrict the resolution of FNR and FPR
    x_ax = []
    if fnmr.shape[1]>det_resolution:
        num = 0
        for i in range(det_resolution) :
            x_ax.append(math.ceil(num))
            num = num + fnmr.shape[1]/det_resolution
    else:
        x_ax = range(0,fnmr.shape[1])
    fnmr_o = np.zeros([1,len(x_ax)])
    fmr_o = np.zeros([1,len(x_ax)])
    for j in range(fnmr_o.shape[1]):
        fnmr_o[0,j] = fnmr[0,int(x_ax[j])]
        fmr_o[0,j] = fmr[0,int(x_ax[j])]

    return d_indx, eer,fnmr_fmr,fnmr_o,fmr_o

def EER(dissim,label1,output_path,algo_name):

    # modified by Yunlong Wang, 20200817
    # decidability index 
    
    thesim = np.array(dissim).reshape(1,len(dissim))
    label = np.array(label1).reshape(1,len(label1))
    sim = np.ones([thesim.shape[0],thesim.shape[1]]) - thesim
    thesim1 = []
    thesim2 = []
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j]>0  :
                thesim1.append(sim[i,j])
            else :
                thesim2.append(sim[i,j])

    d_indx = abs(np.mean(thesim1)-np.mean(thesim2))/math.sqrt((np.var(thesim1)+np.var(thesim2))/2)

    gen_n, bins1 ,patches1= plt.hist(thesim1,256) 
    imp_n, bins2 ,patches2= plt.hist(thesim2,256) 
    gen_x = []
    for i in range(len(bins1)):
        if i != len(bins1)-1:
            gen_x.append((bins1[i]+bins1[i+1])/2)   
    imp_x = []
    for i in range(len(bins2)):
        if i != len(bins2)-1:
            imp_x.append((bins2[i]+bins2[i+1])/2)
    plt.cla()

    plt.figure(1)
    plt.axis([0.0,1.0,0.0,0.1]) 

    x1 = np.append(np.array(gen_x),np.array(gen_x)[::-1])
    y1 = np.append(np.array(gen_n)/sum(gen_n),np.zeros([1,len(gen_x)]))
    plt.fill(x1, y1 , color = "r", alpha = 0.3)
    x2 = np.append(np.array(imp_x),np.array(imp_x)[::-1])
    y2 = np.append(np.array(imp_n)/sum(imp_n),np.zeros([1,len(imp_x)]))
    plt.fill(x2, y2 , color = "g", alpha = 0.3)
    
    plt.legend(['Genuines','Impostors'])
    plt.plot(gen_x, gen_n/sum(gen_n), 'k');
    plt.plot(imp_x, imp_n/sum(imp_n), 'k');
    plt.xlabel('Dissimilarity',{'size':16})
    plt.ylabel('Frequency',{'size':16})
    title_str = '%s %.4f'%(algo_name, d_indx)
    plt.title(title_str,{'size':20}) 
    plt.savefig(output_path+algo_name+'_genuine_impostor_distribution.png')
    
    indx = np.zeros([1,len(dissim)])
    for i in range(thesim.shape[0]):
        A = sim[i,:].tolist()
        indx[i,:] = np.array([i for i,v in sorted(enumerate(A), key=lambda x:(x[1],x[0]))])

    sort_label = np.zeros([1,len(dissim)])
    neg_sample = 0
    pos_sample = 0
    sort1 = np.zeros([1,len(dissim)])
    sort2 = np.zeros([1,len(dissim)])

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            sort_label[i,j] = label[i,int(indx[i,j])]
            if sort_label[i,j] == 1 :
                sort1[i,j] = 0
                sort2[i,j] = 1
                pos_sample = pos_sample+1
            elif sort_label[i,j] == 0:
                sort1[i,j] = 1
                sort2[i,j] = 0
                neg_sample = neg_sample+1

    tpr = np.append(np.array([0]),np.cumsum(sort2)).reshape(1,len(dissim)+1) / pos_sample 
    fpr = np.append(np.array([0]),np.cumsum(sort1)).reshape(1,len(dissim)+1) / neg_sample
    fnr = np.ones([1,len(tpr)]) - tpr
    tnr = np.ones([1,len(fpr)]) - fpr

    acc_level = math.floor(math.log(neg_sample)/math.log(10))

    fnmr_fmr = []
    for lv_cnt in range(acc_level) :
        max1 = 0
        for j in range(fpr.shape[1]):
            if fpr[0,j] <= 10**(-1*(lv_cnt+1)):
                if j > max1 :
                    max1 = j
        fnmr_fmr.append(fnr[0,max1])
    thenp = np.zeros([1,len(dissim)+1])
    max2 = 0
    for j in range(fpr.shape[1]):
        if fnr[0,j] > fpr[0,j]:
            if j > max2 :
                max2 = j
    eer = fnr[0,max2]
    
    return d_indx, eer,fnmr_fmr,fnr,fpr

def draw_DET_curve( fnmr, fmr, output_path, algo_name, det_resolution = 100000 ) :
    
    x_ax = []
    if fnmr.shape[1]>det_resolution:
        num = 0
        for i in range(det_resolution) :
            x_ax.append(math.ceil(num))
            num = num + fnmr.shape[1]/det_resolution
    else:
        x_ax = range(0,fnmr.shape[1])
    
    fmr_o = np.zeros([1,len(x_ax)])
    fnmr_o = np.zeros([1,len(x_ax)])
    for j in range(fnmr_o.shape[1]):
        fmr_o[0,j] = fmr[0,int(x_ax[j])]
        fnmr_o[0,j] = fnmr[0,int(x_ax[j])] 
     
    plt.figure(2)
    plt.semilogx(fmr_o.tolist()[0],fnmr_o.tolist()[0],color="blue")
    tt = np.logspace(-5,0,1000)
    plt.semilogx(tt,tt,color=[0.7,0.7,0.7])
    plt.xlabel('FMR')
    plt.ylabel('FNMR')
    plt.title('DET curve') 
    plt.savefig(output_path+algo_name+'_DET_curve.png')
    
    return 

def draw_CMC_curve( cmc_curve_point , output_path, algo_name ) :
    
    plt.figure(3)
    plt.plot(cmc_curve_point[:,0], cmc_curve_point[:,1], 'o-');
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    plt.title('CMC curve of Rank-N accuracy') 
    plt.savefig(output_path+algo_name+'_CMC_curve.png')
    
    return