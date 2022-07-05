

"""
This is the main file for our Python implementation of CoHOG published in IEEE RAL 2020. The HOG_feature folder contains files that have been 
modified by us, however, the original rights for all employed libraries lie with the corresponding contributors and we thanks them for 
open-sourcing their implementations. For the rest of the libraries (skimage, scipy, matplotlib, cv2, numpy etc), please use standard methods of
installation for your IDE, if not already installed.

For any queries or problems, send me an email at mubarizzaffar@gmail.com. If you find this useful, please cite our paper titled 
"CoHOG: A Light-weight, Compute-efficient and Training-free Visual Place Recognition Technique for Changing Environments".

Created on Thu Jan  3 11:18:57 2019

@author: mubariz
"""

import cv2
import numpy as np
import os   #Imported here if your image reading methodolgy uses os.listdirectory sort of implementation.
from Hog_feature.Hog_feature.hog import initialize
from Hog_feature.Hog_feature.hog import extract
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import csv  
import time
from scipy import linalg

####################### PARAMETERS #########################
magic_width=512
magic_height=512
cell_size=16  #HOG cell-size
bin_size=8  #HOG bin size
image_frames=1  #1 for grayscale, 3 for RGB
descriptor_depth=bin_size*4*image_frames # x4 is here for block normalization due to nature of HOG
ET=0.65 # Entropy threshold, vary between 0-1.
GT=0.91
SN=3
AP=0.0
BP=0.5

total_Query_Images=198
query_index_offset=0
total_Ref_Images=198
ref_index_offset=0

total_no_of_regions=int((magic_width/cell_size-1)*(magic_width/cell_size-1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix=np.zeros([total_no_of_regions,total_no_of_regions],dtype=np.float32)
d1d2matches_maxpooled=np.zeros([total_no_of_regions],dtype=np.float32)
d1d2matches_regionallyweighted=np.zeros([total_no_of_regions],dtype=np.float32)

matched_local_pairs=[]
ref_desc=[]
ref_desc_s = []

ref = 0
num = 0
num_regions = []
rep_sub = []
subsets_index = [ ]
############################################################

dataset_name='GardensSmall'   #Please modify according to your needs
save_visual_matches_dir='/home/airs/'+dataset_name+'/'
# os.makedirs(save_visual_matches_dir)     # If the directory doesn't exist already.


query_directory='/home/airs/day_left/query/'       #Please modify according to your needs
ref_directory='/home/airs/day_left/ref/'     #Please modify according to your needs

out_directory='/home/airs/entropy_extracted_regions/'        # Please modify. This directory is for visualizing the entropy-based regions extraction.


def save_visual_matches(query,GT,retrieved):                     #For visualizing the correct and incorrect matches
    query_img=cv2.imread(query_directory+get_query_image_name(query))
    query_img=cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    gt_img=cv2.imread(ref_directory+get_ref_image_name(GT))
    gt_img=cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    retrieved_img=cv2.imread(ref_directory+get_ref_image_name(retrieved))
    retrieved_img=cv2.cvtColor(retrieved_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure()

    ax1 = fig.add_subplot(131)  
    plt.axis('off')
    ax1.imshow(query_img)
    ax2 = fig.add_subplot(132)
    plt.axis('off')
    ax2.imshow(retrieved_img)
    ax3 = fig.add_subplot(133)
    plt.axis('off')
    ax3.imshow(gt_img)
    ax1.title.set_text('Query Image')
    ax2.title.set_text('Retrieved Image')
    ax3.title.set_text('Ground-Truth')
    plt.show()
    
    fig.savefig(save_visual_matches_dir+str(query)+'.jpg',bbox_inches='tight')

def largest_indices_thresholded(ary):
    good_list=np.where(ary>=ET)
#    no_of_good_regions=len(good_list[0])
#     print(len(good_list))
#     print(len(good_list[0]))
        
#    if (no_of_good_regions<min_no_of_regions):
#        good_list=largest_indices(ary,back_up_regions)
    
    return good_list 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
      
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_query_image_name(j):
    k=str(j+query_index_offset)

    return k + '.jpg'

def get_ref_image_name(j):
    k=str(j+ref_index_offset)

    return k + '.jpg'

#@jit(nopython=False)
def conv_match_dotproduct(d1,d2,regional_gd,total_no_of_regions):            #Assumed aspect 1:1 here

    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs
    

    np.dot(d1,d2,out=d1d2dot_matrix)

    np.max(d1d2dot_matrix,axis=1,out=d1d2matches_maxpooled)               #Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled,regional_gd,out=d1d2matches_regionallyweighted)   # Weighting regional matches with regional goodness

    score=np.sum(d1d2matches_regionallyweighted)/np.sum(regional_gd)    #compute final match score
    # !!!!! score is similarity !!!!!
    return score

# build subsets of origin dataset, offline
while ref < 90:
    try:
        img_1 = cv2.imread(ref_directory + get_ref_image_name(ref), 0)
    except (IOError, ValueError) as e:
        img_1 = None
        print('Exception! \n \n \n \n', ref)

# get vector_1

    if (img_1 is not None):
        img_1 = cv2.resize(img_1, (magic_height, magic_width))
        height, width, angle_unit = initialize(img_1, cell_size, bin_size)
        vector_1 = extract()
        vector_1 = np.asfortranarray(vector_1.transpose(), dtype=np.float32)
        ################# Entropy Map ###############################

        img_gray = cv2.resize(img_as_ubyte(img_1), (100, 100))
        entropy_image = cv2.resize(entropy(img_gray, disk(5)), (magic_width, magic_height))

        ################# Finding Regions #####################
        local_goodness = np.zeros([int(magic_height / cell_size - 1), int(magic_width / cell_size - 1)],
                                  dtype=np.float32)

        for a in range(int(magic_height / cell_size - 1)):
            for b in range(int(magic_width / cell_size - 1)):
                local_staticity = 1  # Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a * cell_size:a * cell_size + 2 * cell_size,
                                       b * cell_size:b * cell_size + 2 * cell_size]) / (8 * (cell_size * 4 * cell_size))

                if (local_entropy >= ET):
                    local_goodness[a, b] = 1
                else:
                    local_goodness[a, b] = 0

        regional_goodness = local_goodness.flatten()
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions = np.sum(regional_goodness)
        if ref == 0:
            num_regions.append(no_of_good_regions)
        sub_length = 0
        flag = 0

    #########################################################################
    #########################################################################
        while flag == 0:
            try:
                img_2 = cv2.imread(ref_directory + get_ref_image_name(ref+sub_length+1), 0)
            except (IOError, ValueError) as e:
                img_2 = None
                print('Exception! \n \n \n \n', ref)
            confusion_vector=np.zeros(total_Ref_Images,dtype=np.float32)

            if (img_2 is not None):

                img_2=cv2.resize(img_2,(magic_height,magic_width))

                height,width,angle_unit=initialize(img_2, cell_size, bin_size)
                vector_2 = extract()
                vector_2=np.asfortranarray(vector_2,dtype=np.float32)

                ################# Entropy Map ###############################

                img_gray = cv2.resize(img_as_ubyte(img_2),(100,100))
                entropy_image=cv2.resize(entropy(img_gray, disk(5)),(magic_width,magic_height))

                ################# Finding Regions #####################
                local_goodness=np.zeros([int(magic_height/cell_size-1),int(magic_width/cell_size-1)],dtype=np.float32)

                for a in range (int(magic_height/cell_size-1)):
                    for b in range (int(magic_width/cell_size-1)):
                        local_staticity=1 #Disabling staticity here, can be accommodated in future by employing YOLO etc.
                        local_entropy = np.sum(entropy_image[a*cell_size:a*cell_size + 2*cell_size, b*cell_size:b*cell_size + 2*cell_size])/(8*(cell_size*4*cell_size))

                        if (local_entropy>=ET):
                            local_goodness[a,b]=1
                        else :
                            local_goodness[a,b]=0

                regional_goodness=local_goodness.flatten()
                regions = largest_indices_thresholded(local_goodness)
                no_of_good_regions=np.sum(regional_goodness)
                num_regions.append(no_of_good_regions)

                score=conv_match_dotproduct(vector_2,vector_1,regional_goodness,total_no_of_regions)
                # print(score, ref, ref + sub_length+1)
                sub_length += 1
                if score < GT:
                    flag = 1
                    tmp = []
                    for j in range(sub_length):
                        tmp.append(ref+j)
                    subsets_index.append(tmp)
                    ref = ref + sub_length
                    num += 1



rep_index = []
for rep in range(len(subsets_index)):
    reg_max = 0
    for j in range(len(subsets_index[rep])):
        if num_regions[subsets_index[rep][j]] > reg_max:
            reg_max = num_regions[subsets_index[rep][j]]
            index = subsets_index[rep][j]
    rep_index.append(index)

print(subsets_index)
print(num_regions)
print(len(subsets_index))
print(len(num_regions))
print(rep_index)


for ref in range(len(rep_index)):
    try:
        img_1 = cv2.imread(ref_directory+get_ref_image_name(rep_index[ref]), 0)
    except (IOError, ValueError) as e:
        img_1=None
        print('Exception! \n \n \n \n',ref)

    if (img_1 is not None):

        img_1=cv2.resize(img_1,(magic_height,magic_width))
        startencodetimer=time.time()
        height,width,angle_unit=initialize(img_1, cell_size, bin_size)
        vector_1 = extract()
        vector_1=np.asfortranarray(vector_1.transpose(),dtype=np.float32)
        ref_desc.append(vector_1)

for ref_1 in range(len(subsets_index)):
    tmp = []
    for ref_2 in range(len(subsets_index[ref_1])):
        try:
            img_1 = cv2.imread(ref_directory+get_ref_image_name(subsets_index[ref_1][ref_2]), 0)
        except (IOError, ValueError) as e:
            img_1=None
            print('Exception! \n \n \n \n',ref)

        if (img_1 is not None):
            img_1=cv2.resize(img_1,(magic_height,magic_width))
            startencodetimer=time.time()
            height,width,angle_unit=initialize(img_1, cell_size, bin_size)
            vector_1 = extract()
            vector_1=np.asfortranarray(vector_1.transpose(),dtype=np.float32)
            tmp.append(vector_1)
    ref_desc_s.append(tmp)



# query retrieval, online
for query in range(80):

    confusion_vector=np.zeros(total_Ref_Images,dtype=np.float32)
    try:
        img_2 = cv2.imread(query_directory+get_query_image_name(query), 0)
        img_2rgb=cv2.imread(query_directory+get_query_image_name(query))

#        img_2 = cv2.imread(query_directory+get_query_image_name(query))

    except (IOError, ValueError) as e:
        img_2=None
        print('Exception! \n \n \n \n')

    if (img_2 is not None):

        img_2=cv2.resize(img_2,(magic_height,magic_width))
        img_2rgb=cv2.resize(img_2rgb,(magic_height,magic_width))

        startencodetimer=time.time()

        height,width,angle_unit=initialize(img_2, cell_size, bin_size)
        vector_2 = extract()
        vector_2=np.asfortranarray(vector_2,dtype=np.float32)

        ################# Entropy Map ###############################
#        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_as_ubyte(img_2),(100,100))
        ent_time=time.time()
        entropy_image=cv2.resize(entropy(img_gray, disk(5)),(magic_width,magic_height))
        # print('Entropy Time:',time.time()-ent_time)


        ################# Finding Regions #####################
        local_goodness=np.zeros([int(magic_height/cell_size-1),int(magic_width/cell_size-1)],dtype=np.float32)

        for a in range (int(magic_height/cell_size-1)):
            for b in range (int(magic_width/cell_size-1)):
                local_staticity=1 #Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a*cell_size:a*cell_size + 2*cell_size, b*cell_size:b*cell_size + 2*cell_size])/(8*(cell_size*4*cell_size))


                if (local_entropy>=ET):
                    local_goodness[a,b]=1
                else :
                    local_goodness[a,b]=0

        regional_goodness=local_goodness.flatten()
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions=np.sum(regional_goodness)
        encodetime=time.time() - startencodetimer


        img_regions_display=cv2.cvtColor(img_2rgb, cv2.COLOR_BGR2RGB)

        for reg_index in range(len(regions[0])):

            cv2.rectangle(img_regions_display,(regions[1][reg_index]*cell_size,regions[0][reg_index]*cell_size),(regions[1][reg_index]*cell_size + cell_size,regions[0][reg_index]*cell_size + cell_size),(0,255,0),3)


        for ref in range(len(rep_index)):
            score=conv_match_dotproduct(vector_2,ref_desc[ref],regional_goodness,total_no_of_regions)
            # print(score, rep_index[ref])
            confusion_vector[ref] = score

        # save_visual_matches(query,query,rep_index[np.argmax(confusion_vector)])   # Uncomment this for saving visual samples of correctly and incorrectly matched images.
        top_k=SN
        average = []
        best = []
        best_match = []
        top_k_idx=confusion_vector.argsort()[::-1][0:top_k]
        for num1 in range(top_k):
            confusion_vector_2 = []
            for num2 in range(len(ref_desc_s[top_k_idx[num1]])):
                score = conv_match_dotproduct(vector_2,ref_desc_s[top_k_idx[num1]][num2],regional_goodness,total_no_of_regions)
                confusion_vector_2.append(score)
                # print("confusion_vector_2: ", confusion_vector_2, "   ", len(ref_desc_s[top_k_idx[num1]]))
            best_match_index = np.argmax(confusion_vector_2)
            # get best match and average similarity
            average.append(np.average(confusion_vector_2))
            best.append(np.max(confusion_vector_2))

            print("best match", subsets_index[top_k_idx[num1]][best_match_index])
            print("best match score", np.max(confusion_vector_2))
            print("average score", np.average(confusion_vector_2))


            best_match.append(subsets_index[top_k_idx[num1]][best_match_index])
            # save_visual_matches(query, query, subsets_index[top_k_idx[num1]][best_match_index])

        weighted_score = []
        for j in range(top_k):
            weighted_score.append(average[j]*AP+best[j]*BP)
        selected_seq = np.argmax(weighted_score)
        save_visual_matches(query, query, best_match[selected_seq])

