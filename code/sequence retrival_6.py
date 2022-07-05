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

####################### PARAMETERS #########################
magic_width=512
magic_height=512
cell_size=16  #HOG cell-size
bin_size=8  #HOG bin size
image_frames=1  #1 for grayscale, 3 for RGB
descriptor_depth=bin_size*4*image_frames # x4 is here for block normalization due to nature of HOG
ET=0.5# Entropy threshold, vary between 0-1.
ST=0.03# similarity threshold
SN= 10 # bsets number
PR_1=0.00# precision recall threshold
PR_2=0.00# precision recall threshold

total_Query_Images=190
query_index_offset=0
total_Ref_Images=190
ref_index_offset=0

ref = 0 # start index for reference
query = 50 # start index for query


#################### GLOBAL VARIABLES ######################

total_no_of_regions=int((magic_width/cell_size-1)*(magic_width/cell_size-1))
# variables for calculate similarity scores
d1d2dot_matrix=np.zeros([total_no_of_regions,total_no_of_regions],dtype=np.float32)
d1d2matches_maxpooled=np.zeros([total_no_of_regions],dtype=np.float32)
d1d2matches_regionallyweighted=np.zeros([total_no_of_regions],dtype=np.float32)
matched_local_pairs=[]

# array to store all the descriptors of images
ref_desc=[]
ref_desc_s = []
num = 0
num_regions = []
rep_sub = []
subsets_index = []  # 2d array that store index, subsets_index[no_of_subset][no_of_image_in_subset]
rep_index = []  # array of all representative image in each subset
top_k_idx = []
query_sequence_list = []
sequence_index = []
pre_score=1
final_results = []
retrieved_num = 0
TP1 = 0
TP2 = 0
no_matching = 0
############################################################

dataset_name='GardensSmall'   #Please modify according to your needs
save_visual_matches_dir='./output'+dataset_name+'/'
# os.makedirs(save_visual_matches_dir)     # If the directory doesn't exist already.


query_directory = './dataset/dataset_4/ref/'  # Please modify according to your needs
ref_directory = './dataset/dataset_4/ref/'  # Please modify according to your needs

out_directory='./output'        # Please modify. This directory is for visualizing the entropy-based regions extraction.


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

def read_ref_images(ref):
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

        local_goodness = np.zeros([int(magic_height / cell_size - 1), int(magic_width / cell_size - 1)],
                                  dtype=np.float32)
        sum_entropy = 0
        sum_num = 0
        for a in range(int(magic_height / cell_size - 1)):
            for b in range(int(magic_width / cell_size - 1)):
                local_staticity = 1  # Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a * cell_size:a * cell_size + 2 * cell_size,
                                       b * cell_size:b * cell_size + 2 * cell_size]) / (8 * (cell_size * 4 * cell_size))
                sum_entropy += local_entropy
                sum_num += 1
                if (local_entropy >= ET):
                    local_goodness[a, b] = 1
                else:
                    local_goodness[a, b] = 0

        regional_goodness = local_goodness.flatten()
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions = np.sum(regional_goodness)

        return vector_1,regional_goodness,no_of_good_regions

def read_query_images(ref):
    try:
        img_1 = cv2.imread(query_directory + get_ref_image_name(ref), 0)
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

        local_goodness = np.zeros([int(magic_height / cell_size - 1), int(magic_width / cell_size - 1)],
                                  dtype=np.float32)
        sum_entropy = 0
        sum_num = 0
        for a in range(int(magic_height / cell_size - 1)):
            for b in range(int(magic_width / cell_size - 1)):
                local_staticity = 1  # Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a * cell_size:a * cell_size + 2 * cell_size,
                                       b * cell_size:b * cell_size + 2 * cell_size]) / (8 * (cell_size * 4 * cell_size))
                sum_entropy += local_entropy
                sum_num += 1
                if (local_entropy >= ET):
                    local_goodness[a, b] = 1
                else:
                    local_goodness[a, b] = 0

        regional_goodness = local_goodness.flatten()
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions = np.sum(regional_goodness)

        return vector_1,regional_goodness,no_of_good_regions

def build_map():
    for ref in range(len(rep_index)):
        try:
            img_1 = cv2.imread(ref_directory + get_ref_image_name(rep_index[ref]), 0)
        except (IOError, ValueError) as e:
            img_1 = None
            print('Exception! \n \n \n \n', ref)

        if (img_1 is not None):
            img_1 = cv2.resize(img_1, (magic_height, magic_width))
            startencodetimer = time.time()
            height, width, angle_unit = initialize(img_1, cell_size, bin_size)
            vector_1 = extract()
            vector_1 = np.asfortranarray(vector_1.transpose(), dtype=np.float32)
            ref_desc.append(vector_1)

    for ref_1 in range(len(subsets_index)):
        tmp = []
        for ref_2 in range(len(subsets_index[ref_1])):
            try:
                img_1 = cv2.imread(ref_directory + get_ref_image_name(subsets_index[ref_1][ref_2]), 0)
            except (IOError, ValueError) as e:
                img_1 = None
                print('Exception! \n \n \n \n', ref)

            if (img_1 is not None):
                img_1 = cv2.resize(img_1, (magic_height, magic_width))
                startencodetimer = time.time()
                height, width, angle_unit = initialize(img_1, cell_size, bin_size)
                vector_1 = extract()
                vector_1 = np.asfortranarray(vector_1.transpose(), dtype=np.float32)
                tmp.append(vector_1)
        ref_desc_s.append(tmp)


#############################################################################

while query < total_Query_Images:
    print("new seq!!!!!!!")
    query_sequence_list = []
    curr_seq = []
    tmp =[]
    vector_3, regional_goodness_3, no_of_good_regions_3 = read_query_images(query)
    query_sequence_list.append(vector_3)
    tmp.append(query)

    # first previous image
    sub_length = 0
    query_flag = 0
    pre_score = 1
    init_score = 0
    fore_query = query - 1
    while query_flag == 0 and fore_query > 0:
        vector_4, regional_goodness_4, no_of_good_regions_4 = read_query_images(fore_query)
        score = conv_match_dotproduct(vector_3.transpose(), vector_4, regional_goodness_3,
                                          total_no_of_regions)
        # print("score1",score)
        fore_query -= 1
        if (score > pre_score) or (score < init_score - ST):
            query_sequence_list.append(vector_4)
            tmp.append(fore_query + 1)
            query_flag = 1
        else:
            pre_score = score
            if init_score < score:
                init_score = score

    # further previous image
    sub_length = 0
    query_flag = 0
    pre_score = 1
    init_score = 0
    fore_query = fore_query
    while query_flag == 0 and fore_query > 0:
        vector_5, regional_goodness_5, no_of_good_regions_5 = read_query_images(fore_query)
        score = conv_match_dotproduct(vector_4.transpose(), vector_5, regional_goodness_4,
                                      total_no_of_regions)
        # print("score2", score)
        fore_query -= 1
        if (score > pre_score) or (score < init_score - ST):
            query_sequence_list.append(vector_5)
            tmp.append(fore_query + 1)
            query_flag = 1
        else:
            pre_score = score
            if init_score < score:
                init_score = score


    sub_length = 0
    query_flag = 0
    pre_score = 1
    init_score = 0
    fore_query = fore_query
    while query_flag == 0 and fore_query > 0:
        vector_6, regional_goodness_6, no_of_good_regions_6 = read_query_images(fore_query)
        score = conv_match_dotproduct(vector_5.transpose(), vector_6, regional_goodness_5,
                                      total_no_of_regions)
        # print("score2", score)
        fore_query -= 1
        if (score > pre_score) or (score < init_score - ST):
            query_sequence_list.append(vector_6)
            tmp.append(fore_query + 1)
            query_flag = 1
        else:
            pre_score = score
            if init_score < score:
                init_score = score



    sub_length = 0
    query_flag = 0
    pre_score = 1
    init_score = 0
    fore_query = fore_query
    while query_flag == 0 and fore_query > 0:
        vector_7, regional_goodness_7, no_of_good_regions_7 = read_query_images(fore_query)
        score = conv_match_dotproduct(vector_6.transpose(), vector_7, regional_goodness_6,
                                      total_no_of_regions)
        # print("score2", score)
        fore_query -= 1
        if (score > pre_score) or (score < init_score - ST):
            query_sequence_list.append(vector_7)
            tmp.append(fore_query + 1)
            query_flag = 1
        else:
            pre_score = score
            if init_score < score:
                init_score = score

    print("query seq:", tmp[0], tmp[1], tmp[2], tmp[3],tmp[4])
    sequence_index.append(tmp)

    curr_seq = tmp

    query += 1



