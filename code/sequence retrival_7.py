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
SN= 20 # subsets number
PR_1=0.00# precision recall threshold
PR_2=0.00# precision recall threshold

total_Query_Images=190
query_index_offset=0
total_Ref_Images=190
ref_index_offset=0

ref = 0 # start index for reference
query = 30 # start index for query


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
fore_limit = []
############################################################

dataset_name='GardensSmall'   #Please modify according to your needs
save_visual_matches_dir='./output'+dataset_name+'/'
# os.makedirs(save_visual_matches_dir)     # If the directory doesn't exist already.


query_directory = './dataset/dataset_4/query/'  # Please modify according to your needs
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

# build subsets of origin dataset, offline
while ref < total_Ref_Images:

    vector_1,regional_goodness_1,no_of_good_regions_1=read_ref_images(ref)

    if ref == 0:    # new subset start
        num_regions.append(no_of_good_regions_1)
    sub_length = 0
    flag = 0
    pre_score=1
    init_score=0
    while flag == 0:
        vector_2,regional_goodness_2,no_of_good_regions_2=read_ref_images(ref+sub_length+1)
        num_regions.append(no_of_good_regions_2)
        score=conv_match_dotproduct(vector_1.transpose(),vector_2 ,regional_goodness_1,total_no_of_regions)
        print(score, ref, ref + sub_length+1)
        sub_length += 1
        if (score > pre_score) or (score < init_score - ST):
            flag = 1
            tmp = []
            for j in range(sub_length):
                tmp.append(ref+j)
            subsets_index.append(tmp)
            ref = ref + sub_length
            num += 1
        else:
           pre_score = score
           if init_score<score:
             init_score = score

# image with most regions is considered to contain most information
for rep in range(len(subsets_index)):
    reg_max = 0
    for j in range(len(subsets_index[rep])):
        if num_regions[subsets_index[rep][j]] > reg_max:
            reg_max = num_regions[subsets_index[rep][j]]
            index = subsets_index[rep][j]
    rep_index.append(index)

build_map()
# visualize procedure of building the map
print(subsets_index)
# print(num_regions)
print(len(subsets_index))
print(len(num_regions))
print(rep_index)

# n_ref_desc=np.array(ref_desc)
# n_ref_desc.tofile("ref_desc.dat", sep=",", format='%d')
# print(n_ref_desc.shape)
# np.savetxt("num_regions.csv", num_regions, delimiter=",")
# np.savetxt("ref_desc.csv", ref_desc, delimiter=",")
# np.savetxt("ref_desc_s.csv", ref_desc_s, delimiter=",")
# np.savetxt("subsets_index.csv", subsets_index, delimiter=",")
# np.savetxt("rep_index.csv", rep_index, delimiter=",")

##################################################################################
# query retrieval, build dynamic query sequence list
# while PR_1 < 0.01:
#     retrieved_num = 0
#     query = 20
#     TP1=0
#     TP2=0




# for l in range(21):
#     fore_limit.append(1)
# ref = 20
# while ref < total_Ref_Images:
#     tmp =[]
#     vector_3, regional_goodness_3, no_of_good_regions_3 = read_query_images(query)
#     tmp.append(ref)
#
#     # first previous image
#     sub_length = 0
#     query_flag = 0
#     pre_score = 1
#     init_score = 0
#     fore_ref = ref - 1
#     while query_flag == 0 and fore_ref > 0:
#         vector_4, regional_goodness_4, no_of_good_regions_4 = read_query_images(fore_ref)
#         score = conv_match_dotproduct(vector_3.transpose(), vector_4, regional_goodness_3,
#                                           total_no_of_regions)
#         # print("score1",score)
#         fore_ref -= 1
#         if (score > pre_score) or (score < init_score - ST):
#             tmp.append(fore_ref + 1)
#             query_flag = 1
#         else:
#             pre_score = score
#             if init_score < score:
#                 init_score = score
#
#     # further previous image
#     sub_length = 0
#     query_flag = 0
#     pre_score = 1
#     init_score = 0
#     fore_ref = fore_ref
#     while query_flag == 0 and fore_ref > 0:
#         vector_5, regional_goodness_5, no_of_good_regions_5 = read_query_images(fore_ref)
#         score = conv_match_dotproduct(vector_4.transpose(), vector_5, regional_goodness_4,
#                                       total_no_of_regions)
#         # print("score2", score)
#         fore_ref -= 1
#         if (score > pre_score) or (score < init_score - ST):
#             tmp.append(fore_ref + 1)
#             query_flag = 1
#         else:
#             pre_score = score
#             if init_score < score:
#                 init_score = score
#
#     fore_limit.append(tmp[2])
#     print(tmp[2])
#     ref+=1
#     print(ref)
#     print(len(fore_limit))
#     print(fore_limit)
#
# np.savetxt("fore_limit.csv", fore_limit, delimiter=",")

fore_limit = np.loadtxt('fore_limit.csv', dtype=np.float, delimiter=',')



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

    sequence_index.append(tmp)

    sub_length = 0
    query_flag = 0
    pre_score = 1
    init_score = 0
    fore_query = fore_query
    while query_flag == 0 and fore_query > 0:
        vector_8, regional_goodness_8, no_of_good_regions_8 = read_query_images(fore_query)
        score = conv_match_dotproduct(vector_7.transpose(), vector_8, regional_goodness_7,
                                      total_no_of_regions)
        # print("score2", score)
        fore_query -= 1
        if (score > pre_score) or (score < init_score - ST):
            query_sequence_list.append(vector_8)
            tmp.append(fore_query + 1)
            query_flag = 1
        else:
            pre_score = score
            if init_score < score:
                init_score = score

    print("query seq:", tmp[0], tmp[1], tmp[2], tmp[3],tmp[4],tmp[5])
    sequence_index.append(tmp)


    curr_seq = tmp

    query += 1



    # build the best match matrix for seq list
    seq_best = []
    seq_best_match = []
    seq_top_k = []
    confusion_vector = np.zeros(total_Ref_Images, dtype=np.float32)
    for seq in range(len(query_sequence_list)):
        vector_0 = query_sequence_list[seq]
        # finding most similar matches in the representative subset of the database
        for ref in range(len(rep_index)):
            score=conv_match_dotproduct(vector_0.transpose(),ref_desc[ref],regional_goodness_3,total_no_of_regions)
            # store all the scores in the subset
            confusion_vector[ref] = score

        top_k=SN
        best = []
        best_match = []
        top_k_idx=confusion_vector.argsort()[::-1][0:top_k]
        for num1 in range(top_k):
            confusion_vector_2 = []
            verify_num = 0
            for num2 in range(len(ref_desc_s[top_k_idx[num1]])):
                score = conv_match_dotproduct(vector_0.transpose(),ref_desc_s[top_k_idx[num1]][num2],regional_goodness_3,total_no_of_regions)
                confusion_vector_2.append(score)
                # print("confusion_vector_2: ", confusion_vector_2, "   ", len(ref_desc_s[top_k_idx[num1]]))
            best_match_index = np.argmax(confusion_vector_2)
            # get best match similarity
            best.append(np.max(confusion_vector_2))
            # print("np.max(confusion_vector_2)",np.max(confusion_vector_2))
            # print("best match", subsets_index[top_k_idx[num1]][best_match_index])
            best_match.append(subsets_index[top_k_idx[num1]][best_match_index])

            # visualize the matches
            # save_visual_matches(query, query, best_match[selected_seq])

        seq_best.append(best)
        seq_best_match.append(best_match)
        seq_top_k.append(top_k_idx)
        print("candidates:",best_match)

    sequence_score = []
    sequence_results = []
    seq_tmp = []
    loop_id = 0
    final_results = []




    for seq1 in range(len(seq_best_match[0])):
        for seq2 in range(len(seq_best_match[1])):
            if seq_best_match[0][seq1] > seq_best_match[1][seq2] and seq_best_match[1][seq2] > fore_limit[seq_best_match[0][seq1]]:
                for seq3 in range(len(seq_best_match[2])):
                    if seq_best_match[1][seq2] > seq_best_match[2][seq3] and seq_best_match[2][seq3] > fore_limit[seq_best_match[1][seq2]]:
                        for seq4 in range(len(seq_best_match[3])):
                            if seq_best_match[2][seq3] > seq_best_match[3][seq4] and seq_best_match[3][seq4] > fore_limit[seq_best_match[2][seq3]]:
                                for seq5 in range(len(seq_best_match[4])):
                                    if seq_best_match[3][seq4] > seq_best_match[4][seq5] and seq_best_match[4][seq5] > fore_limit[seq_best_match[3][seq4]]:
                                        for seq6 in range(len(seq_best_match[5])):
                                            if seq_best_match[4][seq5] > seq_best_match[5][seq6] and seq_best_match[5][seq6] > fore_limit[seq_best_match[4][seq5]]:
                                                # print("seg-set index", top_k_idx[seq1],top_k_idx[seq2],top_k_idx[seq3])
                                                # loop_id += 1
                                                # # print("id",loop_id)
                                                seq_tmp = []
                                                sequence_score.append(seq_best[0][seq1]+seq_best[1][seq2]+seq_best[2][seq3]+seq_best[3][seq4]+seq_best[4][seq5]+seq_best[5][seq6])
                                                seq_tmp.append(seq_best_match[0][seq1])
                                                seq_tmp.append(seq_best_match[1][seq2])
                                                seq_tmp.append(seq_best_match[2][seq3])
                                                seq_tmp.append(seq_best_match[3][seq4])
                                                seq_tmp.append(seq_best_match[4][seq5])
                                                seq_tmp.append(seq_best_match[5][seq6])
                                                sequence_results.append(seq_tmp)
                                                # print("seq index:",seq_best_match[0][seq1],seq_best_match[1][seq2],seq_best_match[2][seq3])
                                                # print("seq score:",seq_best[0][seq1]+seq_best[1][seq2]+seq_best[2][seq3])
    if len(sequence_score):
        selected_seq = np.argmax(sequence_score)
        rest_seq = np.argsort(sequence_score)[::-1]
        if PR_1 > 0:
            if len(sequence_score)>1:
                if sequence_score[selected_seq] - sequence_score[rest_seq[1]] > PR_1:
                    final_results.append(sequence_results[selected_seq])
                    retrieved_num += 1
        else:
            final_results.append(sequence_results[selected_seq])
            retrieved_num += 1
            for k in range(len(rest_seq)-1):
                if sequence_score[selected_seq] - sequence_score[rest_seq[k+1]] < PR_2:
                    final_results.append(sequence_results[k+1])
                    retrieved_num += 1
                else:
                    break
        # print("sequence_results:", sequence_results[selected_seq][0], sequence_results[selected_seq][1],
        #       sequence_results[selected_seq][2])

        for k in range(len(final_results)):
            if abs((final_results[k][0] + final_results[k][3])/2 - (curr_seq[0] + curr_seq[3])/2) < 3:
                TP1 += 1
        pre_tp2 = TP2
        for k in range(len(final_results)):
            if abs((final_results[k][0] + final_results[k][3])/2 - (curr_seq[0] + curr_seq[3])/2) < 3:
                TP2 += 1
                break
        if pre_tp2 == TP2:
            print("not found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("match sequence not found")
        no_matching += 1
    # print("selected_seq", selected_seq)
    # print("sequence_score",sequence_score[selected_seq])
    for k in range(len(sequence_score)):
        print("sequences", sequence_results[np.argsort(sequence_score)[::-1][k]])
    if(len(sequence_results)):
        print("sequence_results",sequence_results[selected_seq])
precision = TP1/retrieved_num
recall = TP2/ (query-50)

print("precision",precision)
print("recall",recall)
print("no_matching",no_matching)

    # PR_1 +=0.0005

