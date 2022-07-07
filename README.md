# A Sequence-Based Visual Place Recognition Technique with Segmented Database and Compact Sequence List

This is the repo for the paper "A Sequence-Based Visual Place Recognition Technique with Segmented Database and Compact Sequence List". It contains implementing code and latex paper file in it. 

## Abstract

Sequence-based visual place recognition algorithms have been proved to be able to handle environmental changes caused by illumination, weather, and time of the day with hand-crafted descriptors. However, exhaustive search for all images in query sequence is computationally expensive. In this paper, we propose a technique that can significantly reduce the size of searching space for sequence match while remaining a state-of-the-art accuracy. Firstly, we managed to achieve a better selection for reference candidates of images in query sequence by segmenting database according to similarity. Then, a much more informative and compact query sequence is designed by removing all the unnecessary images in origin query sequence. State-of-the-art performance is reported on public dataset with challenging environmental changes. Our algorithm shows comparable accuracy with other current best results, and exceeds all the other methods in the dataset with illumination variation. In addition, the decrease of execution time and higher success rate for selecting candidates of query images for sequence match are also provided.

## Methodology

![fig_1](D:\GitHub\Sequence-Based-VPR\pictures\fig_1.png)

## Experiment Result

##### Some matching results of query image and retrieved image

![1](.\pictures\1.jpg)

![2](.\pictures\2.jpg)

![4](.\pictures\4.jpg)

![5](.\pictures\5.jpg)



![9](.\pictures\9.jpg)

![10](.\pictures\10.jpg)

![11](.\pictures\11.jpg)

![7](.\pictures\7.jpg)

![8](.\pictures\8.jpg)

![6](.\pictures\6.jpg)

##### Accuracy comparison with other algorithms

<img src=".\pictures\fig_2.png" alt="fig_2" style="zoom: 50%;" />

##### Execution time comparison with CoHoG

<img src=".\pictures\fig_3.png" alt="fig_3" style="zoom: 67%;" />

##### Accuracy comparison of various query sequence length withConSequential-SLAM

<img src=".\pictures\fig_4.png" alt="fig_4" style="zoom:67%;" />

## To run the code

 Firstly,use pip to install **python-opencv, numpy, matplotlib** and **scikit-image** .

Run frame_retrieval to see single frame matching results, the output folder has stored our running results.

Run frame_retrieval to see sequence matching results, the query and reference sequence index will be printed out.

## Future work

This paper has been rejected by the IROS 2022. The manipulating tricks  used to simply searching process have been considered as innovative according to the viewers. However, there are still some serious problems, such as the related work is too short, some methods are not illustrated clearly and the experiment is not convincing enough. I will fix these problems first with the help of the reviews' advice. After that I'm considering using some network based descriptors like NetVLAD to replace the HoG descriptors which I am currently using. This could help to improve the accuracy because of the application of neural network which could probably lead to a more accurate and efficient algorithm. 

 
