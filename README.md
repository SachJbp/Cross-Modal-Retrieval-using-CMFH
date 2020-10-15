# Cross-Modal-Retrieval-using-CMFH

This is my implementation of Cross-Modal retrieval using Collective Matrix Factorization Hashing (CMFH) originally described in [link](http://ise.thss.tsinghua.edu.cn/MIG/CVPR2014%20Collective%20Matrix%20Factorization%20Hashing%20for%20Multimodal%20Data.pdf). CMFH helps us to generate unified embeddings for different modes of data in such a way that similar semantic data is nearer ( For eg, a video and its corresponding text being nearer in the common embedding space). In this repo, we demonstrate training and testing for video-text and text-video retrievals on MSR-VTT-10K dataset.

To train CMFH from scratch follow these steps:

1) Since, generation of feature matrices is a time comsuming task, you may want to download the precomputed feature matrices [X1](https://drive.google.com/file/d/1n8YSl67smU1Kp_F2yQD1HJMrVFUs1qgi/view?usp=sharing) for training videos ,[X2](https://drive.google.com/file/d/1Aso3fYLRGnzYjvpwir7BrT0ifCaYBacg/view?usp=sharing) for corresponding annotated texts, [X1_test](https://drive.google.com/file/d/1Aa42WnOK1u0rkULDw5wGk2TV25rFvHWq/view?usp=sharing) for test videos and [X2_test](https://drive.google.com/file/d/1rospQzlPUSKnxQB26BApojWhLiKVP1A-/view?usp=sharing) for corresponding annotated texts and put it in feature_matrices folder. Feature matrices are of dimension (d * n) where where d is the length of embedding a single video or a text and n is the number of samples in training set.

For folks willing to generate feature matrices themselves, can download the train videos from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared?fbclid=IwAR3ZsoQiKf_SZjV15sGyoSr20C8A2FteXgoXS0B2Acgzq1wLpZzERP76ktc) and put them in respective folder. 

2) Run the following command

> python train.py

For folks , just interested in testing the joint embeddings for video-text and text-video retrievals can follow these steps:
1) Download pre-trained projection matrices P1 from [here](https://drive.google.com/file/d/1k-WjlaCeFdgx20cZlAaDp9tmQVE6Tiw0/view?usp=sharing) and P2 from [here](https://drive.google.com/file/d/1-2vcpE3zjdrDrXbFe9ihdN9uGVY6Tsuz/view?usp=sharing) and save them as P1.npy and P2.npy respectively in projection_matrices folder.
2) Run the cells in test notebook as instructed to run the webapp and play around by entering YouTube IDs of smaller videos(< 1 min ) and get the matching texts from MSR-VTT-10K training data texts or enter a sentence and get top 10 relevant video YouTube URLs from MSR-VTT-10K training videos. 
