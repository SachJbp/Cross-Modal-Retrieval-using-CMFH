# Cross-Modal-Retrieval-using-CMFH

This is an implementation of Cross-Modal retrieval using Collective Matrix Factorization Hashing (CMFH) published in [link](http://ise.thss.tsinghua.edu.cn/MIG/CVPR2014%20Collective%20Matrix%20Factorization%20Hashing%20for%20Multimodal%20Data.pdf). It helps us to generate unified embeddings for different modes of data. In this repo we demonstrate training and testing for video-text and text-video retrievals on MSR-VTT-10K dataset.

To train CMFH from scratch follow these steps:

1) Since, generation of feature matrices is a time comsuming task, you may want to download the precomputed feature matrices [X1](https://drive.google.com/file/d/1-wi8A6UuQxhpffeBnkjmML4ipOL5AxpS/view?usp=sharing) ,[X2](https://drive.google.com/file/d/1-xYAKsnN7gzVJdoSHr61aIyGmFi2mdqb/view?usp=sharing) and put it in feature_matrices folder.

For folks willing to generate feature matrices themselves, can download the train videos from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared?fbclid=IwAR3ZsoQiKf_SZjV15sGyoSr20C8A2FteXgoXS0B2Acgzq1wLpZzERP76ktc) and put them in respective folder. 

2) Run the following command

> python train.py

For folks , just interested in testing the joint embeddings for video-text and text-video retrievals can follow these steps:
1) Download pre-trained projection matrices P1 from [here](https://drive.google.com/file/d/1k-WjlaCeFdgx20cZlAaDp9tmQVE6Tiw0/view?usp=sharing) and P2 from [here](https://drive.google.com/file/d/1-2vcpE3zjdrDrXbFe9ihdN9uGVY6Tsuz/view?usp=sharing) and save them as P1.npy and P2.npy respectively in projection_matrices folder.
2) Run the cells in test notebook as instructed to run the webapp  
