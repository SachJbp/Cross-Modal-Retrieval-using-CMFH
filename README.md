# Cross-Modal-Retrieval-using-CMFH

This is an implenmentation of Cross-Modal retrival using Collection Matrix Factorization Hashing (CMFH). It helps us to generate unified embeddings for different modes of data. In this repo we demonstrate training and testing for video-text and text-video retrievals on MSR-VTT-10K dataset.

To train CMFH from scratch follow these steps:

1) Since, generation of feature matrices is a time comsuming task, you may want to download the precomputed feature matrices [X1](https://drive.google.com/file/d/1-wi8A6UuQxhpffeBnkjmML4ipOL5AxpS/view?usp=sharing) ,[X2](https://drive.google.com/file/d/1-xYAKsnN7gzVJdoSHr61aIyGmFi2mdqb/view?usp=sharing) and put it in feature_matrices folder.

For folks willing to generate feature matrices, can download the train and test videos from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared?fbclid=IwAR3ZsoQiKf_SZjV15sGyoSr20C8A2FteXgoXS0B2Acgzq1wLpZzERP76ktc) and put them in respective folders 

2) Run the following command

> python train.py

For folks , just interested in testing the joint embeddings for video-text and text-video retrievals can follow these steps:
1) Download pre-trained projection matrices P1 from [here] (https://drive.google.com/file/d/1k-WjlaCeFdgx20cZlAaDp9tmQVE6Tiw0/view?usp=sharing) and P2 from [here](https://drive.google.com/file/d/1-2vcpE3zjdrDrXbFe9ihdN9uGVY6Tsuz/view?usp=sharing) and save them as P1.npy and P2.npy respectively.
2) Run as cells in test notebook as instructed to run the webapp  
