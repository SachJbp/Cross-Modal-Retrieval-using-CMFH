from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
from GenerateFeatureMatrices import generate_feature_matrices
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
import glob
import drawnow
logging.set_verbosity(logging.ERROR)

#get feature matrices for training
file_X1 = glob.glob('feature_matrices/X1.npy')
file_X2 = glob.glob('feature_matrices/X2.npy')
file_X1_test = glob.glob('feature_matrices/X1_test.npy')
file_X2_test = glob.glob('feature_matrices/X2_test.npy')

if not (file_X1 and file_X2 and file_X1_test and file_X2_test) and 1 < 0:
    with open('train_val_videodatainfo.json') as f:
            train_data = json.load(f)

    with open('test_videodatainfo.json') as f:
          test_data = json.load(f)
    print("Generating feature matrix for training...")
    X1, X2 = generate_feature_matrices(train_data)
    print("Generating feature matrix for testing...")
    X1_test_,X2_test = generate_feature_matrices(test_data)

    #save the feature matrices for later use
    np.save('X1.npy',X1)
    np.save('X2.npy',X2)
    np.save('X1_test.npy',X1_test)
    np.save('X2_test.npy',X2_test)

else:
    X1 = np.load('feature_matrices/X1.npy')
    X2 = np.load('feature_matrices/X2.npy')
    X1_test = np.load('feature_matrices/X1_test.npy')
    X2_test = np.load('feature_matrices/X2_test.npy')

#Initialize Ut, Pt by random matrices, and centering X(t) by means, t = 1, 2.
k = 48   #common embedding size 48 
d_1 = 400  
d_2 = 512
P1  = np.random.randn(k,d_1)
P2 = np.random.randn(k,d_2)
U1 = np.random.randn(d_1,k)
U2 = np.random.randn(d_2,k)

#center the data
mu1 = np.mean(X1[:,:],axis = 1)
mu2 = np.mean(X2[:,:],axis = 1)
std1 = np.std(X1[:,:],axis = 1)
std2 = np.std(X2[:,:],axis = 1)
mu1 = mu1.reshape(400,1)
mu2 = mu2.reshape(512,1)

X_1c = (X1 - mu1) 
X_2c = (X2 - mu2) 

X_1_testc = (X1_test - mu1)
X_2_testc = (X2_test - mu2)

#Set hyperparameters for CMFH training
iterations = 100
λ = 0.5
µ = 100
γ = 0.01


loss= [ ]
import matplotlib.pyplot as plt
from drawnow import drawnow

def makeFig():
    plt.scatter(l1,loss)

l1 = []

#Start training
for i in range(iterations):
  print(i)
  a1 = λ* U1.T.dot(U1) + (2*µ + γ)*np.identity(k) + (1-λ)* U2.T.dot(U2) + (2*µ + γ)*np.identity(k)
  a = np.linalg.inv(a1)
  b1 = (λ*U1.T + µ*P1).dot(X_1c) 
  b2 = ((1-λ)*U2.T + µ*P2).dot(X_2c)
  b = b1 + b2
  V = a.dot(b)  #eq1
  #print(V.shape)
  l1.append(i)
  P1 = V.dot(X_1c.T).dot(np.linalg.inv(X_1c.dot(X_1c.T) + (γ/µ)*np.identity(d_1)))
  P2 = V.dot(X_2c.T).dot(np.linalg.inv(X_2c.dot(X_2c.T) + (γ/µ)*np.identity(d_2)))
  #print(P1.shape)

  U1 = X_1c.dot(V.T).dot(np.linalg.inv(V.dot(V.T) +(γ/λ)*np.identity(k)))
  U2 = X_2c.dot(V.T).dot(np.linalg.inv(V.dot(V.T) +(γ/(1-λ))*np.identity(k)))
  loss1 = λ * np.linalg.norm(X1 - U1.dot(V)) + (1-λ)* np.linalg.norm(X2 - U2.dot(V))
  loss2 = µ * (np.linalg.norm(V - P1.dot(X1)) + np.linalg.norm(V - P2.dot(X2)) )
  loss.append(loss1+loss2)

np.save('projection_matrices/P1.npy',P1)
np.save('projection_matrices/P2.npy',P2)
drawnow(makeFig)



    
    
