import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
import glob

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

#load i3d model
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

def predict(sample_video):
        '''Embeds the video using i3d model trained on Kinetics-400 dataset
        '''
        global i3d
        # Add a batch axis to the sample video.
        model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

        logits = i3d(model_input)['default'][0]
        probabilities = tf.nn.softmax(logits)
        return np.array([probabilities]).T

#Universal Sentence Encoder module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
use_model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed1(input):
      #print(type(input))
      return use_model(input)

def semantic_sim(a,b):
    u=embed1(a).numpy()
    v=embed1(b).numpy()
    dist = spatial.distance.cosine(u, v)
    return 1-dist

def generate_feature_matrices(dataJSON):
        ''' Generates Feature matrices of different modes of data
        '''
        vid_dir = "TrainVideo/"
        X1 = np.array([])
        X2 = np.array([])
        last_vid_id = ''

        for data_dict in dataJSON['sentences']:
          vid_id = data_dict['video_id']
          caption = data_dict['caption']
          
          txt_feature = embed1([caption]).numpy().T
          
          if not (last_vid_id == vid_id):
            
            #using i3d
            video_path = 'TrainVideo/' + vid_id+'.mp4'
            sample_video = load_video(video_path)
            vid_feature = predict(sample_video)

          if X1 == 0:
            X1 = vid_feature
          else:
            X1 = np.concatenate([ X1 , vid_feature],axis = 1)

          if X2.size == 0:
            X2 = txt_feature
          else:
            X2 = np.concatenate([ X2 , txt_feature],axis = 1)
 
          last_vid_id = vid_id
        
        return X1, X2
