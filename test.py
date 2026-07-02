import numpy as np
import os
import cv2
from numpy.linalg import norm
import tensorflow
import joblib

from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors 

feature_list=np.array(joblib.load("featuers.pkl"))
images=joblib.load("images.pkl")

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img=image.load_img('.test_img/gold_star.jpeg',target_size=(224,224))
img_array=image.img_to_array(img)
expand_array=np.expand_dims(img_array,axis=0)
preprocess_img=preprocess_input(expand_array)
result=model.predict(preprocess_img).flatten()
normalized=result/norm(result)


neighours=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighours.fit(feature_list)

distances,incidies=neighours.kneighbors(normalized.reshape(1,-1))


for file in incidies[0]:
   frame=cv2.imread(images[file])
   cv2.imshow('output',cv2.resize(frame,(300,300)))
   cv2.waitKey(0)



