
from numpy.linalg import norm
from tqdm import tqdm
import joblib

# source_folder = r"C:\Users\Dell\.cache\kagglehub\datasets\paramaggarwal\fashion-product-images-dataset\versions\1\fashion-dataset\images"
# destination_folder = "images"


# for file in os.listdir(source_folder):
#     if file.lower().endswith((".jpg", ".jpeg", ".png")):
#         source_path = os.path.join(source_folder, file)
#         destination_path = os.path.join(destination_folder, file)

#         shutil.copy(source_path, destination_path)

# print("All images copied successfully!")

import numpy as np
import os
import cv2
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])



def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expand_array=np.expand_dims(img_array,axis=0)
    prerocess_img=preprocess_input(expand_array)
    result=model.predict(prerocess_img,verbose=0).flatten()
    normalized_img=result/norm(result)
    return normalized_img


filenames=[]

for file in os.listdir("images"):
    filenames.append(os.path.join("images",file))




feature_list=[]
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))


joblib.dump(feature_list,'featuers.pkl',compress=3)
joblib.dump(filenames,'images.pkl',compress=3)













