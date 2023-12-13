import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm

import tensorflow
import keras

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalMaxPool2D
from keras.applications.resnet50 import ResNet50, preprocess_input

import pickle

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPool2D()])


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expand_img)
    result = model.predict(preprocess_img).flatten()
    norm_result = result / norm(result)

    return norm_result


file_path_name = []
file_name = []
for file in os.listdir("fashion-dataset/images"):
    file_path_name.append(os.path.join("fashion-dataset", "images", file))
    file_name.append(file)

feature_list = []

for file in tqdm(file_path_name):
    feature_list.append(extract_features(file, model))

print(np.array(feature_list).shape)

pickle.dump(feature_list, open("feature_list.pkl ", "wb"))
pickle.dump(file_name, open("file_path_name.pkl ", "wb"))
