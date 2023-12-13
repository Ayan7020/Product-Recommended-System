import pickle
import numpy as np
from numpy.linalg import norm
import os
import cv2
from flask import Flask,redirect,render_template,request,send_from_directory
import tensorflow
import keras

from sklearn.neighbors import NearestNeighbors

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalMaxPool2D 
from keras.applications.resnet50 import ResNet50,preprocess_input


feature = np.array(pickle.load(open('feature_list.pkl','rb')))
file_name = np.array(pickle.load(open('file_name.pkl','rb')))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,244,3))
model.trainable = False
model = Sequential([
    model,
    GlobalMaxPool2D()
]) 
neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature)


app = Flask(__name__)  
 
@app.route('/')
def main(): 
    return render_template('index.html')

@app.route('/upload',methods=['POST','GET'])
def upload(): 
    global model, neighbors, feature, file_name
    image_var = request.files['image'] 
    image_name = image_var.filename
    temp_filename = os.path.join("temp_image.jpg")
    image_var.save(temp_filename)
    img = image.load_img(temp_filename,target_size=(224,244))  
    img_array = image.img_to_array(img)  
    expand_img = np.expand_dims(img_array,axis=0)  
    preprocess_img = preprocess_input(expand_img)  
    result = model.predict(preprocess_img).flatten() 
    norm_result = result/norm(result) 
    
    distances,indices = neighbors.kneighbors([norm_result]) 
    names = []
    names.append(image_name) 
    os.remove(temp_filename)  
    for file in indices[0][1:6]:
        names.append(file_name[file])   
    print(names)    
    return render_template('main.html',image_path = names) 

@app.route('/images/<filename>')
def path(filename):
    return send_from_directory('E:\Product Recommendation System\Fashion-dataset\images',filename)

 
@app.route('/test/<filename>')
def test_path(filename):
    return send_from_directory('E:\Product Recommendation System\Test',filename) 



if __name__ == '__main__':
    app.run(debug=True)   