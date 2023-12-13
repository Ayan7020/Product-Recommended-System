## End to End Deep learning and Machine learning Projects
### Requirement modules 
* Scikitlearn
* Tensorflow
* Keras
* numpy
* pickle
* Flask
* tqdm

### Dataset 
* https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
## Project Discription
### For app.py
* Loads a pre-trained ResNet50 model with ImageNet weights.
* Creates a new model with the ResNet50 base and a GlobalMaxPool2D layer.
* Defines a function to extract normalized features from an image using the ResNet50 model.
* Iterates through image files in the "fashion-dataset/images" directory, extracts features, and stores them in a feature list.
* Saves the feature list and file names using the Pickle module.

### For main.py
* In main.py is uses for backend as for website
* BE carefull About file path

