## End to End Porduct Recommended System 
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
# Image Feature Extraction with ResNet50

🚀 **Exciting News: Unleashing the Power of Machine Learning!** 🌟

Thrilled to share a recent project where I harnessed the capabilities of deep learning to extract image features using a ResNet50 model. 🖼️✨

## What's Inside:

- Leveraged TensorFlow and Keras to build a powerful image feature extraction pipeline.
- Utilized the ResNet50 architecture to capture intricate details from images.
- Ensured efficiency and scalability by incorporating a GlobalMaxPool2D layer.

## Key Insights:

- The extracted features serve as a robust foundation for various machine learning applications.
- Achieved seamless integration with the fashion dataset, enabling scalable feature extraction.

## Results & Impact:

- Successfully processed a diverse set of images from the fashion dataset.
- The script generates feature vectors, paving the way for enhanced pattern recognition and classification.

## Technologies Involved:

- Python, TensorFlow, Keras, and the powerful ResNet50 architecture.
- Implemented Pickle for efficient storage and retrieval of feature vectors.

## What's Next:

- Excited to explore applications in recommendation systems, image similarity, and beyond.
- Continuous learning and improvement to push the boundaries of what's possible.

## Acknowledgments:

- Gratitude to the open-source community and the creators of TensorFlow and Keras for empowering innovation.

## Curious to Learn More? Dive into the Code:

[GitHub Repository](https://github.com/your-username/your-repo)

Excited to share this journey of exploration and discovery! 🚀✨ #MachineLearning #DeepLearning #ImageProcessing #DataScience

### For app.py
* Loads a pre-trained ResNet50 model with ImageNet weights.
* Creates a new model with the ResNet50 base and a GlobalMaxPool2D layer.
* Defines a function to extract normalized features from an image using the ResNet50 model.
* Iterates through image files in the "fashion-dataset/images" directory, extracts features, and stores them in a feature list.
* Saves the feature list and file names using the Pickle module.

### For main.py
* In main.py is uses for backend as for website
* BE carefull About file path

