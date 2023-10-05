# Competition - Digit Recognizier

## Competiton Description 
<p>MNIST is a dataset of handwritten images that has served as the basis for benchmarking classification algorithms. In this competition, the goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 

The Kaggle Digit Recognizer is a machine learning problem that involves **classifying handwritten digits**. The dataset contains 70,000 images of handwritten digits, each of which is 28 pixels in height and 28 pixels in width, __which is 128.13mb in size__. The goal is to correctly **identify the digit in each image (Image Identification or Classification problem)**.</p>

## Important Concepts 
- **Neural Networks models**
For digit recognition, neural networks are a popular machine learning technique. The neural network uses a large number of handwritten digits, known as training examples, to automatically infer rules for recognizing handwritten digits. TensorFlow, an open-source Python library developed by the Google Brain labs for deep learning research, can be used to build and train a neural network to recognize and predict the correct label for the digit displayed.

- **Computer Vision Fundamentals**
Convolutional Neural Networks (CNNs) are a popular machine learning technique used for image classification tasks. They are particularly well-suited for this task because they can learn and extract intricate features from raw image data. 

CNNs consist of several layers, including an input layer, convolutional layers, pooling layers, fully connected layers, and an output layer. The input layer takes in the raw image data as input, which is typically represented as matrices of pixel values. The convolutional layers are responsible for feature extraction, while the pooling layers reduce the spatial dimensions of the feature maps produced by the convolutional layers. The fully connected layers are used to classify the images based on the features extracted by the convolutional and pooling layers. 

There are several CNN architectures that can be used for image classification tasks, including VGG-16 and ResNet . These architectures have achieved state-of-the-art performance on several benchmark datasets.

- **Dimension Reduction**

Yes, dimensionality reduction is a preprocessing method that can be used to reduce the number of features in a dataset while retaining as much information as possible. It is an essential technique for preparing data for text classification, and it can be used as a preprocessing step for just about any machine learning application.

Dimensionality reduction techniques include 
1. Principal Component Analysis ( PCA ) - Unsupervised, linear method
2. Linear Discriminant Analysis (LDA) - Supervised, linear method
3. t-distributed Stochastic Neighbour Embedding (t-SNE) - Nonlinear, probabilistic method
4. singular value decomposition (SVD) - sparse data method

Each technique projects the data onto a lower-dimensional space while preserving important information. Feature selection and feature extraction are two main approaches to dimensionality reduction. Feature selection involves selecting a subset of the original features that are most relevant to the problem at hand, while feature extraction involves creating new features by combining or transforming the original features.

- **GPUs**


## What I Intend to Learn
- working image data,
- using Kaggle APIs
- EDA for image data,
- Visualizing image data,
- Feature engineering and extraction for image data,
- building pipelines for image classification,
- building models to learn the classification problem,
- Evaluating and
- Hyperparameter turning of model to improme performance.

#### Source: 
1. [CNN Image Classification | Image Classification Using CNN.](https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/.)
2. [Bing](https://bing.com/search?q=CNN+techniques+in+image+classification.)
3. [Image Classification in CNN: Everything You Need to Know.](https://www.upgrad.com/blog/image-classification-in-cnn/.)
4. [Image Classification Using CNN (Convolutional Neural Networks).](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/.)
5. [Neural networks and deep learning.](http://neuralnetworksanddeeplearning.com/chap1.html.)
6. [Dimensionality Reduction - Introduction to Machine Learning - Wolfram.](https://www.wolfram.com/language/introduction-machine-learning/dimensionality-reduction/.)
7. [Introduction to Dimensionality Reduction - GeeksforGeeks.](https://www.geeksforgeeks.org/dimensionality-reduction/.)
8. [Dimensionality Reduction and Preprocessing for Machine Learning in Python.](https://www.datainsightonline.com/post/dimensionality-reduction-and-preprocessing-for-machine-learning-in-python.)
9. [Dimensionality Reduction Using Convolutional Autoencoders.](https://link.springer.com/chapter/10.1007/978-981-19-0619-0_45.)

#### Further Reading
1. https://link.springer.com/chapter/10.1007/978-981-16-3153-5_48.
2. https://ieeexplore.ieee.org/document/8877011/.
3. https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_612.
4. https://bing.com/search?q=dimension+reduction+preprocessing.
5. https://www.kaggle.com/code/arthurtok/interactive-intro-to-dimensionality-reduction
