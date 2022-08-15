# cat-facia-lpain
1.For deployment instructions, development environment and system requirements.
tensorflow's version is 2.9.1
keras' version is 2.9.0
The spyder is used and the implementation of the system is always executed on a mac system.

2.Introduction to the methodology and documentation of the relevant items.
The project is divided into two parts, the first part has three code files, namely 'customfinal.py', 'resnet50.py' and 'Vgg16.py' all three. These models are trained using overall evaluation.
The second part is trained on each of the seven features of the cat's face, followed by a decision tree model and a Bayesian model for final pain level recognition.
The file mainly consists of a convolutional neural network model with seven different features and a dataset with seven different features with the processing file. A 'main.py' file is included, which introduces the decision tree model and the Bayesian model.
The ultimate aim of the project is to classify the level of facial pain in cats by two means: overall recognition classification and feature recognition classification.

3.Technical methods used
The architecture used for the system is TensorFlow, Keras.Regarding the framework TensorFlow, it is clear from its name that it consists of two parts: tensor, which stands for tensor, and flow, which stands for the computation of graphs about data flows.TensorFlow is a machine learning system with large-scale learning capabilities.(Martín Abadi. 2016)TensorFlow is supported in a variety of computer languages such as python, java, c++ etc.; and TensorFlow is easy to deploy and can be used on Android as well as ios for relevant recognition and use.It is also partly because Google's use of it has allowed it to become better popular and used.But there is no denying that PyTorch is also a very good deep learning framework.
The project is related to machine learning. One of the applications of deep learning is convolutional neural networks. There are three neural network models used in the first approach, namely the custom CNN, ResNet50 and Vgg16.
The second method uses mainly convolutional neural networks, decision trees and NaviesBayesian.
