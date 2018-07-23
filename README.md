# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

I used the <a href="Image Classifier Project.ipynb">notebook</a> to test and try methods and functions.
Then built a Network class for the app (<a href="train.py">train.py</a>, <a href="predict.py">predict.py</a>) used both for training, validation and prediction.

It uses transferlearning and hook the classifier or last feature map of the architecture chosen.<br>
Ive tested this with resnet101 and vgg16 as base architechtures and reached >97% Accurracy on both in 8 to 10 epochs.

The attached classifier consists of one hidden ReLu activated layer with default 512 features and will LogSoftmax the output to number of features depending on dataset loaded.
