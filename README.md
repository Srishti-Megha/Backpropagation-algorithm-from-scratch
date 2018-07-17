# Backpropagation-algorithm-from-scratch
Implementation of backpropagation algorithm in python from scratch to classify points inside and outside the circle. <br/>
<br/>
Note: <br/>
The network build here is of one hidden layer.<br/>
The activation function used is sigmoid activation function.

# Requirements:
Python 3.6. <br/>
NumPy. <br/>
OpenCV 2. <br/>
matplotlib <br/> 

# Files:
gen_data.py: Generate points inside and outside the points of the circle of radius 4 in the range (-5, 5). <br/>
feedforward.py: Calculates preactivation of neurons in hidden layer, output layer and output of neurons in hidden layer, output layer. <br/>
backpropagation.py: Calculates updated weights and biases after the execution of backpropagation algorithm. <br/>
circle.py: Main file which calls functions from gen_data.py, feedforward.py, backpropagation.py and calculates and plots the epoch error, misclassified points of the training data and misclassified points of the test data. <br/>
circle2.py: Same as that of circle.py but here all the functions from gen_data.py, feedforward.py, backpropagation.py are written in the same file. <br/>
Understanding Multilayer Perceptron.pdf: File contains the results of the model in Experiment section. <br/>
<br/>
Note: <br/>
Training and test data are generated using gen_data.py file.
