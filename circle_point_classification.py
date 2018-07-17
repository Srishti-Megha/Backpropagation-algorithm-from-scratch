import gen_data
import feedforword
import backpropagation
import random
import math
import matplotlib.pyplot as plt
import numpy as np

total_no_of_data_points = 2000
total_no_of_test_data_point = 650
no_of_hidden_neurons = 88
no_of_output_layer_neurons = 1
no_of_epochs = 1000

learning_rate = 0.1

iiii = 0

#Average Error in Epoch
error_1 = []

#Error calculation after the last update in the epoch
error_2 = []


xx = []

z = gen_data.data_generator(total_no_of_data_points/2, flag = 0)

random.shuffle(z)
#print(len(z))

no_of_input_neurons = len(z[0])-1
#print(no_of_input_neurons)



W_1 = [[random.uniform(-1,1) for i in range(no_of_hidden_neurons)]for j in range(no_of_input_neurons)]          # weights between output and hidden layer

W_2 = [[random.uniform(-1,1) for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]   # weights between hidden layer and output

bias_1 = [random.uniform(0,1) for i in range(no_of_hidden_neurons)]                                             # bias between input and hidden layer neurons

bias_2 = [random.uniform(0,1) for i in range(no_of_output_layer_neurons)]                                       # bias between hidden and output layer neurons

e = [0 for i in range(no_of_output_layer_neurons)]                                                              # error of neurons in output layer

#t = [[] for i in range(no_of_hidden_neurons)]                                                                  # output of neurons in hidden layer
#
#o = [[] for i in range(no_of_output_layer_neurons)]                                                            # calculated output

#print (z)
#print()
for epoch in range(no_of_epochs):
    E1 = 0
    random.shuffle(z)
    print("epoch = %d" %epoch)
    xx.append(epoch)
    for data_point in range(total_no_of_data_points):

#        print("W_1 : {}" .format(W_1))
#        print()
#        print("W_2 : {}" .format(W_2))
#        print()
#        print("bias_1 : {}" .format(bias_1))
#        print()
#        print("bias_2 : {}" .format(bias_2))
#
        #feedforword.feedforword(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii)
        a, a_2, t, o = feedforword.feedforward(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii)

#        print("a : {}" .format(a))
#        print()
#        print("a_2 : {}" .format(a_2))
#        print()
#        print("t : {}" .format(t))
#        print()
#        print("o : {}" .format(o))
#        print()
#        #print("a : {}" .format(a))
#        print()

#        if epoch == 0 or epoch == no_of_epochs - 1:
        if o[0] >= 0.5:
            pp = 1
        else:
            pp = 0


        # Plot Misclassified ponts in training after last epoch updation
        if epoch == no_of_epochs - 1:
            if o[0] >= 0.5:
                pp = 1
            else:
                pp = 0
    
            if pp == z[data_point][2]:
    
                if z[data_point][2] == 0 :
                    plt.scatter(z[data_point][0],z[data_point][1],color='lightblue')
                else:
                    plt.scatter(z[data_point][0],z[data_point][1],color='lightsalmon')
    
            else:
                if z[data_point][2] == 0:
                    plt.plot(z[data_point][0],z[data_point][1],'k+')
                else:
                    plt.plot(z[data_point][0],z[data_point][1],'g+')


        W_1, W_2, bias_1, bias_2, E = backpropagation.backpropogation(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, learning_rate, a, a_2, t, o)
        #sum of square of error
        E1 = E1 + E
        
    #mean sum of square of error
    E1 = E1 / total_no_of_data_points
    #Average Error in Epoch
    error_1.append(E1)
    
    # Plot Misclassified ponts in training after last epoch updation
    if epoch == no_of_epochs - 1:
        plt.title("Misclassified ponts in training after last epoch updation ")
        plt.show()
        
    #Error calculation after the last update in the epoch
    sq_e = 0
    E = 0
    for data_point in range(total_no_of_data_points):

        iiii = 0
        _, _, _, o = feedforword.feedforward(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii)
        for output_neuron in range(no_of_output_layer_neurons):
            
            # error = actual_output - calculated_output
            e[output_neuron] = (o[output_neuron] - z[data_point][2+output_neuron])
            
            # calculating the sum of square of error
            sq_e = sq_e + (e[output_neuron])**2
            
        sq_e = sq_e/2
        #sum of square of error
        E = E + sq_e
    # mean sum of square of error
    E = E / total_no_of_data_points
    #Error calculation after the last update in the epoch
    error_2.append(E)



# Plot of Error
error_1 = numpy.array(error_1)
pl.plot(error_1)
pl.xlabel("No. of epochs")
pl.ylabel("Error")
pl.title("Average Error in Epoch")
pl.show()

error_2 = numpy.array(error_2)    
pl.plot(error_2)
pl.xlabel("No. of epochs")
pl.ylabel("Error")
pl.title("Error calculation after the last update in the epoch")
pl.show()

count = 0


#testing
misclassified_points = []
p = 0   
      
#test point generation using gen_data.py  
test_data = gen_data.data_generator(total_no_of_test_data_point/2, flag = 1)
random.shuffle(test_data)
for data_point in range(total_no_of_test_data_point):
    iiii = 1
    #Calculate output of the test data
    a, a_2, t, o = feedforword.feedforward(test_data, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii)

    if o[0] >= 0.5:
        pp = 1
    else:
        pp = 0

#    if pp == test_data[data_point][2]:
#
#        if test_data[data_point][2] == 0 :
#            pl.scatter(test_data[data_point][0],test_data[data_point][1],'lightblue')
#        else:
#            pl.scatter(test_data[data_point][0],test_data[data_point][1],'lightred')

    if pp != test_data[data_point][2]:
        misclassified_points.append([])
        misclassified_points[p].append(test_data[data_point][0])
        misclassified_points[p].append(test_data[data_point][1])
        p = p+1
        if test_data[data_point][2] == 0:
            plt.plot(test_data[data_point][0],test_data[data_point][1],'k+')
        else:
            plt.plot(test_data[data_point][0],test_data[data_point][1],'m+')
        


#class_region
x_q = -5
data = []
while x_q <= 5:
    y_q = -5
    while y_q <= 5:
        x_q = round(x_q, 2)
        y_q = round(y_q, 2)
        c = [x_q, y_q]
        data.append(c)
        y_q = y_q + 0.09
        
    x_q = x_q + 0.09
    
for i in range(len(data)):
    if data[i][0]**2 + data[i][1]**2 >= 4:
        plt.scatter(data[i][0], data[i][1], color = 'lightsalmon')
        
    else:
        plt.scatter(data[i][0], data[i][1], color = 'lightblue')
#Plot of Misclassified points
plt.title("Plot of Misclassified points during Test")
plt.show()
































##region of classification
#region = gen_data.data_generator(2500/2, flag = 0)
#
#for data_points in range(2500):
#    if region[data_points][0] **2 + region[data_points][1] **2 > 4:
#        plt.plot(region[data_points][0], region[data_points][1],'ro')
#        
#    else:  
#        plt.plot(region[data_points][0], region[data_points][1],'bo')
#        
#plt.show()

#cc = 0
#x_1 = 0
#y_1 = 0
#for i in range(3):
#    x = -i / 5
#    while(cc < 40):
#        for y in range(3):
#        		if (x)**2 + (y)**2 > 4:
#        		    plt.scatter(z[i][0],z[i][1],color='lightsalmon')
#        		    
#        		else:
#        		    plt.scatter(z[i][0],z[i][1],color='cyan')


###########%%%%%%%%%%%%#####################
"""
#testing

test_data = gen_data.data_generator(total_no_of_test_data_point/2, flag = 1)
random.shuffle(test_data)
for data_point in range(total_no_of_test_data_point):
    iiii = 1
    a, a_2, t, o = feedforword.feedforward(test_data, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii)

    if o[0] >= 0.5:
        pp = 1
    else:
        pp = 0

    if pp == test_data[data_point][2]:
        count = count + 1
#        if test_data[data_point][2] == 0 :
#            plt.plot(test_data[data_point][0],test_data[data_point][1],'bo')
#        else:
#            plt.plot(test_data[data_point][0],test_data[data_point][1],'ro')

    else:
        
#        print("output signal: {}" .format(o[0]))
#       
#        print("test_data {}   {}" .format(test_data[data_point][0], test_data[data_point][1]))
#        print()
#        print()
        if test_data[data_point][2] == 0:
            plt.plot(test_data[data_point][0],test_data[data_point][1],'k+')
        else:
            plt.plot(test_data[data_point][0],test_data[data_point][1],'g+')

accuracy = count/ total_no_of_test_data_point * 100
plt.show()

error = np.array(error)
#print(error)
xx = np.array(xx)
#plt.plot(xx, error)
#plt.show()

#print("accuracy: {}" .format(accuracy))




"""




##############%%%%%%%%%%%%%%%%%###########################

