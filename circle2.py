import random
from random import shuffle
import math
import numpy
import matplotlib.pyplot as pl
import feedforword
import gen_data
#import matplotlib.backends.backend_pdf

#pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")


total_no_of_data_points = 2000
total_no_of_test_data_point = 650

no = total_no_of_data_points / 2
z =[[]]

no_of_input_neurons = 2


no_of_hidden_neurons = 88
no_of_output_layer_neurons = 1
no_of_epochs = 1000

learning_rate_2 = 0.1
learning_rate_1 = 0.1

iiii = 0
   
#Error corresponding to last training point in the epoch
error = []

#Average Error in Epoch
error_1 = []

#Error calculation after the last update in the epoch
error_2 = []

i = 0

while (i < no):
    x = random.uniform(-5,5)
    y = random.uniform(-5,5)
    if (x)**2 + (y)**2 > 4:
        z[i].append(x)
        z[i].append(y)
        z[i].append(1)
        pl.scatter(z[i][0],z[i][1],color='lightsalmon')
        i = i + 1
        #print (z)
        if i < no:
            z.append([])

while (i < 2*no):
    x = random.uniform(-5,5)
    y = random.uniform(-5,5)
    if (x)**2 + (y)**2 <= 4:
        z.append([])
        z[i].append(x)
        z[i].append(y)
        z[i].append(0)
        pl.scatter(z[i][0],z[i][1],color='lightblue')
        i = i + 1


pl.title("Train Data Points")
pl.show()

#shuffle the data points
random.shuffle(z)




W_1 = [[random.uniform(-1,1) for i in range(no_of_hidden_neurons)]for j in range(no_of_input_neurons)]  # weights between output and hidden layer

W_2 = [[random.uniform(-1,1) for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]  # weights between hidden layer and output

bias_1 = [random.uniform(0,1) for i in range(no_of_hidden_neurons)]                  # bias between input and hidden layer neurons

bias_2 = [random.uniform(0,1) for i in range(no_of_output_layer_neurons)]                    # bias between hidden and output layer neurons

#print()
for epoch in range(no_of_epochs):
    E1 = 0
    random.shuffle(z)
    print("epoch = %d" %epoch)
    
    for data_point in range(total_no_of_data_points):
        a = [0 for i in range(no_of_hidden_neurons)]                       # preactivation of neurons in hidden layer
        a_2 = [0 for i in range(no_of_output_layer_neurons)]               # preactivation of neurons in output layer
        t = [0 for i in range(no_of_hidden_neurons)]                       # output of neurons in hidden layer
        o = [0 for i in range(no_of_output_layer_neurons)]                         # calculated output
        del_o = [0 for i in range(no_of_output_layer_neurons)]                         # partial derivative of total errors w.r.t error signals of neurons of output layer

        del_output_act = [0 for i in range(no_of_output_layer_neurons)]                # gradient w.r.t output activation function

        del_bias_2 = [0 for i in range(no_of_output_layer_neurons)]                    # derivative w.r.t bias of neuron in output layer

        del_W_2 = [[0 for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]  # partial derivative of total errors w.r.t weights

        del_t = [0 for i in range(no_of_hidden_neurons)]                       # error signal in neurons of hidden layer

        del_t_act = [0 for i in range(no_of_hidden_neurons)]

        del_bias_1 = [0 for i in range(no_of_hidden_neurons)]                  # derivative w.r.t bias of neuron in hidden layer

        del_W_1 = [[0 for i in range(no_of_hidden_neurons)]for j in range(2)]  # partial derivative of total errors w.r.t weights

        e = [0 for i in range(no_of_output_layer_neurons)]                         # error of neurons in output layer

        E = 0

        W_2_new = [[0 for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]  # weights between output and hidden layer


        for hidden_neuron in range(no_of_hidden_neurons):

            for input_neuron in range(no_of_input_neurons):

                v = z[data_point][input_neuron] * W_1[input_neuron][hidden_neuron]
                a[hidden_neuron] = a[hidden_neuron] + v
            a[hidden_neuron] = a[hidden_neuron] + bias_1[hidden_neuron]

            t[hidden_neuron] = 1 / (1 + math.exp(-a[hidden_neuron]))


        for output_neuron in range(no_of_output_layer_neurons):

            for hidden_neuron in range(no_of_hidden_neurons):
                v = t[hidden_neuron] * W_2[hidden_neuron][output_neuron]
                a_2[output_neuron] = a_2[output_neuron] + v
            a_2[output_neuron] = a_2[output_neuron] + bias_2[output_neuron]
            o[output_neuron] = 1 / (1 + math.exp(-a_2[output_neuron]))

#            print ("z  {}".format(z[data_point][2]))
#            print ("o  {} ".format(o[output_neuron]) )
#            print()
            
            
        # Plot Misclassified ponts in training after last epoch updation
        if epoch == no_of_epochs - 1:
            if o[0] >= 0.5:
                pp = 1
            else:
                pp = 0
    
            if pp == z[data_point][2]:
    
                if z[data_point][2] == 0 :
                    pl.scatter(z[data_point][0],z[data_point][1],color='lightblue')
                else:
                    pl.scatter(z[data_point][0],z[data_point][1],color='lightsalmon')
    
            else:
                if z[data_point][2] == 0:
                    pl.plot(z[data_point][0],z[data_point][1],'k+')
                else:
                    pl.plot(z[data_point][0],z[data_point][1],'g+')

    

        #print "pp = %d" %pp


    #derivatives


        for output_neuron in range(no_of_output_layer_neurons):

            e[output_neuron] = (o[output_neuron] - z[data_point][2+output_neuron])
            #print
            E = E + (e[output_neuron])**2
        E = E/2
        E1 = E1 + E


    #derivative of error signals w.r.t output layer neurons
        for output_neuron in range(no_of_output_layer_neurons):
            del_o[output_neuron] = e[output_neuron]
            #print ("del_o = %f" %(del_o[output_neuron]))

    #gradient w.r.t output activation function
            del_output_act[output_neuron] = o[output_neuron] * (1 - o[output_neuron]) * del_o[output_neuron]
            #print ("del_output_act = %f" %(del_output_act[output_neuron]))

    #derivative w.r.t weights between hidden and output layer

        for hidden_neuron in range(no_of_hidden_neurons):

            for output_neuron in range(no_of_output_layer_neurons):

                del_W_2[hidden_neuron][output_neuron] = t[hidden_neuron] * del_output_act[output_neuron]
                #print ("del_W_2 = %f" %(del_W_2[hidden_neuron][output_neuron]))

                W_2_new[hidden_neuron][output_neuron] = W_2[hidden_neuron][output_neuron] - (learning_rate_2 * del_W_2[hidden_neuron][output_neuron])

    # derivative w.r.t bias_2
        for output_neuron in range(no_of_output_layer_neurons):
            del_bias_2[output_neuron] = 1* del_output_act[output_neuron]
            bias_2[output_neuron] = bias_2[output_neuron] - (learning_rate_2 * del_bias_2[output_neuron])

    # derivatives of signal errors in neurons of hidden layer

        for hidden_neuron in range(no_of_hidden_neurons):

            for output_neuron in range(no_of_output_layer_neurons):

                del_t[hidden_neuron] = del_t[hidden_neuron] + del_output_act[output_neuron] * W_2[hidden_neuron][output_neuron]
                #print ("del_t[%d] = %f" %(j, del_t[j]))

    #derivative w.r.t hidden_neuron activaton
        for hidden_neuron in range(no_of_hidden_neurons):
            del_t_act[hidden_neuron] = del_t[hidden_neuron]*t[hidden_neuron] * (1 - t[hidden_neuron])

    # derivative w.r.t bias_1
        for hidden_neuron in range(no_of_hidden_neurons):
            del_bias_1[hidden_neuron] = 1* del_t_act[hidden_neuron]
            bias_1[hidden_neuron] = bias_1[hidden_neuron] - (learning_rate_1 * del_bias_1[hidden_neuron])

    #derivative w.r.t weights between input and hidden layer
        for input_neuron in range(no_of_input_neurons):
            for hidden_neuron in range(no_of_hidden_neurons):
                del_W_1[input_neuron][hidden_neuron] = del_t_act[hidden_neuron] * z[data_point][input_neuron]
                #print ("del_W_1[%d][%d] = %f" %(j, k, del_W_1[j][k]))

            W_1[input_neuron][hidden_neuron] = W_1[input_neuron][hidden_neuron] - (learning_rate_1 * del_W_1[input_neuron][hidden_neuron])

        for hidden_neuron in range(no_of_hidden_neurons):

            for output_neuron in range(no_of_output_layer_neurons):
                W_2[hidden_neuron][output_neuron] = W_2_new[hidden_neuron][output_neuron]
    
    
    #Error corresponding to last training point in the epoch
    error.append(E)
    
    #Average Error in Epoch
    E1 = E1 / total_no_of_data_points
    error_1.append(E1)
    
    if epoch == no_of_epochs - 1:
        pl.title("Misclassified ponts in training after last epoch updation ")
        pl.show()
    
    #Error calculation after the last update in the epoch
    sq_e = 0
    E_2 = 0
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
        E_2 = E_2 + sq_e
    # mean sum of square of error
    E_2 = E_2 / total_no_of_data_points
    error_2.append(E_2)
        



#        if o[0] >= 0.5:
#            pp = 1
#        else:
#            pp = 0
#
#        if pp == z[data_point][2]:
#           # c = c+ 1
#            if z[data_point][2] == 0 :
#                pl.plot(z[data_point][0],z[data_point][1],'bo')
#            else:
#                pl.plot(z[data_point][0],z[data_point][1],'ro')
#
#        else:
#            if z[data_point][2] == 0:
#                pl.plot(z[data_point][0],z[data_point][1],'k+')
#            else:
#                pl.plot(z[data_point][0],z[data_point][1],'g+')
    #pl.show()
#
#    print(c)

#######################################################
########%%%%%%%%%%%%%%%%%%############   Plot of Error
#error = numpy.array(error)    
#pl.plot(error)
#pl.xlabel("No. of epochs")
#pl.ylabel("Error")
#pl.title("Error corresponding to last training point in the epoch")
#pl.show()    

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
            pl.plot(test_data[data_point][0],test_data[data_point][1],'k+')
        else:
            pl.plot(test_data[data_point][0],test_data[data_point][1],'m+')
        


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
        pl.scatter(data[i][0], data[i][1], color = 'lightsalmon')
        
    else:
        pl.scatter(data[i][0], data[i][1], color = 'lightblue')
#Plot of Misclassified points
pl.title("Plot of Misclassified points during Test")
pl.show()
