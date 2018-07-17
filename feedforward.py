

import random
import math
import matplotlib.pyplot as pl


def feedforward(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, iiii):


    a = [0 for i in range(no_of_hidden_neurons)]                       # preactivation of neurons in hidden layer
    a_2 = [0 for i in range(no_of_output_layer_neurons)]               # preactivation of neurons in output layer
    t = [0 for i in range(no_of_hidden_neurons)]                       # output of neurons in hidden layer
    o = [0 for i in range(no_of_output_layer_neurons)]                         # calculated output

#    print("W_1 : {}" .format(W_1))
#    print("vfhcc")
#    print("W_2 : {}" .format(W_2))
#    print()
#    print("bias_1 : {}" .format(bias_1))
#    print()
#    print("bias_2 : {}" .format(bias_2))

    #loop through neurons in hiddden layer
    for hidden_neuron in range(no_of_hidden_neurons):
	
	#loop through the input neurons(connected with the hidden neuron)
        for input_neuron in range(no_of_input_neurons):

            v = z[data_point][input_neuron] * W_1[input_neuron][hidden_neuron]
            a[hidden_neuron] = a[hidden_neuron] + v
        a[hidden_neuron] = a[hidden_neuron] + bias_1[hidden_neuron]

        t[hidden_neuron] = 1 / (1 + math.exp(-a[hidden_neuron]))

    # loop through the neurons in output layer 
    for output_neuron in range(no_of_output_layer_neurons):
	# loop through the hidden neurons connected to the neurons in output layer  
        for hidden_neuron in range(no_of_hidden_neurons):
            v = t[hidden_neuron] * W_2[hidden_neuron][output_neuron]
            a_2[output_neuron] = a_2[output_neuron] + v
        a_2[output_neuron] = a_2[output_neuron] + bias_2[output_neuron]
        o[output_neuron] = 1 / (1 + math.exp(-a_2[output_neuron]))

        #print "o[%d]= " %j
#        if iiii == 1:
#            if o[output_neuron] >= 0.5:
#                pp = 1
#            else:
#                pp = 0
#    
#            if pp == z[data_point][2]:
#    
#                if z[data_point][2] == 0 :
#                    pl.scatter(z[data_point][0],z[data_point][1],color='lightblue')
#                else:
#                    pl.scatter(z[data_point][0],z[data_point][1],color='lightsalmon')
#    
#            else:
#                if z[data_point][2] == 0:
#                    pl.plot(z[data_point][0],z[data_point][1],'k+')
#                else:
#                    pl.plot(z[data_point][0],z[data_point][1],'g+')
#
##
#    pl.show()
    #print "pp = %d" %pp

    return(a, a_2, t, o)




