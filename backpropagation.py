
import feedforword

import random
import math
import matplotlib.pyplot as plt

def backpropogation(z, data_point, no_of_input_neurons, no_of_hidden_neurons, no_of_output_layer_neurons, W_1, W_2, bias_1,bias_2, learning_rate, a, a_2, t, o):



    e = [0 for i in range(no_of_output_layer_neurons)]                         # error of neurons in output layer
    E = 0

    #derivatives

    del_o = [0 for i in range(no_of_output_layer_neurons)]                         # partial derivative of total errors w.r.t error signals of neurons of output layer

    del_output_act = [0 for i in range(no_of_output_layer_neurons)]                # gradient w.r.t output activation function

    del_bias_2 = [0 for i in range(no_of_output_layer_neurons)]                    # derivative w.r.t bias of neuron in output layer

    del_W_2 = [[0 for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]  # partial derivative of total errors w.r.t weights

    del_t = [0 for i in range(no_of_hidden_neurons)]                       # error signal in neurons of hidden layer

    del_t_act = [0 for i in range(no_of_hidden_neurons)]

    del_bias_1 = [0 for i in range(no_of_hidden_neurons)]                  # derivative w.r.t bias of neuron in hidden layer

    del_W_1 = [[0 for i in range(no_of_hidden_neurons)]for j in range(2)]  # partial derivative of total errors w.r.t weights

    W_2_new = [[random.uniform(-1,1) for i in range(no_of_output_layer_neurons)]for j in range(no_of_hidden_neurons)]

    for output_neuron in range(no_of_output_layer_neurons):

        e[output_neuron] = (o[output_neuron] - z[data_point][2+output_neuron])
        
        E = E + (e[output_neuron]**2)
    E = E/2



#derivative w.r.t error signals of output layer neurons
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

            W_2_new[hidden_neuron][output_neuron] = W_2[hidden_neuron][output_neuron] - (learning_rate * del_W_2[hidden_neuron][output_neuron])

# derivative w.r.t bias_2
    for output_neuron in range(no_of_output_layer_neurons):
        del_bias_2[output_neuron] = 1* del_output_act[output_neuron]
        bias_2[output_neuron] = bias_2[output_neuron] - (learning_rate * del_bias_2[output_neuron])

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
        bias_1[hidden_neuron] = bias_1[hidden_neuron] - (learning_rate * del_bias_1[hidden_neuron])

#derivative w.r.t weights between input and hidden layer
    for input_neuron in range(no_of_input_neurons):
        for hidden_neuron in range(no_of_hidden_neurons):
            del_W_1[input_neuron][hidden_neuron] = del_t_act[hidden_neuron] * z[data_point][input_neuron]
            #print ("del_W_1[%d][%d] = %f" %(j, k, del_W_1[j][k]))

        W_1[input_neuron][hidden_neuron] = W_1[input_neuron][hidden_neuron] - (learning_rate * del_W_1[input_neuron][hidden_neuron])


    for hidden_neuron in range(no_of_hidden_neurons):

        for output_neuron in range(no_of_output_layer_neurons):

            W_2[hidden_neuron][output_neuron] = W_2_new[hidden_neuron][output_neuron]
        
#    print(E)

    return(W_1, W_2, bias_1, bias_2, E)

