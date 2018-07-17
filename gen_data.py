###################### Generate data between -5 and 5 #############################

import random
from random import shuffle
import math
import numpy
import matplotlib.pyplot as plt

def data_generator(no, flag):
    
    z =[[]]
    
    i = 0

    #data for class1
    while (i < no):
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
        
	radius = 4 	# radius of circle.
        
        if (x)**2 + (y)**2 > radius:
            z[i].append(x)
            z[i].append(y)
            z[i].append(1)
            plt.scatter(z[i][0],z[i][1],color='lightsalmon')
            i = i + 1
            #print (z)
            
    
            if i < no:
                z.append([])
            
    #data for class1
    while (i < 2*no):   
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
        if (x)**2 + (y)**2 <= radius:
            z.append([])
            z[i].append(x)
            z[i].append(y)
            z[i].append(0)
            plt.scatter(z[i][0],z[i][1],color='lightblue')
            i = i + 1
        
            
#    z = numpy.array(z) 
#    print (z)
    
    print (len(z))
    
    #print (z.shape)

    if flag == 1:
        plt.title("Test Data Points")
    else:
        plt.title("Train Data Points")
        
    plt.show()
    return(z)

#data_generator(1000)
