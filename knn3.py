import numpy as np
import math
#import cv2

import statistics as st
from PIL import Image

trainingdata = np.loadtxt("rsTrain.txt")

x_train = trainingdata[:,0:4]
x_train1=  trainingdata[:,0:1]
x_train2=  trainingdata[:,1:2]
x_train3=  trainingdata[:,2:3]
x_train4=  trainingdata[:,3:4]

band1 = np.fromfile('band1.irs', dtype ='uint8')
band2 = np.fromfile('band2.irs', dtype='uint8')
band3 = np.fromfile('band3.irs', dtype='uint8')
band4 = np.fromfile('band4.irs', dtype='uint8')

reshape_band1  = np.reshape(band1, (512,512) )
reshape_band2 = np.reshape(band2, (512, 512) )
reshape_band3  = np.reshape(band3, (512, 512) )
reshape_band4 = np.reshape(band4, (512, 512) )
values = [3,5,7,5,21]
j = 0
imagenes = list(list())
while j < 266144:
	A = np.square(x_train1- band1[j]) 
	B = np.square(x_train2- band2[j]) 
	C = np.square(x_train3- band3[j]) 
	D = np.square(x_train4- band4[j])
	E = np.sqrt(A+B+C+D)
	F = np.array([])
	i = 0
	for x in E:
	    if i <= 100:
	        F.np.append([x , 1])
	    else:
	        F.np.append([x , 0])
	    i = i+1
	G = np.sort(F)
	#print(G[0][1])
	print(F)
	j = j+1
	h = 0
	#for k in values:
		#print(k)
		#m = st.mode(G[1][0:k])
		#if(m==1):
		#	imagenes[h].append(255)
		#	h=h+1



"""A = np.array([])
for x in x_train1:
   A = x - band1
A = np.square(A)


B = np.array([])
for x in x_train2:
    B = x - band2
B = np.square(B)


C = np.array([])
for x in x_train3:
    C = x - band3
C = np.square(C)


D = np.array([])
for x in x_train4:
    D = x - band4
D = np.square(D)

f_of_y = (A + B + C + D)
#print(f_of_y)

square_distance = np.sqrt(f_of_y)
#X  = np.reshape(square_distance, (512, 512) )



#print("%d bytes" % ( square_distance.size * square_distance.itemsize  ) )

#E = np.array([])
#i = 0
#for x in square_distance:
#    if i <= 100:
#        E= np.append([x , 1])
#    else:
#        E= np.append([x , 0])
#    i = i+1



zero = np.zeros(100)
print(zero)



ones = np.ones(100)
print(ones)







#print(E)
square_shape  = np.reshape(square_distance, (512, 512))
#print(square_shape)
#print(np.sort(square_shape))


#print(trainingdata.shape)





"""

