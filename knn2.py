import numpy as np
import math
#import cv2

import statistics as st
from PIL import Image

print("espero por favor ;v")

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
values = [3,5,7,15,21]
j = 0
imagenes = [list(),list(),list(),list(),list()]
while j < 262144:
	A = np.square(x_train1- band1[j]) 
	B = np.square(x_train2- band2[j]) 
	C = np.square(x_train3- band3[j]) 
	D = np.square(x_train4- band4[j])
	E = np.sqrt(A+B+C+D)
	F,moda = list(),list()
	i = 0
	for x in E:
	    if i <= 100:
	        F.append([x , 1])
	    else:
	        F.append([x , 0])
	    i = i+1
	G = sorted(F)
	#print(G[0][1])
	j = j+1
	l = 0
	h = 0
	for k in values:
		while l<k:
			moda.append(G[l][1])
			l = l +1
			#print(moda)
		m = st.mode(moda)
		#print(m)
		if(m==1):
			imagenes[h].append(255)
		else:
			imagenes[h].append(0)
		h=h+1
	print(j)
#print("por favor oni chan")


g = 0
print(imagenes)
while g < 5:
	array_ = np.array([imagenes[g]])

	shape_here = np.reshape( array_, (512, 512))
	#print(shape_here)
	#print(shape_here.shap
	img_Array = Image.fromarray(np.uint8(shape_here))
	img_Array.show()
	g = g + 1


#imagen_array = np.array([imagen_uno])

#reshape_img = np.reshape(imagenes[0], (24, 24) )
#print(reshape_img)
#print(reshape_img.size)
#myImg = Image.fromarray(imagenes[0])
#myImg.show()




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

