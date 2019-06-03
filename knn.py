import numpy as np
import math
#import cv2

from PIL import Image

trainingdata = np.loadtxt("rsTrain.txt")

x_train = trainingdata[:,0:4]
x_train1=  trainingdata[:,0:1]
x_train2=  trainingdata[:,1:2]
x_train3=  trainingdata[:,2:3]
x_train4=  trainingdata[:,3:4]
print(x_train2)


y = trainingdata[:,-1]
print(y)
#print(x_train.shape)


#x_train2 = np.reshape(x_train, (800))
#print(x_train2)

#for x in x_train2:
 #   print(x)

#print(x_train)

#img1 = Image.fromarray(testdata)
#img1.show()

band1 = np.fromfile('band1.irs', dtype ='uint8')


band2 = np.fromfile('band2.irs', dtype='uint8')
band3 = np.fromfile('band3.irs', dtype='uint8')
band4 = np.fromfile('band4.irs', dtype='uint8')



reshape_band1  = np.reshape(band1, (512,512) )
reshape_band2 = np.reshape(band2, (512, 512) )
reshape_band3  = np.reshape(band3, (512, 512) )
reshape_band4 = np.reshape(band4, (512, 512) )


#for x in x_:
 #   print(element)


A = np.array([])
for x in x_train1:
   A = x - band1
   #print(x)
   #print(A)
A = np.square(A)
print(A.size)

#print(np.sort(A))


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

#print(A[0])
#print(B[0])
#print(C[0])
#print(D[0])


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







