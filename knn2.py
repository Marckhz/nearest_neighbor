import numpy as np
import statistics as st
from PIL import Image, ImageDraw


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
	j = j+1
	l = 0
	h = 0
	for k in values:
		while l<k:
			moda.append(G[l][1])
			l = l +1
		m = st.mode(moda)
		if(m==1):
			imagenes[h].append(255)
		else:
			imagenes[h].append(0)
		h=h+1
g = 0
while g < 5:
	array_ = np.array([imagenes[g]])
	shape_here = np.reshape( array_, (512, 512))
	img_Array = Image.fromarray(np.uint8(shape_here))
	write_k = ImageDraw.Draw(img_Array)
	#write_k.text((10, 10), "When K is %d", fill=(255,255,0) ) % (values[g])
	write_k.text( (24,24), "When K equals %d" % (values[g]), fill='yellow')
	img_Array.save("when k is %d.bmp" % (values[g])  )
	g = g + 1
