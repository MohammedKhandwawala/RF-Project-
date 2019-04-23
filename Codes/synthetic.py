# RF and Optical Communication Course Project
# Abhishek Nair, Mohit Shrivastava, Mohammed Khandwawala

import numpy as np
from scipy.signal import spectrogram, hanning,convolve , chirp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn.cluster import k_means 
from scipy.stats import skew, kurtosis
import pandas as pd
import cv2
import math
import os 
import imutils 
import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import scipy.misc
from sklearn.decomposition import NMF

# Frequencies at which external noise is added
f1 = 275
f2 = 1333
f3 = 4150
f4 = 9644
f5 = 21212
f6 = 40000
f7 = 20000

K=int(1e5)

# The signal is assumed to be gaussian in nature due to CLT
sample = np.random.normal(0,1,10*K)  # 1ms interval 

# Adding a chirp signal as a noise source
#t = np.linspace(0,1,1001)
#C = chirp(t,5000,1000,1)
#plt.plot(t,C)
#plt.show()

# Time signals 
t1 = np.linspace(0,10,10*K)
t2 = np.linspace(0,10,10*K)
t3 = np.linspace(0,10,10*K)
t4 = np.linspace(0,10,10*K)
t5 = np.linspace(0,10,10*K)
t6 = np.linspace(0,10,10*K+1)

# chirp signal
C = (0.5)*chirp(t6,f7,10,f6)

# sinusoidal signals
sino1 = (0.5)*np.sin(2*np.pi*f1*t1)
sino2 = (0.5)*np.sin(2*np.pi*f2*t2)
sino3 = (0.5)*np.sin(2*np.pi*f3*t3)
sino4 = (0.5)*np.sin(2*np.pi*f4*t4)
sino5 = (0.5)*np.sin(2*np.pi*f5*t5)


# Simulating gaussian noise, same characteristics as the signal
#x = np.random.uniform(0,10000 - 1000)

# Adding the sinusoidal and chirp signals at different positions 
sample[5*K:] = sample[5*K:]+sino1[:5*K]
sample[:5*K] = sample[:5*K]+sino2[:5*K]
sample[K:6*K] = sample[1*K:6*K]+sino3[:5*K]
sample[4*K:9*K] = sample[4*K:9*K]+sino4[:5*K]
sample[2*K:7*K] = sample[2*K:7*K]+sino5[:5*K]
sample[:] = sample[:]+C[:10*K]

# computing 512 point spectrogram with an overlap of 75%
# keeping overlap allows us to retain phase information in the spectrogram 

NFFT = 512
overlap = 0.25
overlap_samples = int(round(NFFT*overlap)) # overlap in samples

f, t , S = spectrogram(sample,nperseg=NFFT,noverlap=overlap_samples,nfft=NFFT)


print(S.shape)
S = S[:,:256]
t = range(256)
f = range(257)

# Compute average spectrum
avg_S = np.mean(S,axis=1)

# plot the spectrogram of the signal
plt.imshow(abs(S))
plt.ylabel('f')
plt.xlabel('t')
plt.title(" Spectrogram of the input signal " )
plt.colorbar()
plt.show()
'''
# Applying the hanning window to the spectrogram in order to remove noise but retain interference
h = hanning(5)
han2d = np.sqrt(np.outer(h,h))
S = convolve(S,han2d,mode="same")

# Compute average spectrum, that is average intensity at every frequency
avg_S = np.mean(S,axis=1)

plt.pcolormesh(t, f, abs(S), cmap=plt.get_cmap('jet'))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title( " Spectrogram after application of hanning window " )
plt.colorbar()
plt.show()
'''
# implementing a thresholding operation
mean = np.mean(S)
std = np.std(S)

thresh = 3*std 

G = np.zeros(S.shape)
G[abs(abs(S) -  mean) >=  thresh] = np.max(S)



Sblank = np.where(abs(abs(S) -  mean) >=  thresh, np.zeros(S.shape), S )
Sblank = (Sblank - np.min(Sblank))/(np.max(Sblank) - np.min(Sblank))

Savg = np.where(abs(abs(S) -  mean) >=  thresh, mean*np.ones(S.shape), S )
Savg = (Savg - np.min(Savg))/(np.max(Savg) - np.min(Savg))

plt.imshow(Savg)
plt.title('Mitigation Using Averaging')
plt.xlabel('t')
plt.ylabel('f')
plt.show()

np.set_printoptions(threshold=sys.maxsize)

def pre_process_image(img, save_in_file, morph_size=(8, 8)):
	#Canny edge detector
	pre = cv2.Canny(img,60,60)
	#preview of image after canny
	plt.title("Detecting RFI")	
	plt.xlabel("t")
	plt.ylabel("f")
	plt.imshow(pre)
	plt.show()
	#binnary thresholding
	pre = cv2.threshold(pre, 10, 64, cv2.THRESH_BINARY_INV)[1]	
	
	# dilate the image to have broader boundaries	
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(pre,kernel,iterations = 1)
	plt.imshow(pre)
	plt.show()
	
	pre = ~pre
	if save_in_file is not None:
		cv2.imwrite(save_in_file, pre)
	return pre


def find_text_boxes(pre, min_cell_height_limit=0, max_cell_height_limit=2):
	# Looking for the text spots contours
	contours = cv2.findContours(pre,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if imutils.is_cv2() else contours[1]
	# Getting the texts bounding boxes based on the cell size (assumptions)
	boxes = []
	for contour in contours:
		box = cv2.boundingRect(contour)
		h = box[2]
		if min_cell_height_limit < h < max_cell_height_limit:
			boxes.append(box)
	return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
	rows = {}
	cols = {}
	# Clustering the bounding boxes by their positions
	for box in boxes:
		(x, y, w, h) = box
		col_key = x // cell_threshold
		row_key = y // cell_threshold
		cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
		rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

	# Filtering out the clusters having less than 2 cols
	table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
	# Sorting the row cells by x coord
	table_cells = [list(sorted(tb)) for tb in table_cells]
	# Sorting rows by the y coord
	table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

	return table_cells

#Not used in the code
def build_lines(table_cells):
	if table_cells is None or len(table_cells) <= 0:
		return [], []

	max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
	max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

	max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
	max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

	hor_lines = []
	ver_lines = []

	for box in table_cells:
		x = box[0][0]
		y = box[0][1]
		hor_lines.append((x, y, max_x, y))

	for box in table_cells[0]:
		x = box[0]
		y = box[1]
		ver_lines.append((x, y, x, max_y))

	(x, y, w, h) = table_cells[0][-1]
	ver_lines.append((max_x, y, max_x, max_y))
	(x, y, w, h) = table_cells[0][0]
	hor_lines.append((x, max_y, max_x, max_y))

	return hor_lines, ver_lines



spec = S.copy()

z=spec.copy()

#converting array to an image
z = Image.fromarray(z)
z = np.asarray(z)
z = z*(255/np.max(z))
z = np.uint8(z)
plt.imshow(z)
plt.show()

img = np.array(z)
image_number = 'synth'
pre_file = os.path.join("pre"+ image_number +".png")
out_file = os.path.join("out"+ image_number +".png")
tilt_file = os.path.join("tilt"+ image_number +".png")


#img = cv2.imread(os.path.join(in_file))	

#pretiltilting image by 10 degrees before processing to simulate tilted image
'''
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 10, 1.0)
img = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(tilt_file, img)
'''

pre_processed = pre_process_image(img, pre_file)

text_boxes = find_text_boxes(pre_processed)
cells = find_table_in_boxes(text_boxes)
hor_lines, ver_lines = build_lines(cells)

# Visualize the result
vis = img.copy()

"""
h_count = 0
v_count = 0

coor = set([])

for line in hor_lines:
	[x1, y1, x2, y2] = line
	cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
	coor.add((x1, y1))
	coor.add((x2, y2))
	h_count+=1


for line in ver_lines:
	[x1, y1, x2, y2] = line
	cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 0), 1)
	coor.add((x1, y1))
	coor.add((x2, y2))
	v_count+=1

img = np.array(z)
"""

index = np.argwhere(z > 40)

for i in range(len(index)):
	spec[index[i][0]][index[i][1]] = 0
	z[index[i][0]][index[i][1]] = 0

z = (z - np.min(z))/(np.max(z) - np.min(z))

plt.title("Orignal Spectrogram")
plt.xlabel("t")
plt.ylabel("f")
plt.imshow(z)
plt.show()

print(z[150])
'''
z[149] = z[149]*0
z[150] = z[150]*0
z[151] = z[151]*0
'''

print("Filtered")

def matrix_factorization(R, P, Q, K, steps=1000, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print (step)	
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
        print(e)
    return P, Q.T

'''
def matrix_factorization(R, P, Q, K, steps=500, alpha=0.002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print(step)
        ij = np.where(R > 0)
        #print(type(ij[0]))
        #print(R>0)
        eij = R[ij] - np.matmul(P,Q)[ij]
        #print(eij.shape)
        ii = np.array(list(ij[0]))
        jj = np.array(list(ij[1]))
        #print(ii) 
        for k in range(K):
            P[ii][:,k] = P[ii][:,k] + alpha * (2 * np.multiply( eij, Q[k][jj] ) - beta * P[ii][:,k] )
            Q[k][jj] = Q[k][jj] + alpha * ( 2 * np.multiply( eij, P[ii][:,k] ) - beta * Q[k][jj] )
        
        e = 0;
        e = e + sum(pow(R[ij] - np.matmul(P,Q)[ij], 2))
        for k in range(K):
            e = e + sum((beta/2) * ( pow(P[ii][:,k],2) + pow(Q[k][jj],2) ))
        if e < 0.001:
            break
        print(e)
	
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
        print(e)

    return P, Q.T

'''
###############################################################################
minimum = np.min(z)
maximum = np.max(z)

plt.title("Removed RFI affected values")
plt.xlabel("t")
plt.ylabel("f")
plt.imshow(z)
plt.show()



N = len(z)
M = len(z[0])
'''
K = 64

P = np.random.rand(N,K)

Q = np.random.rand(M,K)

W,H = matrix_factorization(z, P, Q, K, steps=100)
'''

z_rep = np.load("spec.npy")
W = np.load("Parray_synth.npy")
H = np.load("Qarray_synth.npy")
z_rec = np.matmul(W,H.T)


plt.title("Reproduced using NMF (Synthetic)")
plt.xlabel("t")
plt.ylabel("f")
z_rec = np.matmul(W,H.T)
plt.imshow(z_rec)
plt.show()

plt.title("Reproduced using NMF (Synthetic)")
plt.xlabel("f")
plt.ylabel("t")
plt.imshow(z_rep)
plt.show()

print(z_rec[150])

S = (S - np.min(S))/(np.max(S) - np.min(S))

error = (S - z_rec)**2/(S.shape[0]*S.shape[1])
error = error.sum()
print("NMf to orignal error ",error)

error = (Sblank - z_rec)**2/(Sblank.shape[0]*Sblank.shape[1])
error = error.sum()
print("blank to orignal error ",error)

error = (Savg - z_rec)**2/(Savg.shape[0]*Savg.shape[1])
error = error.sum()
print("average to orignal error ",error)
