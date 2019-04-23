#importing required libs
import cv2
import numpy as np
import math
import os 
import imutils 
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import scipy.misc
from sklearn.decomposition import NMF

#to print full numpy array
np.set_printoptions(threshold=sys.maxsize)

def pre_process_image(img, save_in_file, morph_size=(8, 8)):
	#Canny edge detector
	pre = cv2.Canny(img,40,20)
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


#select image number for the file to choose
image_number = "3"
x = np.load("X_pol_"+image_number+".npy")
y = np.load("Y_pol_"+image_number+".npy")
spec = np.sqrt((x**2 + y**2)/2)

spec = spec[:,:256]
print(spec.shape)
z=spec.copy()

#converting array to an image
z = Image.fromarray(z)
z = np.asarray(z)
z = z*(255/np.max(z))
z = np.uint8(z)
plt.title("Spectrogram of data from CAS")
plt.xlabel('t')
plt.ylabel('f')
plt.imshow(z)
plt.show()

img = np.array(z)

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

index = np.argwhere(z > 30)

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

K = 64

P = np.random.rand(N,K)

Q = np.random.rand(M,K)

#W,H = matrix_factorization(z, P, Q, K, steps=50)


z_rep = np.load("spec.npy")
W = np.load("Parray.npy")
H = np.load("Qarray.npy")

plt.title("Reproduced using NMF")
plt.xlabel("t")
plt.ylabel("f")
z_rec = np.matmul(W,H.T)
plt.imshow(z_rec)
plt.show()

plt.title("Reproduced using NMF")
plt.xlabel("f")
plt.ylabel("t")
plt.imshow(z_rep)
plt.show()

print(z_rec[150])

print("gg")
