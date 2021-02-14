# Bengisu Ayan - 2236974
# Ceren Gürsoy - 2237485

import numpy as np
import cv2

A1 = "THE3-Images/A1.png"
A2 = "THE3-Images/A2.png"
A3 = "THE3-Images/A3.png"
A4 = "THE3-Images/A4.png"
A5 = "THE3-Images/A5.png"
A6 = "THE3-Images/A6.png"

# read A1
a1 = cv2.imread(A1, cv2.IMREAD_GRAYSCALE)

# binarize A1
a1_thresh = 80
a1_binarized = np.uint8((a1 < a1_thresh) * 255)

# define 3*3 rectangular structuring element
a1_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# apply closing
a1_out = cv2.morphologyEx(a1_binarized, cv2.MORPH_CLOSE, a1_kernel)

# count connected components
count1, _ = cv2.connectedComponents(a1_out)

print("The number of flying jets in image A1 is", count1 - 1)
cv2.imwrite('part1_A1.png', a1_out)

####################################################################################

# read A2
a2 = cv2.imread(A2, cv2.IMREAD_GRAYSCALE)

# binarize A2
a2_thresh1 = 110
a2_thresh2 = 75
a2_bool = (a2 < a2_thresh1) & (a2 > a2_thresh2)
a2_binarized = np.uint8(a2_bool * 255)

# define 5*5 rectangular structuring element for opening
a2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# apply opening
a2_out = cv2.morphologyEx(a2_binarized, cv2.MORPH_OPEN, a2_kernel)

# define 45*45 rectangular structuring elememnt for closing
a2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
# apply closing
a2_out = cv2.morphologyEx(a2_out, cv2.MORPH_CLOSE, a2_kernel)

# count connected components
count2, _ = cv2.connectedComponents(a2_out)

print("The number of flying jets in image A2 is",count2-1)
cv2.imwrite('part1_A2.png', a2_out)

####################################################################################

# read A3
a3 = cv2.imread(A3, cv2.IMREAD_GRAYSCALE)

# binarize A3
a3_thresh = 60
a3_binarized = np.uint8((a3 < a3_thresh) * 255)

# define 3*3 elliptical structuring element
a3_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# apply opening
a3_out = cv2.morphologyEx(a3_binarized, cv2.MORPH_OPEN, a3_kernel)

# define 5*5 elliptical structural element
a3_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# apply closing
a3_out = cv2.morphologyEx(a3_out, cv2.MORPH_CLOSE, np.array(a3_kernel, dtype=np.uint8))

# count connected components
count3, _ = cv2.connectedComponents(a3_out)

print("The number of flying jets in image A3 is", count3-1)
cv2.imwrite('part1_A3.png', a3_out)
####################################################################################

# read A4
a4 = cv2.imread(A4, cv2.IMREAD_GRAYSCALE)

# binarize A4
a4_thresh = 50
a4_binarized = np.uint8((a4 < a4_thresh) * 255)

# define 3*3 rectangular structuring element for erosion
a4_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# apply erosion
a4_out = cv2.erode(a4_binarized, a4_kernel, iterations=1)

# define 7*7 elliptical structuring element for closing
a4_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# apply closing
a4_out = cv2.morphologyEx(a4_out, cv2.MORPH_CLOSE, np.array(a4_kernel, dtype=np.uint8))

# define 3*3 rectangular structuring element for opening
a4_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# apply opening
a4_out = cv2.morphologyEx(a4_out, cv2.MORPH_OPEN, np.array(a4_kernel, dtype=np.uint8))

# define 6*6 rectangular structuring element for dilation
a4_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
# apply dilation
a4_out = cv2.dilate(a4_out, a4_kernel, iterations=1)

# count connected components
count4, _ = cv2.connectedComponents(a4_out)

print("The number of flying jets in image A4 is", count4-1)
cv2.imwrite('part1_A4.png', a4_out)

####################################################################################

# read a5
a5 = cv2.imread(A5)
b = a5[:, :, 0]

# binarize A5
a5_thresh = 150
a5_binarized = np.uint8((b < a5_thresh) * 255)

# define 5*5 elliptical structuring element for erosion
a5_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# apply erosion
a5_out = cv2.erode(a5_binarized, a5_kernel, iterations=1)

# define 7*7 elliptical structuring element for opening
a5_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# apply opening
a5_out = cv2.morphologyEx(a5_out, cv2.MORPH_OPEN, np.array(a5_kernel, dtype=np.uint8))

# define 7*7 rectangular structuring element for dilation
a5_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
# apply dilation
a5_out = cv2.dilate(a5_out, a5_kernel, iterations=1)

# define 11*11 rectangular structuring element for closing
a5_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# apply closing
a5_out = cv2.morphologyEx(a5_out, cv2.MORPH_CLOSE, np.array(a5_kernel, dtype=np.uint8))

# count connected components
count5, _ = cv2.connectedComponents(a5_out)

print("The number of flying jets in image A5 is", count5 - 1)
cv2.imwrite('part1_A5.png', a5_out)

####################################################################################

# read A6
a6 = cv2.imread(A6, cv2.IMREAD_GRAYSCALE)

# binarize A6
a6_thresh = 15
a6_binarized = np.uint8((a6 < a6_thresh) * 255)

# define 4*4 elliptical structuring element for opening
a6_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
# apply opening
a6_out = cv2.morphologyEx(a6_binarized, cv2.MORPH_OPEN, np.array(a6_kernel, dtype=np.uint8))

# define 5*5 elliptical structuring element for dilation
a6_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# apply dilation
a6_out = cv2.dilate(a6_out,a6_kernel,iterations = 1)

# define 11*11 rectangular structuring element for closing
a6_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
# apply closing
a6_out = cv2.morphologyEx(a6_out, cv2.MORPH_CLOSE, np.array(a6_kernel, dtype=np.uint8))

# remove the component with mountain using a mask
ret, labels = cv2.connectedComponents(a6_out)

mask = np.array(labels, dtype=np.uint8)
mask[labels == 6] = 255
a6_out = a6_out - mask

# count connected components
count6, labels = cv2.connectedComponents(a6_out)

print("The number of flying jets in image A6 is", count6 - 1)
cv2.imwrite('part1_A6.png', a6_out)
