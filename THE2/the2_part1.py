# Bengisu Ayan - 2236974
# Ceren Gürsoy - 2237485

from numpy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import cv2
import math

A1 = "A1.png"
A2 = "A2.png"
A3 = "A3.png" 

# butterworth filter
def butterworth_filter(size_x, size_y, r, n):
    filt = np.zeros([size_x, size_y])
    for u in range(size_x):
        for v in range(size_y):
            val = ((((((u-int(size_x/2)) ** 2)) + ((v-int(size_y/2)) ** 2))/ r) ** (2*n))
            filt[u][v] = ((1.0 + val) ** (-1))
    return filt

# bandreject filter for image2
def image2_filter(size_x, size_y):
    center_x = size_x / 2
    center_y = size_y / 2
    filt = np.ones([size_x, size_y])
    for i in range(size_x):
        for j in range(size_y):
            distance = math.sqrt(((i - center_x)**2) + ((j - center_y)**2))
            if (distance > 63 and distance < 85) or (distance > 280 and distance < 320):
                filt[i][j] = 0
    return filt

# band reject filter for blue channel
def image3_blue_filter(size_x, size_y):
    filt = np.ones([size_x, size_y])
    for i in range(size_x):
        for j in range(size_y):
            if (j > 1344 and j < 1352) and (i>311 and i<321):
                filt[i][j] = 0
            if (j > 1437 and j < 1447) and (i>252 and i<262):
                filt[i][j] = 0
            if (j > 507 and j < 517) and (i>824 and i<834):
                filt[i][j] = 0
            if (j > 414 and j < 424) and (i>880 and i<890):
                filt[i][j] = 0
            if (j > 910 and j < 940) and (i>0 and i<500):
                filt[i][j] = 0
            if (j > 910 and j < 940) and (i>700 and i<1141):
                filt[i][j] = 0
            if (j > 0 and j < 700) and (i>560 and i<570):
                filt[i][j] = 0
            if (j > 1200 and j < 1857) and (i>560 and i<570):
                filt[i][j] = 0
            if (i > 539 and i < 544) or (i > 595 and i < 602) or (j > 880 and j < 884) or (j > 972 and j < 979):
                filt[i][j] = 0
            if (i > 480 and i < 525) and (j > 1000 and j < 1100):
                filt[i][j] = 0
            if (i > 620 and i < 670) and (j > 750 and j < 850):
                filt[i][j] = 0
    return filt


# band reject filter for green channel
def image3_green_filter(size_x, size_y):
    filt = np.ones([size_x, size_y])
    for i in range(size_x):
        for j in range(size_y):
            if (j>1528 and j<1538 and i>197 and i <204) or (j>1435 and j<1444 and i>254 and i<263) or (j>1344 and j<1352 and i>311 and i<320):
                filt[i][j] = 0
            if (j>507 and j<514 and i>824 and i<832) or (j>413 and j<420 and i>881 and i<889) or (j>322 and j<330 and i>940 and i<948):
                filt[i][j] = 0
            if (j>1060 and j<1075 and i>480 and i<495) or (j>780 and j<795 and i>650 and i<665) or (j>1015 and j<1030 and i>510 and i<520) or (j>825 and j<845 and i>623 and i<633):
                filt[i][j] = 0
            if (j > 910 and j < 940) and (i>0 and i<500) and (j > 910 and j < 940) and (i>700 and i<1141) and (j > 0 and j < 700) and (i>560 and i<570):
                filt[i][j] = 0
            if (i > 539 and i < 543) or (i > 595 and i < 602) or (j > 880 and j < 884) or (j > 972 and j < 978):
                filt[i][j] = 0
    return filt


# band reject filter for red channel
def image3_red_filter(size_x, size_y):
    filt = np.ones([size_x, size_y])
    for i in range(size_x):
        for j in range(size_y):
            if (j > 507 and j < 513) and (i>309 and i<316):
                filt[i][j] = 0
            if (j > 412 and j < 422) and (i>251 and i<261):
                filt[i][j] = 0
            if (j > 1342 and j < 1352) and (i>824 and i<832):
                filt[i][j] = 0
            if (j > 1435 and j < 1445) and (i>880 and i<888):
                filt[i][j] = 0
            if (j > 910 and j < 940) and (i>0 and i<500):
                filt[i][j] = 0
            if (j > 910 and j < 940) and (i>700 and i<1141):
                filt[i][j] = 0
            if (j > 0 and j < 800) and (i>560 and i<570):
                filt[i][j] = 0
            if (j > 1000 and j < 1857) and (i>560 and i<570):
                filt[i][j] = 0
            if (i > 539 and i < 543) or (i > 594 and i < 602) or (j > 880 and j < 884) or (j > 972 and j < 979):
                filt[i][j] = 0
            if (i > 480 and i < 525) and (j > 750 and j < 850):
                filt[i][j] = 0
            if (i > 620 and i < 670) and (j > 1000 and j < 1100):
                filt[i][j] = 0

    return filt



def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


#######################  FIRST IMAGE   ########################

# read image
a1 = cv2.imread(A1, 0)

# fourier transform the image
a1_ft = fft2(a1)

# flip the corners
a1_ft_shifted = fftshift(a1_ft)

# butterworth filter
butterworth_filt = butterworth_filter(a1.shape[0], a1.shape[1], 480, 0.48)

# apply butterworth filter to image
a1_ft_filtered = a1_ft_shifted * butterworth_filt

# flip the corners back and inverse transform it
a1_denoised = ifft2(ifftshift(a1_ft_filtered))
cv2.imwrite("A1_denoised.png", np.abs(a1_denoised))

#######################  SECOND IMAGE   ########################

# read image
a2 = cv2.imread(A2, 0)

# fourier transform it
a2_ft = fft2(a2)

# flip the corners
a2_ft_shifted = fftshift(a2_ft)

#band pass filter
bandpass_filt = image2_filter(a2.shape[0], a2.shape[1])

#apply bandpass filter
a2_ft_filtered = a2_ft_shifted * bandpass_filt

# flip the corners back and inverse transform it
a2_denoised = ifft2(ifftshift(a2_ft_filtered))

# adjust color
a2_flatten = np.asarray(np.abs(a2_denoised))
a2_imadjust = imadjust(a2_flatten,a2_flatten.min(),a2_flatten.max(),0,255)
cv2.imwrite("A2_denoised.png", a2_imadjust)


#######################  THIRD IMAGE   ########################

# read image
a3 = cv2.imread(A3)

# separate into 3 channels
blue, green, red = a3[:, :, 0], a3[:, :, 1], a3[:, :, 2]

# apply fourier transform to each channel
blue_ft = fft2(blue)
green_ft = fft2(green)
red_ft = fft2(red)

# flip the corners
blue_ft_shifted = fftshift(blue_ft)
green_ft_shifted = fftshift(green_ft)
red_ft_shifted = fftshift(red_ft)

# design different filters for each channel
blue_filter = image3_blue_filter(blue.shape[0], blue.shape[1])
green_filter = image3_green_filter(green.shape[0], green.shape[1])
red_filter = image3_red_filter(red.shape[0], red.shape[1])

# apply filters to each channel
blue_ft_filtered = blue_ft_shifted * blue_filter
green_ft_filtered = green_ft_shifted * green_filter
red_ft_filtered = red_ft_shifted * red_filter

# flip the corners and inverse transform the channels
blue_denoised = ifft2(ifftshift(blue_ft_filtered))
green_denoised = ifft2(ifftshift(green_ft_filtered))
red_denoised = ifft2(ifftshift(red_ft_filtered))

# merge the channels
image3_denoised = np.dstack((np.abs(blue_denoised), np.abs(green_denoised), np.abs(red_denoised)))
cv2.imwrite("A3_denoised.png", image3_denoised)
