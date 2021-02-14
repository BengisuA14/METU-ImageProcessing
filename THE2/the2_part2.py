# Bengisu Ayan - 2236974
# Ceren Gürsoy - 2237485

from math import sqrt
import numpy as np
import cv2


# fourier transform the image
def fourier_transform(image):
    image_f_trans = np.fft.fft2(image)
    image_f_shift = np.fft.fftshift(image_f_trans)  # flip the corners

    return image_f_shift


# butterworth filter
def butter_worth_filter(size_x, size_y, r, n):
    b_filter = np.zeros([size_x, size_y])
    for u in range(size_x):
        for v in range(size_y):
            val = pow((u - int(size_x / 2)), 2) + pow((v - int(size_y / 2)), 2)
            if val != 0:  # to prevent zero division
                x = pow(r / sqrt(val), 2 * n)
                b_filter[u][v] = 1 / (1 + x)

    return b_filter


# apply butterworth filter to image
def apply_filter(f_shifted_image, r, n):
    b_filter = butter_worth_filter(f_shifted_image.shape[0], f_shifted_image.shape[1], r, n)
    filtered_image = np.multiply(f_shifted_image, b_filter)

    return filtered_image


# inverse transform the image
def inverse_fourier_transform(filtered_image):
    image_inv_f_shift = np.fft.ifftshift(filtered_image)  # flip the corners back
    image_inv_f_trans = np.fft.ifft2(image_inv_f_shift)

    return image_inv_f_trans


B1 = cv2.imread('B1.jpg', 0)  # read image as grayscale
B2 = cv2.imread('B2.jpg', 0)  # read image as grayscale
B3 = cv2.imread('B3.jpg', 0)  # read image as grayscale

b1_f_shift = fourier_transform(B1)  # fourier transform the image
b1_filtered = apply_filter(b1_f_shift, 80, 2)  # apply butterworth filter to image with r=80 and n=2
b1_inv_f_shift = inverse_fourier_transform(b1_filtered)  # inverse transform the image
b1_abs = np.abs(b1_inv_f_shift)  # take absolute in order to save
cv2.imwrite('B1_output.png', b1_abs)  # save image

b2_f_shift = fourier_transform(B2)  # fourier transform the image
b2_filtered = apply_filter(b2_f_shift, 80, 2)  # apply butterworth filter to image with r=80 and n=2
b2_inv_f_shift = inverse_fourier_transform(b2_filtered)  # inverse transform the image
b2_abs = np.abs(b2_inv_f_shift)  # take absolute in order to save
cv2.imwrite('B2_output.png', b2_abs)  # save image

b3_f_shift = fourier_transform(B3)  # fourier transform the image
b3_filtered = apply_filter(b3_f_shift, 50, 2)  # apply butterworth filter to image with r=50 and n=2
b3_inv_f_shift = inverse_fourier_transform(b3_filtered)  # inverse transform the image
b3_abs = np.abs(b3_inv_f_shift)  # take absolute in order to save
cv2.imwrite('B3_output.png', b3_abs)  # save image