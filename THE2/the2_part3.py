# Bengisu Ayan - 2236974
# Ceren Gürsoy - 2237485

import numpy as np
import cv2
import pywt
import os
import pandas as pd

C1 = "C1"
C2 = "C2"
C3 = "C3"


def compress(path_to_input_file, path_to_compressed_output_file):
    # read image
    image = cv2.imread(path_to_input_file + '.png', 0)
    cv2.imwrite(path_to_input_file + '_gray.png', image)

    # wavelet transform the image with bior4.4 and level 4
    wavelet_transformed = pywt.wavedec2(image, 'bior4.4', level=4)

    # transform coefficients to array format
    coeffs_array, coeff_slices = pywt.coeffs_to_array(wavelet_transformed)

    # first threshold values
    # keep first %10 biggest coeffs
    keep = 0.1
    sorted_coeffs = np.sort(np.abs(coeffs_array.reshape(-1)))
    threshold = sorted_coeffs[int(np.floor((1 - keep) * len(sorted_coeffs)))]
    ind = np.abs(coeffs_array) > threshold
    coeffs_left = coeffs_array * ind

    # quantize the floats to int, keep them as int16 since we do not have as large values as int32
    coeffs_array_quantized = coeffs_left.astype(np.int16)

    # save the coeffs and slices as npz file
    np.savez_compressed(path_to_compressed_output_file, a=coeffs_array_quantized, b=coeff_slices)


def decompress(path_to_compressed_input_file, path_to_output_file):
    # read compressed file
    inp = np.load(path_to_compressed_input_file + '.npz', allow_pickle=True)

    # get coeffs and slices as arrays
    coeffs_array = inp['a']
    slices = inp['b']

    # transform array format to coefficients
    coeffs = pywt.array_to_coeffs(coeffs_array, slices, 'wavedec2')

    # inverse wavelet transform
    image = pywt.waverec2(coeffs, wavelet='bior4.4')

    # save the decompressed image
    cv2.imwrite(path_to_output_file + '.png', image)


def mse(path_to_original, path_to_decompressed):
    # read original image
    A = cv2.imread(path_to_original, 0)

    # read decompressed image
    B = cv2.imread(path_to_decompressed, 0)

    # measure MSE
    val = np.sum(np.square(np.subtract(A, B)))
    mse = val / (A.shape[0] * A.shape[1])

    return mse


compress(C1, C1 + '_compressed')
decompress(C1 + '_compressed', C1 + '_decompressed')
compress(C2, C2 + '_compressed')
decompress(C2 + '_compressed', C2 + '_decompressed')
compress(C3, C3 + '_compressed')
decompress(C3 + '_compressed', C3 + '_decompressed')

# Our algorithm MSE

mse_our_c1 = mse(C1 + '_gray.png', C1 + '_decompressed.png')
mse_our_c2 = mse(C2 + '_gray.png', C2 + '_decompressed.png')
mse_our_c3 = mse(C3 + '_gray.png', C3 + '_decompressed.png')

# Our algorithm Compression ratio

ratio_our_c1 = os.path.getsize(C1 + '_gray.png') / os.path.getsize(C1 + '_compressed.npz')
ratio_our_c2 = os.path.getsize(C2 + '_gray.png') / os.path.getsize(C2 + '_compressed.npz')
ratio_our_c3 = os.path.getsize(C3 + '_gray.png') / os.path.getsize(C3 + '_compressed.npz')

# save JPEG for qualities 1 10 50 80 for C1, C2, C3

c1 = cv2.imread(C1 + '_gray.png', 0)
cv2.imwrite(C1 + '_1.jpeg', c1, [int(cv2.IMWRITE_JPEG_QUALITY), 1])
cv2.imwrite(C1 + '_10.jpeg', c1, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
cv2.imwrite(C1 + '_50.jpeg', c1, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
cv2.imwrite(C1 + '_80.jpeg', c1, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

c2 = cv2.imread(C2 + '_gray.png', 0)
cv2.imwrite(C2 + '_1.jpeg', c2, [int(cv2.IMWRITE_JPEG_QUALITY), 1])
cv2.imwrite(C2 + '_10.jpeg', c2, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
cv2.imwrite(C2 + '_50.jpeg', c2, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
cv2.imwrite(C2 + '_80.jpeg', c2, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

c3 = cv2.imread(C3 + '_gray.png', 0)
cv2.imwrite(C3 + '_1.jpeg', c3, [int(cv2.IMWRITE_JPEG_QUALITY), 1])
cv2.imwrite(C3 + '_10.jpeg', c3, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
cv2.imwrite(C3 + '_50.jpeg', c3, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
cv2.imwrite(C3 + '_80.jpeg', c3, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

# ratio for JPEG 1 10 50 80 for C1, C2, C3
ratio_jpeg1_c1 = os.path.getsize(C1 + '_gray.png') / os.path.getsize(C1 + '_1.jpeg')
ratio_jpeg1_c2 = os.path.getsize(C2 + '_gray.png') / os.path.getsize(C2 + '_1.jpeg')
ratio_jpeg1_c3 = os.path.getsize(C3 + '_gray.png') / os.path.getsize(C3 + '_1.jpeg')

ratio_jpeg10_c1 = os.path.getsize(C1 + '_gray.png') / os.path.getsize(C1 + '_10.jpeg')
ratio_jpeg10_c2 = os.path.getsize(C2 + '_gray.png') / os.path.getsize(C2 + '_10.jpeg')
ratio_jpeg10_c3 = os.path.getsize(C3 + '_gray.png') / os.path.getsize(C3 + '_10.jpeg')

ratio_jpeg50_c1 = os.path.getsize(C1 + '_gray.png') / os.path.getsize(C1 + '_50.jpeg')
ratio_jpeg50_c2 = os.path.getsize(C2 + '_gray.png') / os.path.getsize(C2 + '_50.jpeg')
ratio_jpeg50_c3 = os.path.getsize(C3 + '_gray.png') / os.path.getsize(C3 + '_50.jpeg')

ratio_jpeg80_c1 = os.path.getsize(C1 + '_gray.png') / os.path.getsize(C1 + '_80.jpeg')
ratio_jpeg80_c2 = os.path.getsize(C2 + '_gray.png') / os.path.getsize(C2 + '_80.jpeg')
ratio_jpeg80_c3 = os.path.getsize(C3 + '_gray.png') / os.path.getsize(C3 + '_80.jpeg')

# mse for JPEG 1 10 50 80 for C1, C2, C3
mse_jpeg1_c1 = mse(C1 + '_gray.png', C1 + '_1.jpeg')
mse_jpeg1_c2 = mse(C2 + '_gray.png', C2 + '_1.jpeg')
mse_jpeg1_c3 = mse(C3 + '_gray.png', C3 + '_1.jpeg')

mse_jpeg10_c1 = mse(C1 + '_gray.png', C1 + '_10.jpeg')
mse_jpeg10_c2 = mse(C2 + '_gray.png', C2 + '_10.jpeg')
mse_jpeg10_c3 = mse(C3 + '_gray.png', C3 + '_10.jpeg')

mse_jpeg50_c1 = mse(C1 + '_gray.png', C1 + '_50.jpeg')
mse_jpeg50_c2 = mse(C2 + '_gray.png', C2 + '_50.jpeg')
mse_jpeg50_c3 = mse(C3 + '_gray.png', C3 + '_50.jpeg')

mse_jpeg80_c1 = mse(C1 + '_gray.png', C1 + '_80.jpeg')
mse_jpeg80_c2 = mse(C2 + '_gray.png', C2 + '_80.jpeg')
mse_jpeg80_c3 = mse(C3 + '_gray.png', C3 + '_80.jpeg')

data_ratio = [[ratio_our_c1, ratio_jpeg1_c1, ratio_jpeg10_c1, ratio_jpeg50_c1, ratio_jpeg80_c1],
              [ratio_our_c2, ratio_jpeg1_c2, ratio_jpeg10_c2, ratio_jpeg50_c2, ratio_jpeg80_c2],
              [ratio_our_c3, ratio_jpeg1_c3, ratio_jpeg10_c3, ratio_jpeg50_c3, ratio_jpeg80_c3]]

data_mse = [[mse_our_c1, mse_jpeg1_c1, mse_jpeg10_c1, mse_jpeg50_c1, mse_jpeg80_c1],
            [mse_our_c2, mse_jpeg1_c2, mse_jpeg10_c2, mse_jpeg50_c2, mse_jpeg80_c2],
            [mse_our_c3, mse_jpeg1_c3, mse_jpeg10_c3, mse_jpeg50_c3, mse_jpeg80_c3]]


columns = ['Our Algorithm', 'JPEG1 ', 'JPEG10 ', 'JPEG50 ', 'JPEG80 ']
rows = ['C1', 'C2', 'C3']

# print MSE values:
dataframe_mse = pd.DataFrame(data_mse, rows, columns)
print("MSE values of images: \n", dataframe_mse, "\n")

# print compression ratios:
dataframe_ratios = pd.DataFrame(data_ratio, rows, columns)
print("Compression ratios of images: \n", dataframe_ratios)