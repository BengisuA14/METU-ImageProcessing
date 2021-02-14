# Bengisu Ayan
# e2236974
# Ceren Gürsoy
# e2237485

import numpy as np
import cv2


def gaussian_kernel(size, sigma):
    size = int(size / 2)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]

    cons = 1 / (np.sqrt(2.0 * np.pi) * sigma)
    gaus = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * cons
    return gaus/np.sum(gaus)

def conv(image, kernel):
    height = image.shape[0]
    width = image.shape[1]
    new_image = np.zeros([height, width])

    new_height = height + kernel.shape[0] - 1
    new_width = width + kernel.shape[1] - 1
    imagePadded = np.zeros([new_height, new_width])  # for zero padding

    h_index = int((kernel.shape[0] - 1) / 2)
    w_index = int((kernel.shape[1] - 1) / 2)

    imagePadded[h_index: height + h_index,
    w_index: width + w_index] = image  # old image is placed into padded image

    for i in range(height):
        for k in range(width):
            multip = np.multiply(kernel, imagePadded[i: i + kernel.shape[0], k: k + kernel.shape[1]])
            summ = np.sum(multip)
            new_image[i, k] = summ
    return new_image


def sobel(image):
    w_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)  # vertical filter
    w_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)  # horizontal filter

    image_vertical = conv(image, w_v)
    image_horizontal = conv(image, w_h)

    gradient_magnitude = np.sqrt(np.square(image_vertical) + np.square(image_horizontal))
    gradient_magnitude_norm = 255 * (gradient_magnitude - gradient_magnitude.min()) \
                              / (gradient_magnitude.max() - gradient_magnitude.min())

    theta = np.arctan2(image_vertical, image_horizontal)

    return gradient_magnitude_norm, theta


def non_maximal_suppression(edge_map, edge_dir):
    edge_dir = np.rad2deg(edge_dir)
    new_edge_map = edge_map

    for i in range(1, edge_map.shape[0] - 1):
        for j in range(1, edge_map.shape[1] - 1):
            g_vec = edge_dir[i][j]
            pixel = edge_map[i][j]
            if (-22.5 <= g_vec < 22.5) or (157.5 <= g_vec <= 180) or (-180 <= g_vec < -157.5):
                q = edge_map[i + 1][j]
                r = edge_map[i - 1][j]
            elif (22.5 <= g_vec < 67.5) or (-157.5 <= g_vec < -112.5):
                q = edge_map[i + 1][j - 1]
                r = edge_map[i - 1][j + 1]
            elif (67.5 <= g_vec < 112.5) or (-112.5 <= g_vec < -67.5):
                q = edge_map[i][j + 1]
                r = edge_map[i][j - 1]
            elif (112.5 <= g_vec < 157.5) or (-67.5 <= g_vec < -22.5):
                q = edge_map[i - 1][j - 1]
                r = edge_map[i + 1][j + 1]

            if (pixel < q) or (pixel < r):
                new_edge_map[i][j] = 0

    return new_edge_map


def is_connected(i, k, gNH, connectivity):
    if connectivity == 4:
        if ((gNH[i + 1][k] != 0) or (gNH[i - 1][k] != 0) or (gNH[i][k + 1] != 0) or (gNH[i][k - 1] != 0)):
            return True
        else:
            return False
    elif connectivity == 6:
        if ((gNH[i + 1][k + 1] != 0) or (gNH[i - 1][k - 1] != 0) or (gNH[i - 1][k + 1] != 0) or (
                gNH[i + 1][k - 1] != 0)):
            return True
        else:
            return False
    elif connectivity == 8:
        if ((gNH[i + 1][k] != 0) or (gNH[i - 1][k] != 0) or (gNH[i][k + 1] != 0) or (gNH[i][k - 1] != 0)
                or (gNH[i + 1][k + 1] != 0) or (gNH[i - 1][k - 1] != 0) or (gNH[i - 1][k + 1] != 0) or (
                        gNH[i + 1][k - 1] != 0)):
            return True
        else:
            return False
    else:
        print("Connectivity value must be 4, 6 or 8.")


def double_thresholding(suppress, T_H, T_L):
    # initially, both gNH(x, y) and gNL(x, y) are set to 0.
    gNH = np.zeros(suppress.shape)
    gNL = np.zeros(suppress.shape)

    for i in range(suppress.shape[0]):
        for k in range(suppress.shape[1]):
            if suppress[i][k] >= T_H:
                gNH[i][k] = suppress[i][k]
            if suppress[i][k] >= T_L and suppress[i][k] < T_H:
                gNL[i][k] = suppress[i][k]

    # nonzero pixels in gNH are strong and in gNL are weak

    gNH[gNH > 0] = 255

    gNL_valid = np.zeros(gNL.shape)  # Set to zero all pixels in gNL(x, y) that were not marked as valid edge pixels.

    for i in range(1, gNL.shape[0] - 1):
        for k in range(1, gNL.shape[1] - 1):
            if gNL[i][k] != 0:
                if is_connected(i, k, gNH, 8):
                    gNL_valid[i][
                        k] = 9999  # Mark as valid (random number which is 9999) edge pixels all the weak pixels in gNL(x, y) that are connected to p using, say, 8-connectivity.

    # Set to zero all pixels in gNL(x, y) that were not marked as valid edge pixels.
    for i in range(gNL_valid.shape[0]):
        for k in range(gNL_valid.shape[1]):
            if gNL_valid[i][k] != 9999:
                gNL[i][k] = 0
            else:
                gNL[i][k] = 255

    #  append to gNH(x, y) all the nonzero pixels from gNL(x, y).
    for i in range(gNL.shape[0]):
        for k in range(gNL.shape[1]):
            if gNL[i][k] != 0 and gNH[i][k] == 0:
                gNH[i][k] = gNL[i][k]

    return gNH


def canny(image, kernel_size, sigma, threshold_h, threshold_l, name, blurring=True):
    g_filter = gaussian_kernel(kernel_size, sigma)

    if blurring:
        image = conv(image, g_filter)
  
    edge_map, edge_dir = sobel(image)
    suppress = non_maximal_suppression(edge_map, edge_dir)
    d_tresh = double_thresholding(suppress, threshold_h, threshold_l)
    new_image = d_tresh

    return new_image

def sobel_edge_detection(image, kernel_size, sigma, name, blurring=True):
    g_filter = gaussian_kernel(kernel_size, sigma)

    if blurring:
        image = conv(image, g_filter)

    edge_map, edge_dir = sobel(image)
    
    return edge_map


def main():
    B1 = cv2.imread("B1.jpg", cv2.IMREAD_GRAYSCALE)
    B2 = cv2.imread("B2.jpg", cv2.IMREAD_GRAYSCALE)
    B3 = cv2.imread("B3.jpg", cv2.IMREAD_GRAYSCALE)

    kernel_size = 9
    sigma = 5

    highThreshold = 22
    lowThreshold = 15

    B1_canny = canny(B1, kernel_size, sigma, highThreshold, lowThreshold, True)
    cv2.imwrite("B1_output.jpg", B1_canny)

    highThreshold = 20
    lowThreshold = 5

    B2_canny = canny(B2, kernel_size, sigma, highThreshold, lowThreshold, True)
    cv2.imwrite("B2_output.jpg", B2_canny)

    highThreshold = 30
    lowThreshold = 20
    B3_canny = canny(B3, kernel_size, sigma, highThreshold, lowThreshold, True)
    cv2.imwrite("B3_output.jpg", B3_canny)


main()