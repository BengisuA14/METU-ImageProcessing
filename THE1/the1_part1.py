# Bengisu Ayan
# e2236974
# Ceren Gürsoy
# e2237485

import numpy as np
import matplotlib.pyplot as plt
import cv2

A1 = "A1.jpg"
A2 = "A2.jpg"

def plot_histograms(image, title):
    blue, green, red = cv2.split(image)

    data = [blue.flatten(), green.flatten(), red.flatten()]
    titles = ["blue", "green", "red"]
    f,a = plt.subplots(3,1,figsize=(5,7))
    a = a.ravel()
    for idx,ax in enumerate(a):
        ax.set_ylim([0,25000])
        ax.set_xlim([0,255])
        ax.hist(data[idx], 256, [0,256])
        ax.set_title("Histogram of " + titles[idx] +" " + title)
        ax.set_xlabel("RGB Values")
        ax.set_ylabel("Number of pixels")
    plt.tight_layout()
    plt.savefig(title + "_histmatch.jpg")
    plt.close()


# Find the histogram h(rk) 
def get_histogram(image, bins):
    hist = np.zeros(bins)
    
    for pixel in image:
        hist[pixel] += 1
    
    return hist

# Find cumulative histogram hc(rk) 
def cumsum(hist, bins):
    cdf = hist.copy()
    for i in range(bins-1):
        cdf[i+1] += cdf[i]
    return cdf

# T(rk) = round(L-1/NM)hc(rk) 
def transform(input_cdf, ref_cdf, nm_source, nm_ref):
    t_input = np.floor(255 * input_cdf / nm_source)
    t_ref = np.floor(255 * ref_cdf / nm_ref)
    return t_input, t_ref

# create a lookup table indicating which values of input_hist correspond to values of target_hist
def create_lookup(input_hist, target_hist):
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(256):
        for ref_pixel_val in range(256):
            if target_hist[ref_pixel_val] >= input_hist[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table

#histogram matching algorithm for each color
def histogram_match_color(source, ref, nm_source, nm_ref):
    input_hist = get_histogram(source.flatten(), 256)
    input_cdf = cumsum(input_hist, 256)

    ref_hist = get_histogram(ref.flatten(), 256)
    ref_cdf = cumsum(ref_hist, 256)

    t_input, t_ref = transform(input_cdf, ref_cdf, nm_source, nm_ref)
    #create new image from lookup table
    after_transform = np.zeros_like(source)
    lookup = create_lookup(t_input, t_ref)

    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            pixel_val = source[x][y]
            after_transform[x][y] = lookup[pixel_val]
    return after_transform

def match_histograms(src_image, ref_image):
    nm_source = src_image.shape[0]*src_image.shape[1]
    nm_ref = ref_image.shape[0]*ref_image.shape[1]

    source_blue, source_green, source_red = src_image[:, :, 0], src_image[:, :, 1], src_image[:, :, 2]
    ref_blue, ref_green, ref_red = ref_image[:, :, 0], ref_image[:, :, 1], ref_image[:, :, 2]

    red_after_transform = histogram_match_color(source_red, ref_red, nm_source, nm_ref)
    green_after_transform = histogram_match_color(source_green, ref_green, nm_source, nm_ref)
    blue_after_transform = histogram_match_color(source_blue, ref_blue, nm_source, nm_ref)

    image_after_matching = np.dstack((blue_after_transform, green_after_transform, red_after_transform))

    return image_after_matching


def main():
    
    a1 = cv2.imread(cv2.samples.findFile(A1))
    a2 = cv2.imread(cv2.samples.findFile(A2))

    output1 = match_histograms(a1, a2)
    plot_histograms(output1, "A1")
    cv2.imwrite("A1_histmatch_output.jpg", output1)

    output2 = match_histograms(a2, a1)
    plot_histograms(output2, "A2")
    cv2.imwrite("A2_histmatch_output.jpg", output2)



main()
