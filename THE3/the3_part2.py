# Bengisu Ayan - 2236974
# Ceren Gürsoy - 2237485

import numpy as np 
import cv2 

B1 = "THE3-Images/B1.jpg"
B2 = "THE3-Images/B2.jpg"
B3 = "THE3-Images/B3.jpg"
B4 = "THE3-Images/B4.jpg"
B5 = "THE3-Images/B5.jpg"


def segmentation_function(image, name,  blue_mask=False, white_mask=False, yellow_mask=False):
    # Smooth the image
    image = cv2.GaussianBlur(image,(11,11),0)

    # convert to HSV color system
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # define green mask
    # green is applied to all the images as the dogs are all in grass
    low_green = np.array([30, 0, 0])
    high_green = np.array([86, 255, 255])
    mask_final = cv2.inRange(hsv_image, low_green, high_green)

    # define blue mask and apply it
    if blue_mask == True:
        low_blue = np.array([80, 0, 0])
        high_blue = np.array([125, 255, 255])
        mask_blue = cv2.inRange(hsv_image, low_blue, high_blue)
        mask_final = mask_final + mask_blue

    # define white mask and apply it
    if white_mask == True:
        low_white = np.array([0, 0, 200])
        high_white = np.array([145,60,255])
        mask_white = cv2.inRange(hsv_image, low_white, high_white)
        mask_final = mask_final + mask_white

    # define yellow mask and apply it
    if yellow_mask == True:
        low_yellow = np.array([10, 0, 0])
        high_yellow = np.array([33, 255, 100])
        mask_yellow = cv2.inRange(hsv_image, low_yellow, high_yellow)
        mask_final = mask_final + mask_yellow

    # make object white and background black
    mask_final = 255 - mask_final

    # apply opening to final mask
    # define 27*27 elliptical structuring element for opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27))
    # apply opening
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)

    # apply closing to final mask
    # define 41*41 elliptical structuring element for opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
    # apply closing
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)

    # get biggest connected component
    segmented_image = np.zeros_like(mask_final)                                        
    for val in np.unique(mask_final)[1:]:                                     
        mask = np.uint8(mask_final == val)                                     
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  
        largest_label = 1 +  np.argmax(stats[1:, cv2.CC_STAT_AREA])      
        segmented_image[labels == largest_label] = val                          

    return segmented_image


  
  
# Read B1
image_bgr = cv2.imread(B1)
# convert B1 from bgr to rgb
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Apply segmentation to B1 with only green mask
segmented_image = segmentation_function(image, 'B1', blue_mask=False, white_mask=False, yellow_mask=False)
cv2.imwrite("the3_B1_output.png", segmented_image)

# Read B2
image_bgr = cv2.imread(B2)
# convert B2 from bgr to rgb
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Apply segmentation to B2 with green, white, yellow and blue masks
segmented_image = segmentation_function(image,'B2', blue_mask=True, white_mask=True, yellow_mask=True)
cv2.imwrite("the3_B2_output.png", segmented_image)

# Read B3
image_bgr = cv2.imread(B3)
# convert B3 from bgr to rgb
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Apply segmentation to B3 with green and blue masks
segmented_image = segmentation_function(image,'B3', blue_mask=True, white_mask=False, yellow_mask=False)
cv2.imwrite("the3_B3_output.png", segmented_image)

# Read B4
image_bgr = cv2.imread(B4)
# convert B4 from bgr to rgb
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Apply segmentation to B4 with green and blue masks
segmented_image = segmentation_function(image,'B4', blue_mask=True, white_mask=False, yellow_mask=False)
cv2.imwrite("the3_B4_output.png", segmented_image)

# Read B5
image_bgr = cv2.imread(B5)
# convert B5 from bgr to rgb
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Apply segmentation to B2 with green ,yellow and blue masks
segmented_image = segmentation_function(image,'B5', blue_mask=True, white_mask=False, yellow_mask=True)
cv2.imwrite("the3_B5_output.png", segmented_image)


