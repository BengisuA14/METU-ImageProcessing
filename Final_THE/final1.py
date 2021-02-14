import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import label2rgb, lab2rgb
from skimage import segmentation, filters, color
from skimage.future import graph
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import math
from skimage.segmentation import clear_border
import mahotas
import os


def final_q1(input_file_path, output_path):
    i=1
    for filename in os.listdir(input_file_path):
        img = cv2.imread(os.path.join(input_file_path,filename),0)
        step1=q1_step1(img,input_file_path, output_path, str(i))
        step2 = q1_step2(step1,input_file_path, output_path, str(i))
        i = i+1

def q1_step1(img,image_path, output_path,k):

    # adaptive thresholding
    image = cv2.medianBlur(img,3)
    image = cv2.GaussianBlur(image,(5,5),0)
    th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    
    # inverse image if necessary by checking background color
    inv = cv2.bitwise_not(th3)

    # clear border
    clearborder = clear_border(inv)

    cropped_sides = crop_sides(clearborder)

    cropped_sides = cropped_sides.astype(np.uint8)

    # remove small objects
    small_objects_removed = bwareaopen(cropped_sides, 30)


    left, top, right, down = find_boundaries(small_objects_removed)
    otsu = otsu_th(img, left, top, right, down)
    otsu = otsu.astype(np.uint8)

    cv2.imwrite(output_path + image_path[:-4]+k+"_step1.png", otsu)


    return otsu

def find_boundaries(img):

    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]


    left = img.shape[1]
    right = 0
    top = img.shape[0]
    down = 0

    for i in range(num_labels):
        row = stats[i][1]
        col = stats[i][0]
        width = stats[i][2]
        height = stats[i][3]
        if row != 0 and col!=0 and width < img.shape[1] and height < img.shape[0]:
            if row+height > down:
                down = row+height
            if row < top:
                top = row      
            if col < left:
                left = col
            if col+width > right:
                right = col + width

    rect = cv2.rectangle(img, (left, top), (right, down), (255,0,0) , 1)
    
    return left, top, right, down

def otsu_th(img, left, top, right, down):
    cropped = img[top:down,left:right]
    ret2,th2 = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if th2[0][0] == 255:
        th2 = cv2.bitwise_not(th2)

    new_img = np.zeros(img.shape)
    new_img[top:down,left:right] = th2
    return new_img

def skeletonize(img):

    skel = img.copy()
    img = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel



def q1_step2(img, image_path, output_path,k):
    colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255)]

    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]

    boxes = []
    for i in range(num_labels):
        row = stats[i][0]
        col = stats[i][1]
        width = stats[i][2]
        height = stats[i][3]
        if(row != 0 and col != 0):
            boxed = img[col:col+height,row:row+width]
            num_labels, labels_im = cv2.connectedComponents(boxed)
            while num_labels > 2:
                kernel = np.ones((2,2),np.uint8)
                boxed = cv2.dilate(boxed,kernel,iterations = 1)
                num_labels, labels_im = cv2.connectedComponents(boxed)
            boxes.append(boxed)
            img[col:col+height,row:row+width] = boxed

    rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    thinned = img.copy()

    for i in range(num_labels):
        row = stats[i][0]
        col = stats[i][1]
        width = stats[i][2]
        height = stats[i][3]
        if(row != 0 and col != 0):
            rgb = cv2.rectangle(rgb, (row, col), (row+width, col+height), colors[i%3] , 1)

            skeleton = skeletonize(img[col:col+height,row:row+width])
            thinned[col:col+height,row:row+width] = skeleton

            # need to check more, dont thing necessary
            num_labels, labels_im = cv2.connectedComponents(img[col:col+height,row:row+width])
            e = mahotas.euler(labels_im) 
            if e < 0:
                print("Class 1 numeral in " + image_path)
            else:
                print("Class 2 numeral in " + image_path)



    cv2.imwrite(output_path + image_path[:-4]+k+"_step3.png", thinned)
    cv2.imwrite(output_path + image_path[:-4]+k+"_step2.png", rgb)
    return boxes

 

def crop_sides(img):
    x = img.shape[0]
    y = img.shape[1]
    crop_ratio = 0.15
    crop_y_percentage = int(y*crop_ratio)
    crop_x_percentage = int(x*crop_ratio)
    cropped_image = np.zeros((x,y))
    cropped_image[crop_x_percentage:(x-crop_x_percentage),crop_y_percentage:(y-crop_y_percentage)] = img[crop_x_percentage:(x-crop_x_percentage),crop_y_percentage:(y-crop_y_percentage)]

    return cropped_image


def final_q2(image_path, output_path):
    
    #Loading original image
    originImg = cv2.imread(image_path)
    originImg = cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB)
    # lab_image = cv2.cvtColor(originImg, cv2.COLOR_RGB2Lab)
    r, g, b = cv2.split(originImg)

    # Shape of original image    
    originShape = originImg.shape

    x, y = np.mgrid[0:originShape[0],0:originShape[1]]

    new5dImage = cv2.merge((r,g,b,x.astype(np.uint8),y.astype(np.uint8)))

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities    
    flatImg_shape = np.reshape(new5dImage, [-1, 5])

    ms = MeanShift(bandwidth=30.0, bin_seeding=True, max_iter=100)

    # Performing meanshift on flatImg    
    ms.fit(flatImg_shape)

    # (r,g,b) vectors corresponding to the different clusters after meanshift    
    labels=ms.labels_

    # Remaining colors after meanshift    
    cluster_centers = ms.cluster_centers_    

    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)    
 

    # Displaying segmented image    
    segmentedImg = np.reshape(labels, originShape[:2])
    superpixels=label2rgb(segmentedImg,originImg,kind='avg',bg_label=-1,)

    rgb_image =  cv2.cvtColor(superpixels, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path + image_path[:-4]+"_step1.png", rgb_image)
    

    # define rag
    texture_graph =  texture_rag(originImg, segmentedImg)

    show_rag = graph.show_rag(segmentedImg, texture_graph, originImg)
    cbar = plt.colorbar(show_rag)
    plt.savefig(output_path + image_path[:-4]+"_step2.png")

    labels2 = graph.cut_normalized(segmentedImg, texture_graph)
    out2 = label2rgb(labels2, originImg,bg_label=-1,kind='avg')

    cv2.imwrite(output_path + image_path[:-4]+"_step3.png", cv2.cvtColor(out2, cv2.COLOR_RGB2BGR))



def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def texture_rag(image, labels):
       #initialize the RAG
    g = graph.RAG(labels, connectivity=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    _, _, v = cv2.split(hsv)

    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(filters.gabor_kernel(frequency, theta=theta,
                                            sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


   #lets say we want for each node on the graph a label, a pixel count and a total color 
    for n in g:
        g.nodes[n].update({'labels': [n],'pixels': np.zeros(labels.shape), 'gabor':0, 'i':0,'locationx': []
        , 'locationy':[], 'centerx':0, 'centery':0})


    for index in np.ndindex(labels.shape):
        current = labels[index]
        g.nodes[current]['pixels'][index[0]][index[1]] = v[index[0]][index[1]]
        g.nodes[current]['i'] = v[index[0]][index[1]]
        g.nodes[current]['locationx'].append(index[0])
        g.nodes[current]['locationy'].append(index[1])


    for n in g:
        g.nodes[n]['centerx'] = np.mean(np.asarray(g.nodes[n]['locationx']))
        g.nodes[n]['centery'] = np.mean(np.asarray(g.nodes[n]['locationy']))
     

   #calculate your own weights here
    for x, y, d in g.edges(data=True):
        ix = g.nodes[x]['i']
        iy = g.nodes[y]['i']
        shrink = (slice(0, None, 3), slice(0, None, 3))
        p_x = img_as_float(g.nodes[x]['pixels'])[shrink]
        gx = compute_feats(p_x, kernels) * g.nodes[y]['i'] * ix
        p_y = img_as_float(g.nodes[y]['pixels'])[shrink]
        gy = compute_feats(p_y, kernels) * g.nodes[y]['i'] *iy
        error = np.sum(abs(gx - gy))/1000000.0
        texture_dist = np.exp(-error)

        spatial_dist = math.sqrt((g.nodes[x]['centerx'] - g.nodes[y]['centerx'])**2 + (g.nodes[x]['centery'] - g.nodes[y]['centery'])**2)
        if spatial_dist > 100:
            spatial_dist = 0
        else:
            spatial_dist = spatial_dist / 100000.0
            spatial_dist = np.exp(-spatial_dist)

        similarity = texture_dist* spatial_dist

        d['weight'] = similarity
    return g



def final_q3(image_path, output_path):
    image = cv2.imread(image_path)
    # blurred = cv2.GaussianBlur(image,(3,3),0)
        # convert to HSV color system
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # define green mask
    # green is applied to all the images as the dogs are all in grass
    low_green = np.array([35, 0, 0])
    high_green = np.array([86, 255, 255])
    mask_final = cv2.inRange(hsv_image, low_green, high_green)

    # define blue mask and apply it
    low_blue = np.array([80, 0, 0])
    high_blue = np.array([125, 255, 255])
    mask_blue = cv2.inRange(hsv_image, low_blue, high_blue)
    mask_final = mask_final + mask_blue

    # # define white mask and apply it

    low_white = np.array([0, 0, 200])
    high_white = np.array([145,60,255])
    mask_white = cv2.inRange(hsv_image, low_white, high_white)
    mask_final = mask_final + mask_white


    # make object white and background black
    mask_final = 255 - mask_final

    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    small_objects_removed = bwareaopen(opening,150)


    edges = cv2.Canny(small_objects_removed, 50, 200, None, 3)

    cv2.imwrite(output_path + image_path[:-4]+"_step1.png", edges)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    newimage = np.copy(image)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 40, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            cv2.line(newimage, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    cv2.imwrite(output_path + image_path[:-4]+"_step2.png", cdstP)
    cv2.imwrite(output_path + image_path[:-4]+"_step3.png", newimage)

def bwareaopen(imgBW, areaPixels):
        # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy

final_q1("Dataset1/","final/Dataset2/" )
final_q2("Dataset2/1.jpg","final/")