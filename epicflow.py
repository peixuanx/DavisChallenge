from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from scipy import ndimage
from config import *
import numpy as np
import sys

# import matlab engine
sys.path.append( '/home/erhsin/lib' )
import matlab.engine as mateng

# return concatenated image with opticalflow information
def computeOpticalFlow(img1, img2):

    # initiate matlab engine
    matlab_eng = mateng.start_matlab()

    # add working path
    # TODO: the output concatenate file won't contain labels
    # TODO: normalize motion vector ?
    matlab_eng.addpath(MATLAB_PATH)
    out = matlab_eng.epicflow(img1, img2)
    [flow_file, edge_file, match_file] = out.split('$')

    # concatenate with image
    img = misc.imread(img2)
    (H,W,C) = np.shape(img)
    img_e = misc.imread(edge_file, mode='L')
    img_e = np.reshape(img_e, (H,W,1))
    img_m = readMatchFile(match_file, H, W)
    img_concat = np.concatenate((img, img_e, img_m), axis=2)

    return img_concat

# parse match file to image 
def readMatchFile(match_file, H, W):

    # stack y/x motion vectors into 2 channels
    img_m = np.zeros((H,W,2))
    f = open(match_file)
    for line in f:
        [x_s, y_s, x_e, y_e, _, _] = line.split()
        img_m[y_e, x_e, 0] = int(y_e)-int(y_s)
        img_m[y_e, x_e, 1] = int(x_e)-int(x_s)

    return img_m

# main for demo
if __name__=='__main__':
    computeOpticalFlow('../1.jpg', '../2.jpg')
