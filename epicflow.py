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
sys.path.append( '/home/pychien/lib/lib' )
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
    img_concat = np.concatenate((img_e, img_m), axis=2)

    return img_concat

# affine transfrom for optical flow
#
# img:  input image, 2 channels: [magnitude; pole]
# rot:  rotation in degree
# sc:   scaling factor (1.0==same)
# flip: 0:same, 1:ud, 2:lr, 3:udlr
#
def affine(img, rot, sc, flip):

    # progate affine transform
    img_origin = img[:,:,:]
    img[:,:,0] = img[:,:,0] * sc
    img[:,:,1] = img[:,:,1] + rot
    if flip==1:
        img[:,:,1] = img[:,:,1] * -1
    elif flip==2:
        img[:,:,1] = 180 - img[:,:,1]
    elif flip==3:
        img[:,:,1] = 180 + img[:,:,1]

    # normalize degree
    while np.any(img[:,:,1]>180) or np.any(img[:,:,1]<-180):
        img[:,:,1] = np.where(img[:,:,1]<=180, img[:,:,1], img[:,:,1]-360)
        img[:,:,1] = np.where(img[:,:,1]>=-180, img[:,:,1], img[:,:,1]+360)

    # restore zeros
    img[:,:,1] = np.where(img[:,:,0]==0, 0, img[:,:,1])

    return img

# parse match file to image
def readMatchFile(match_file, H, W):

    # stack y/x motion vectors into 2 channels
    img_m = np.zeros((H,W,2))
    f = open(match_file)
    for line in f:
        [x_s, y_s, x_e, y_e, _, _] = line.split()
        diff_y = int(y_e)-int(y_s)
        diff_x = int(x_e)-int(x_s)
        img_m[y_e, x_e, 0] = np.linalg.norm(np.array([diff_y,diff_x]))
        img_m[y_e, x_e, 1] = np.arctan2(diff_y,diff_x) * 180/np.pi

    return img_m

# main for demo
if __name__=='__main__':
    img_concat = computeOpticalFlow('../1.jpg', '../2.jpg')
    img_m = affine(img_concat[:,:,4:], 400, 0.8, 3)
