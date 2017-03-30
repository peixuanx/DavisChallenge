import Data_Distor
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave

d = Data_Distor.Data_Distor(imread('00000.png'))
affineMask, mask = d.genMasks()
imsave('0_affine.png',affineMask[0])
imsave('0.png',mask[0])
imsave('1_affine.png',affineMask[1])
imsave('1.png',mask[1])


