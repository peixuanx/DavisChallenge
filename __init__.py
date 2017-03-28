from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys, os

import fcn16_vgg
import read_pascal
import scipy.misc

from config import *
